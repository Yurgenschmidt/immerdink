#!/usr/bin/env python3
"""Summarise the k45 job (run locally on the copied-back out-h200-k45/ directory, stdlib only):
  python3 -I k45_tab.py <out-h200-k45 dir>
1. K45 single-warp CRT gate. For each gate variant and shape:
     eff_ipc = (ref K5 warp-instructions / 528 SMSPs) / (gate cycles),  gate cycles = gate ms x SM clock (clock64/globaltimer probe)
   i.e. the IPC the one CRT warp per SMSP must reach to retire TODAY's K5 work in the gate's time. Gate: eff_ipc >= 0.375 at
   r_lowk and >= 0.23 at t_df (plan.md §6: = K5 work per SMSP over the K4 time, so equivalently gate ms <= K4 ms).
   The ref instruction count comes from ncu (smsp__inst_executed.sum of k5f_crt_n) when present, else the plan's model.
   Also: ncu-measured issue IPC of the gate itself (its own instructions per active cycle), the ref time against the known K5
   time 0.708 ms at r_lowk (+-10 %), and the L1 sensitivity (pad 120 vs 200 KiB).
2. G2b comparators: per case, every method by median time (TFLOPS-eq first); the G2 comparator = the minimum complete native
   pipeline (cuTENSOR, fix55 emu, emumath DSYRK + mirror if it routes through emulation, DSYRK + mirror / mirror_permute); the kill
   bar = 0.95 x that minimum, against the immerdink SYRK v1 model (plan.md §4.2).
"""
import csv, json, os, sys
from collections import defaultdict

SMSP = 528
GATE = {"r_lowk": 0.375, "t_df": 0.23}
K4_MS = {"r_lowk": 1.58, "t_df": 0.665}                   # plan.md: the K4 time the CRT warp must hide under (printed)
REF_INST_MODEL = {"r_lowk": 547e6, "t_df": 137e6}         # K5 warp-instructions, plan.md (261 thread-inst per output)
REF_MS_KNOWN = {"r_lowk": 0.708}
IMD_SYRK_MODEL = {"t_df": 0.685, "r_lowk": 1.513, "r1_q2048": 0.70}   # ms, plan.md §4.2 (R1 at t_df size: 0.69-0.71)
SHAPE_ORDER = ["r_lowk", "t_df"]


def jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def ncu_gate(path):
    """ncu raw CSV -> list of dicts (one per profiled launch, in launch order); numeric metric columns as floats."""
    if not os.path.exists(path):
        return []
    rows = list(csv.reader(open(path)))
    if len(rows) < 3:
        return []
    hdr = rows[0]
    out = []
    for r in rows[2:]:   # row 1 holds units
        if len(r) != len(hdr):
            continue
        d = dict(zip(hdr, r))
        for k, v in list(d.items()):
            try:
                d[k] = float(v.replace(",", ""))
            except ValueError:
                pass
        out.append(d)
    return out


def gate_section(o):
    ncu = ncu_gate(os.path.join(o, "ncugate", "gate.raw.csv"))
    # launch order of k45_gate --mode ncu: per shape (r_lowk, t_df): ref, E1, E2, E2pf, E4, E4pf, E4x2
    names = ["ref k5f_crt_n<14,4> 256thr", "k45g E1", "k45g E2", "k45g E2pf", "k45g E4", "k45g E4pf", "k45g E4x2"]
    nc = {}
    if ncu and len(ncu) != 14:
        print(f"WARNING: {len(ncu)} ncu rows, expected 14 (2 shapes x 7 kernels): ncu columns below are NOT trusted"); ncu = []
    for i, d in enumerate(ncu):
        sh, v = SHAPE_ORDER[i // 7] if i // 7 < 2 else "?", names[i % 7]
        nc[(sh, v)] = d
    ref_inst = {}
    for sh in SHAPE_ORDER:
        d = nc.get((sh, names[0]))
        ref_inst[sh] = (d.get("smsp__inst_executed.sum"), "ncu") if d and isinstance(d.get("smsp__inst_executed.sum"), float) \
            else (REF_INST_MODEL[sh], "model")
    print("== K45 single-warp CRT gate (one warp per SMSP, <= 96 regs). eff_ipc = ref K5 warp-inst / 528 / gate cycles")
    for pad in (120, 200):
        recs = [r for r in jsonl(os.path.join(o, "gate", f"gate-pad{pad}.jsonl")) if "variant" in r]
        checks = [r for r in jsonl(os.path.join(o, "gate", f"gate-pad{pad}.jsonl")) if r.get("check")]
        if not recs:
            print(f"  pad {pad} KiB: no records"); continue
        print(f"-- pad {pad} KiB dynamic smem (L1 left ~{max(0, 256 - 4 - pad - 32)} KiB); host CRT check: "
              + ", ".join(f"{c['shape']} {c['mismatches']}/{c['samples']}" for c in checks))
        print(f"  {'shape':7} {'variant':28} {'regs':>4} {'exact':>5} {'SMs':>4} {'TF-eq':>7} {'ms_med':>8} {'p10-p90':>15} {'GHz':>5}"
              f" {'eff_ipc':>7} {'need':>5} {'pass':>4} {'ncu_ipc':>7} {'issue%':>6}")
        for r in recs:
            sh = r["shape"]
            gate = r["threads"] == 128
            ghz = r["sm_ghz_probe"] or 1.755
            ri, src = ref_inst[sh]
            eff = (ri / SMSP) / (r["ms_med"] * 1e-3 * ghz * 1e9) if gate and r["ms_med"] > 0 else None
            d = nc.get((sh, r["variant"])) if pad == 120 else None
            nipc = iss = None
            if d and isinstance(d.get("smsp__inst_executed.sum"), float) and isinstance(d.get("smsp__cycles_active.avg"), float):
                nipc = d["smsp__inst_executed.sum"] / SMSP / d["smsp__cycles_active.avg"]
                iss = d.get("smsp__issue_active.avg.pct_of_peak_sustained_active")
            ok = "" if eff is None else ("PASS" if eff >= GATE[sh] else "FAIL")
            extra = ""
            if not gate and sh in REF_MS_KNOWN:
                dev = r["ms_med"] / REF_MS_KNOWN[sh] - 1
                extra = f"  ref vs known {REF_MS_KNOWN[sh]} ms: {dev:+.1%} ({'ok' if abs(dev) <= 0.10 else 'OUTSIDE 10%'})"
            print(f"  {sh:7} {r['variant']:28} {r['regs']:4d} {r['bitexact_vs_ref']:5d} {r['distinct_sms']:4d} {r['tflops_eq_stage']:7.1f}"
                  f" {r['ms_med']:8.4f} {r['ms_p10']:7.4f}-{r['ms_p90']:<7.4f} {ghz:5.2f}"
                  f" {'' if eff is None else f'{eff:7.3f}':>7} {GATE[sh] if gate else '':>5} {ok:>4}"
                  f" {'' if nipc is None else f'{nipc:7.3f}':>7} {'' if iss is None else f'{iss:6.1f}':>6}{extra}")
        print(f"  ref instruction source: " + ", ".join(f"{sh} {ref_inst[sh][0]/1e6:.0f}M ({ref_inst[sh][1]})" for sh in SHAPE_ORDER)
              + f"; K4 budget ms: {K4_MS}")
    stalls = os.path.join(o, "ncugate", "gate.stalls.csv")
    if os.path.exists(stalls):
        print(f"  warp-stall breakdown: see {stalls} (WarpStateStats; top reasons per kernel)")


def g2b_section(o):
    print("\n== G2b native comparators (TFLOPS-eq = dense 2 n^2 k / median; same job)")
    rounds = {rd: jsonl(os.path.join(o, rd, "g2b.jsonl")) for rd in ("g2b", "g2b2")}
    imdb = jsonl(os.path.join(o, "imdb", "imdb.jsonl"))
    cur = None
    for r in imdb:   # imd_bench: the case name sits in its meta record ("case"); timed records follow it ("method", "median_s")
        cur = r.get("case", cur)
        r["_case"] = r.get("case", cur)
    by = defaultdict(lambda: defaultdict(list))
    meta = {}
    for rd, recs in rounds.items():
        for r in recs:
            if r.get("meta"):
                meta = r
            elif r.get("check") == "emulation_routing":
                print(f"  [{rd}] {r['case']}: emumath-DSYRK vs DSYRK bits {r['emumath_dsyrk_vs_dsyrk_bits']} (rel {r['emumath_dsyrk_vs_dsyrk_rel']:.2e}),"
                      f" vs emu55 bits {r['emumath_dsyrk_vs_emu55_bits']} (rel {r['emumath_dsyrk_vs_emu55_rel']:.2e}); native DSYRK vs emu55 rel"
                      f" {r['dsyrk_vs_emu55_rel']:.2e}; emumath-DGEMM vs DGEMM bits {r['emumath_dgemm_vs_dgemm_bits']}, vs emu55 bits"
                      f" {r['emumath_dgemm_vs_emu55_bits']}  -> DSYRK {'ROUTES through emulation' if r['emumath_dsyrk_vs_dsyrk_bits'] else 'runs NATIVE'} on the math-mode handle")
            elif "method" in r:
                by[r["case"]][r["method"]].append((rd, r))
    if meta:
        print(f"  {meta.get('device')} cuBLAS {meta.get('cublas')} cuTENSOR {meta.get('cutensor')} emu bits {meta.get('emu_max_bits')} reps {meta.get('reps')}")
    for case, ms in by.items():
        print(f"-- {case}")
        rows = []
        for m, lst in ms.items():
            meds = [r["ms_med"] for _, r in lst]
            r0 = lst[0][1]
            med = sorted(meds)[len(meds) // 2] if len(meds) != 2 else sum(meds) / 2
            fl = 2.0 * r0["n"] ** 2 * r0["k"]
            rows.append((med, m, fl / (med * 1e9), r0["complete"], meds, r0.get("rel_vs_dgemm")))
        for med, m, tf, comp, meds, rel in sorted(rows):
            print(f"  {m:30} {tf:7.1f} TF-eq {med:8.4f} ms  rounds {' / '.join(f'{x:.4f}' for x in meds)}  {'complete' if comp else 'partial'}"
                  + ("" if rel is None else f"  rel vs dgemm {rel:.1e}"))
        routes = any(r.get("emumath_dsyrk_vs_dsyrk_bits") for rd in rounds.values() for r in rd if r.get("case") == case)
        cand = ["cutensor", "emu55", "dsyrk+mirror", "dsyrk+mirror_permute", "emu55+permute"] + (
            ["emumath-dsyrk+mirror", "emumath-dsyrk+mirror_permute"] if routes else [])
        comp = [(med, m) for med, m, tf, c, _, _ in rows if c and m in cand]
        for r in imdb:   # same-job imd_bench cutensor / emu fix55 rows join the minimum
            if r.get("_case") == case and r.get("method") in ("cutensor", "emu") and isinstance(r.get("median_s"), (int, float)):
                comp.append((1e3 * r["median_s"], f"imd_bench:{r['method']}:{r.get('variant', '')}"))
        if comp:
            mn, mm = min(comp)
            bar = 0.95 * mn
            mod = IMD_SYRK_MODEL.get(case)
            print(f"  G2 comparator = {mm} {mn:.4f} ms; kill bar 0.95x = {bar:.4f} ms"
                  + ("" if mod is None else f"; immerdink SYRK v1 model {mod:.3f} ms -> {'SURVIVES' if mod <= bar else 'KILL (deprioritise Hopper SYRK)'}"))
    if imdb:
        print("\n-- imd_bench (same job, dev 91a5cbe build90dev): immerdink N=14/15, cutensor, emu fix55, dgemm")
        for r in imdb:
            if isinstance(r.get("median_s"), (int, float)) and r.get("method"):
                print(f"  {r['_case']:9} {r['method']:9} {str(r.get('variant', '')):24} N={r.get('N', '')!s:3} {r.get('tflops_eq', 0):7.1f} TF-eq"
                      f" {1e3 * r['median_s']:8.4f} ms  spread {r.get('spread_pct', 0):.1f}%  sm {r.get('sm_mhz_median', '')} MHz")
            elif str(r.get("status", "")).startswith("error"):
                print(f"  {r['_case']:9} {r.get('method')} {r.get('variant', '')}: {r['status']}")


def main():
    o = sys.argv[1] if len(sys.argv) > 1 else "out-h200-k45"
    gate_section(o)
    g2b_section(o)


if __name__ == "__main__":
    main()
