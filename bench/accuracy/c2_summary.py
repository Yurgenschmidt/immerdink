#!/usr/bin/env python3
"""Tables of the C2 study (M-a) and the mechanical application of the pre-registered rules
(NOTES-C1 §4.3 d*, §4.5 stage-4 gate, Q12, Q15, Appendix B revival trigger) to the JSONL
records of c2_gate.py. Light: reads records, calls the GPU-free host library for [M] times.

    python3 bench/accuracy/c2_summary.py DIR [--dstar D]   (markdown on stdout)

Conventions: moduli saved = N_min(fast) - N_min(scheme) at equal sampled max error; an
unreachable target (no N <= 20) counts as 21 in medians and is shown as ">20"; times are
[M] (imd_perf_predict of this worktree's library: DEFAULT regime, COLD, GENERATE, g = N,
in-line operands) plus [E] caller-side costs (c2_gate.kw_costs); TFLOPS-eq first. The memory
of scaled copies (gate (b)) is 8 (mk + nk) bytes: both operands pre-scaled, held next to the
plain ones (an upper estimate; a caller that scales in place, or fuses the scaling into its
producer, holds none).

The stage-4 gate verdict is computed as pre-registered (IR27): (i) on the real R1-R6 of the
>= 2048^3 class (2mnk >= 1.7e10), (ii) by the per-output count (1) on every Tier-L / real
fixture's guard-accepted runs (the noise-robust (2) and (3) are printed as diagnostics), and
(iii) from the FP64-clamp round trips.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_gate as cg  # noqa: E402

EW, NW = [str(t) for t in cg.EW_TARGETS], [str(t) for t in cg.NW_TARGETS]
RANDOM_FAMS = ("outliers", "phi2", "phi4", "mixed")
STRUCT_FAMS = ("decay_opp", "opp_nonsep")
# Appendix B's trigger asks for NON-separable magnitudes; these families are separable
# (row factor x K profile x column factor, or constant scale), every other one counts
SEPARABLE_FAMS = ("decay_opp", "decay_same", "uniform", "same_sign")
GATE_CLASS = 1.7e10            # 2mnk of the >= 2048^3 class (§4.3: gate admissibility)
GAIN_BAR = 0.05                # gate (b): net TFLOPS-eq gain >= 5 %
# NOTES-C1 §1.3 / ccsd.py: which operands are rebuilt every iteration (t1-dressed integrals,
# t2 and its intermediates) and which are invariant (bare Bov and what is built from it)
CPU_ALL = {}
REAL_STATUS = {"R1": "both", "R2": "both", "R2b": "both", "R3": "A", "R4": "both",
               "R5": "both"}


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    # a run recorded twice (a timed-out process that kept running, 2026-10-07) is kept once:
    # its first record; every record's CPU time still counts (cpu_total)
    out, seen = [], set()
    for r in recs:
        key = (r.get("tier"), r.get("family"), r.get("molecule"), str(r.get("size")), r.get("seed"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    CPU_ALL[name] = sum(r.get("cpu_s", 0) for r in recs)
    return out


def n_of(v):
    return 21 if v is None else v


def fmt_n(v):
    return ">20" if v is None else str(v)


def saved(rec, scheme, metric="ew", targets=EW):
    """Moduli saved per target (fast - scheme, None-safe)."""
    nm = rec["nmin"]
    out = []
    for t in targets:
        f, s = nm["fast"][metric][t], nm[scheme][metric][t]
        out.append(None if f is None and s is None else n_of(f) - n_of(s))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def choose_dstar(S):
    """§4.3: at Tier S, m = n = 512, 5 seeds: the d with the fewest guard rejections on the
    random-structure fixtures (outliers, phi 2 and 4, mixed), subject to the median gain on
    decay_opp and opp_nonsep (mean moduli saved over the elementwise targets, median over
    runs) staying within 0.25 moduli of the best d. Ties go to 3."""
    rows = [r for r in S if r["size"][0] == 512]
    table = {}
    for d in (0, 2, 3, 5):
        rej = sum(1 for r in rows if r["family"] in RANDOM_FAMS and r["keq"][f"d{d}"]["guard14"] == 0)
        n_rand = sum(1 for r in rows if r["family"] in RANDOM_FAMS)
        gains = [statistics.mean([g for g in saved(r, f"keq_d{d}") if g is not None] or [0])
                 for r in rows if r["family"] in STRUCT_FAMS]
        table[d] = dict(rejections=rej, random_runs=n_rand, median_gain=med(gains))
    best = max(v["median_gain"] for v in table.values() if v["median_gain"] is not None)
    ok = [d for d, v in table.items() if v["median_gain"] is not None
          and v["median_gain"] >= best - 0.25]
    fewest = min(table[d]["rejections"] for d in ok)
    cands = [d for d in ok if table[d]["rejections"] == fewest]
    dstar = 3 if 3 in cands else min(cands)
    return dstar, table


def fixture_table(recs, dstar, schemes):
    """Per (family, size): guard acceptance, G, worst pair, moduli saved (median over seeds)."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r["family"], tuple(r["size"]))].append(r)
    lines = ["| family | m, k | runs | guard d* (acc.) | G d* med / min | worst pair bits d* "
             "med / min | " + " | ".join(f"saved ew {s}" for s in schemes) + " | saved nw "
             f"keq_d{dstar} |", "|---" * (7 + len(schemes)) + "|"]
    for (fam, size), rs in sorted(groups.items()):
        k = rs[0]["keq"][f"d{dstar}"]
        acc = sum(r["keq"][f"d{dstar}"]["guard14"] for r in rs)
        Gs = [r["keq"][f"d{dstar}"]["G14"] for r in rs if r["keq"][f"d{dstar}"]["G14"] is not None]
        W = [r["keq"][f"d{dstar}"]["worst_pair_bits"] for r in rs]
        cells = []
        for s in schemes:
            if s not in rs[0]["nmin"]:
                cells.append("-")
                continue
            per_t = list(zip(*[saved(r, s) for r in rs]))
            cells.append("/".join(fmt_sv(med(list(x))) for x in per_t))
        per_t = list(zip(*[saved(r, f"keq_d{dstar}", "nw", NW) for r in rs]))
        nwc = "/".join(fmt_sv(med(list(x))) for x in per_t)
        lines.append(f"| {fam} | {size[0]}, {size[1]} | {len(rs)} | {acc}/{len(rs)} | "
                     f"{med(Gs) if Gs else '-'} / {min(Gs) if Gs else '-'} | "
                     f"{med(W):+.1f} / {min(W):+.1f} | " + " | ".join(cells) + f" | {nwc} |")
    return "\n".join(lines)


def fmt_sv(v):
    if v is None:
        return "·"
    return f"{v:+g}" if v != 0 else "0"


def nmin_table(recs, schemes, metric="ew", targets=EW):
    lines = ["| case | " + " | ".join(schemes) + " |", "|---" * (len(schemes) + 1) + "|"]
    for r in recs:
        name = r.get("molecule", "") + " " + r["family"] if r.get("molecule") else \
            f"{r['family']} {r['size'][0]}x{r['size'][1]} s{r['seed']}"
        cells = []
        for s in schemes:
            if s not in r["nmin"]:
                cells.append("-")
                continue
            cells.append("/".join(fmt_n(r["nmin"][s][metric][t]) for t in targets))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def label(r):
    if r.get("molecule"):
        return f"{r['molecule']} {r['family']}"
    return f"{r['family']} {r['size'][0]}x{r['size'][1]}"


def admissible(r):
    """The >= 2048^3 class of the stage-4 gate (2mnk >= 1.7e10, §4.3)."""
    return 2.0 * r["m"] * r["n"] * r["k"] >= GATE_CLASS


def scaled_mb(r):
    """Gate (b)'s memory of scaled copies: both operands pre-scaled next to the plain ones."""
    return 8.0 * (r["m"] + r["n"]) * r["k"] / 1e6


def nm_of(r, scheme, metric, t):
    v = r["nmin"].get(scheme, {}).get(metric, {})
    return v.get(str(t)) if v else None


def time_of(r, N, scheme="plain"):
    p = r["perf"][str(N)]
    return p["kseg2_time_s"] if scheme == "kseg2" else p["time_s"], p["flops_eq"]


def gate_table(perf, recs, dstar, statuses, metrics):
    """Per case and gate target: N (measured on the sample) of fast, K-eq d*, K-seg 2, and
    [M] TFLOPS-eq (time) of fast and of K-eq with its [E] caller costs per scenario."""
    kq = f"keq_d{dstar}"
    lines = ["| case | guard (G) | target | N fast / K-eq / K-seg2 | fast TFLOPS-eq (ms) | " +
             " | ".join(f"K-eq {s} TFLOPS-eq (ms), gain" for s in statuses) +
             " | scaled copies (MB) |", "|---" * (6 + len(statuses)) + "|"]
    for r in recs:
        m, n, k = r["m"], r["n"], r["k"]
        g = r["keq"][f"d{dstar}"]
        name = label(r) + ("" if admissible(r) else " (not gate-admissible: 2mnk < 1.7e10)")
        for metric, t in metrics:
            Nf, Nk, Ns = (nm_of(r, s, metric, t) for s in ("fast", kq, "kseg2"))
            cells = [name, f"{g['guard14']} ({g['G14']})", f"{metric} {t:g}",
                     f"{fmt_n(Nf)} / {fmt_n(Nk)} / {fmt_n(Ns)}"]
            if Nf is None:
                cells.append("-")
            else:
                tf, fl = time_of(r, Nf)
                cells.append(f"{fl / tf * 1e-12:.1f} ({tf * 1e3:.3f})")
            for st in statuses:
                if Nf is None or Nk is None:
                    cells.append("(c)" if Nf is None and Nk is not None and metric != "nw" else "-")
                    continue
                stx = st if st != "real" else statuses_real(r)
                tk, fl = time_of(r, Nk)
                c = cg.kw_costs(perf, m * k, n * k, stx)
                cells.append(f"{fl / (tk + c) * 1e-12:.1f} ({(tk + c) * 1e3:.3f}), "
                             f"{(tf / (tk + c) - 1) * 100:+.0f} %")
            cells.append(f"{scaled_mb(r):.0f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gate_table_agg(perf, recs, dstar, statuses, metrics):
    """gate_table aggregated per (family, size, target): median over seeds of the measured N
    and of the [M] TFLOPS-eq / time, and the median net gain per caller-cost scenario."""
    kq = f"keq_d{dstar}"
    groups = defaultdict(list)
    for r in recs:
        groups[(r["family"], tuple(r["size"]))].append(r)
    lines = ["| fixture | runs | target | N fast / K-eq (median) | fast TFLOPS-eq (ms) | " +
             " | ".join(f"K-eq {s}: TFLOPS-eq (ms), gain" for s in statuses) +
             " | scaled copies (MB) |", "|---" * (6 + len(statuses)) + "|"]
    for (fam, size), rs in sorted(groups.items()):
        m, n, k = rs[0]["m"], rs[0]["n"], rs[0]["k"]
        for metric, t in metrics:
            rows = []
            for r in rs:
                Nf, Nk = nm_of(r, "fast", metric, t), nm_of(r, kq, metric, t)
                if Nf is None or Nk is None:
                    continue
                tf, fl = time_of(r, Nf)
                tk, _ = time_of(r, Nk)
                rows.append((Nf, Nk, tf, fl, tk))
            if not rows:
                lines.append(f"| {fam} {size[0]}x{size[1]} | {len(rs)} | {metric} {t:g} | "
                             "unreachable | - |" + " - |" * len(statuses) +
                             f" {scaled_mb(rs[0]):.0f} |")
                continue
            Nf = med([x[0] for x in rows])
            Nk = med([x[1] for x in rows])
            tf = med([x[2] for x in rows])
            fl = rows[0][3]
            cells = [f"{fam} {size[0]}x{size[1]}", str(len(rows)), f"{metric} {t:g}",
                     f"{Nf:g} / {Nk:g}", f"{fl / tf * 1e-12:.1f} ({tf * 1e3:.2f})"]
            for st in statuses:
                c = cg.kw_costs(perf, m * k, n * k, st)
                tks = [x[4] + c for x in rows]
                gains = [x[2] / (x[4] + c) - 1 for x in rows]
                tkm = med(tks)
                cells.append(f"{fl / tkm * 1e-12:.1f} ({tkm * 1e3:.2f}), {med(gains) * 100:+.0f} %")
            cells.append(f"{scaled_mb(rs[0]):.0f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def statuses_real(r):
    return REAL_STATUS.get(r["family"], "both")


def gate_verdict(perf, S, Lr, real, multi, batch, dstar):
    """The stage-4 gate (NOTES-C1 §4.5, IR27) applied mechanically. (i): some real R1-R6
    contraction of the >= 2048^3 class has guard = 1 at d* and either (b) a net [M] + [E]
    TFLOPS-eq gain >= 5 % at a gate target (ew 1e-8 / 1e-10, nw 1e-10 / 1e-12 relative to the
    plain norms; caller costs from the operand status, REAL_STATUS) or (c) an elementwise
    target <= 1e-8 fast mode cannot reach at N <= 20 that K-eq reaches. A guard-accepted
    contraction whose (b) and (c) this script does not compute (R6, several partners: no
    [M] composition here) leaves (i) INCOMPLETE, not failed, unless another contraction meets
    it. (ii): no Tier-L or real fixture has a sampled output of a guard-accepted K-eq d* run
    above 2x fast's error at the same N (the per-output count (1)). (iii): no FP64-clamp
    violation in any run that scales an operand: every K-eq w of Tier S, Tier L and the real
    operands (fresh and stale), and every several-partner and batched w (B6 scales nothing).
    Stage 4 PASSES if all three are met, is INCOMPLETE if (i) is and nothing fails, and FAILS
    otherwise."""
    kq = f"keq_d{dstar}"
    out, ok_i, undet = [], False, []
    for r in real:
        if r["family"] not in ("R1", "R2", "R2b", "R3", "R4", "R5", "R6"):
            continue
        if r["family"] == "R6":                         # several partners: the union w
            g = r["weights"]["shared"]
            if g["guard"] == 1:
                undet.append(label(r))
            out.append(f"  - {label(r)}: guard {g['guard']} (G {g['G']}) -> (i) not met"
                       if g["guard"] != 1 else f"  - {label(r)}: guard 1, (b)/(c) not computed "
                       f"-> (i) undetermined")
            continue
        if not admissible(r):
            out.append(f"  - {label(r)}: not gate-admissible (2mnk < 1.7e10)")
            continue
        g = r["keq"][f"d{dstar}"]
        gains, c_ok = [], False
        for metric, ts in (("ew", cg.GATE_EW), ("nw", cg.GATE_NW)):
            for t in ts:
                Nf, Nk = nm_of(r, "fast", metric, t), nm_of(r, kq, metric, t)
                if Nf is not None and Nk is not None:
                    tf, _ = time_of(r, Nf)
                    tk, _ = time_of(r, Nk)
                    c = cg.kw_costs(perf, r["m"] * r["k"], r["n"] * r["k"], statuses_real(r))
                    gains.append(tf / (tk + c) - 1)
        for t in cg.EW_TARGETS:
            if t <= 1e-8 and nm_of(r, "fast", "ew", t) is None and nm_of(r, kq, "ew", t) is not None:
                c_ok = True
        b_ok = bool(gains) and max(gains) >= GAIN_BAR
        met = g["guard14"] == 1 and (b_ok or c_ok)
        ok_i |= met
        out.append(f"  - {label(r)}: guard {g['guard14']} (G {g['G14']}); (b) best net gain "
                   + (f"{max(gains) * 100:+.0f} %" if gains else "-") +
                   f"; (c) {'yes' if c_ok else 'no'} -> (i) {'met' if met else 'not met'}")
    lines = ["- (i) MET" if ok_i else
             f"- (i) INCOMPLETE: no contraction meets it, and (b)/(c) are not computed for "
             f"{', '.join(undet)} (guard = 1)" if undet else
             "- (i) NOT MET: no real R1-R6 contraction of the >= 2048^3 class has guard = 1 "
             "with (b) or (c)"] + out
    bad, diag = [], []
    for r in Lr + real:
        v = r.get("regress", {}).get(f"d{dstar}")
        if v is None:
            continue
        if v["count"]:
            bad.append((label(r), r.get("seed"), v["count"], v["max_ratio"]))
        if v.get("systematic") or v.get("metric_N"):
            diag.append((label(r), r.get("seed")))
    fx_ = defaultdict(lambda: [0, 0.0])
    for nm, _, c, mx in bad:
        fx_[nm][0] += c
        fx_[nm][1] = max(fx_[nm][1], mx)
    lines.append(f"- (ii) {'NOT MET' if bad else 'MET'} as pre-registered (per-output count (1) "
                 f"of guard-accepted K-eq d* runs): " +
                 ("; ".join(f"{nm} {c} outputs, max ratio {mx:.3g}" for nm, (c, mx) in
                            sorted(fx_.items())) if bad else "no output above 2x"))
    lines.append(f"  - diagnostic, the noise-robust (2) and (3) (the re-registered (ii) of a "
                 f"revival, §4.5): {len(diag)} runs flagged")
    scaled = S + Lr + real
    viol = sum(v.get("fp64_violations", 0) for x in scaled for v in x.get("keq", {}).values())
    viol += sum(v.get("fp64_violations", 0) for x in scaled for v in x.get("stale", {}).values())
    viol += sum(v.get("fp64_violations", 0) for x in multi + real
                for v in x.get("weights", {}).values())
    viol += sum(x.get("fp64_violations", 0) for x in batch + real if "guard" in x)
    nw = sum(len(x.get("keq", {})) + len(x.get("stale", {})) for x in scaled) + \
        sum(len(x.get("weights", {})) for x in multi + real) + \
        sum(1 for x in batch + real if "guard" in x)
    lines.append(f"- (iii) {'MET' if viol == 0 else 'NOT MET'}: {viol} FP64-clamp violations "
                 f"over {nw} weight vectors (Tier S, Tier L and real K-eq, fresh and stale; "
                 f"several-partner and batched)")
    if ok_i and not bad and viol == 0:
        verdict = "PASSES"
    elif not ok_i and undet and not bad and viol == 0:
        verdict = "INCOMPLETE ((i) undetermined)"
    else:
        verdict = "FAILS"
    lines.append(f"- **Stage 4: {verdict}**" +
                 (" -> closed with the revival trigger \"a real-operand family meeting (i)\"; "
                  "stage 6 deferred (its gate needs a shipped stage 4)" if verdict == "FAILS"
                  else ""))
    return "\n".join(lines)


def kseg_table(recs, dstar, metrics):
    """Q15: K-seg 2 (and with K-eq d*) against K-eq d* and fast, N and [M] time (K-seg's own
    composition: two calls of k/2, the second reading C)."""
    lines = ["| case | target | N fast / K-eq / K-seg2 / K-eq+K-seg2 | t fast | t K-eq (no caller "
             "costs) | t K-seg2 | K-seg2 beyond K-eq? |", "|---|---|---|---|---|---|---|"]
    kq = f"keq_d{dstar}"
    for r in recs:
        for metric, t in metrics:
            Ns = [nm_of(r, s, metric, t) for s in ("fast", kq, "kseg2", f"{kq}+kseg2")]
            tt = []
            for N, sch in ((Ns[0], "plain"), (Ns[1], "plain"), (Ns[2], "kseg2")):
                tt.append(time_of(r, N, sch)[0] if N is not None else None)
            best_other = min([x for x in tt[:2] if x is not None], default=None)
            beyond = tt[2] is not None and (best_other is None or tt[2] < best_other)
            lines.append(f"| {label(r)} | {metric} {t:g} | " + " / ".join(fmt_n(x) for x in Ns) +
                         " | " + " | ".join("-" if x is None else f"{x * 1e3:.3f} ms" for x in tt) +
                         f" | {'yes' if beyond else 'no'} |")
    return "\n".join(lines)


def agg_by_fixture(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["family"], tuple(r["size"]))].append(r)
    return groups


def md1_tables(d):
    """M-d1 summary from DIR/../m_d1."""
    P = print
    md = os.path.join(os.path.dirname(os.path.abspath(d.rstrip("/"))), "m_d1")
    rows = []
    for f in ("md1_jacobi.jsonl", "md1_diis.jsonl", "md1_ppl.jsonl"):
        p = os.path.join(md, f)
        if os.path.exists(p):
            rows += [json.loads(x) for x in open(p)]
    if rows:
        P("\n### M-d1 fixed-point systems: iterations to 1e-7 / 1e-10, final residual, drift\n")
        P("| system | schedule | it 1e-7 | it 1e-10 | final residual (80-bit) | drift vs exact-"
          "contraction ref (max / final) | phases (N: first iteration x length) | success |")
        P("|---|---|---|---|---|---|---|---|")
        for r in rows:
            dr = r.get("drift", {}).get("ref")
            ph = "; ".join(f"{'/'.join(map(str, x[0]))}: {x[1]}x{x[2]}" for x in (r.get("phases") or []))
            sc = r.get("success")
            P(f"| {r['system']} | {r['schedule']} | {r['it_1e7']} | {r['it_1e10']} | "
              f"{r['final_residual']:.2e} | " + (f"{dr[0]:.1e} / {dr[1]:.1e}" if dr else "-") +
              f" | {ph if len(ph) < 120 else ph[:117] + '...'} | " +
              ("-" if sc is None else f"{'PASS' if sc['pass_'] else 'FAIL'} "
               f"(ratio {sc['residual_ratio']:.2f}, +{sc['extra_iterations']['it_1e10']} it)") + " |")
    for f in ("md1_rho.jsonl", "md1_rho_local.jsonl"):
        p = os.path.join(md, f)
        if not os.path.exists(p):
            continue
        rs = [json.loads(x) for x in open(p)]
        P(f"\n### M-d1 rho-miss rate (lagged rho_hat, ELEMENTWISE BOUND, levels up to 20; {f})\n")
        P("| system | target | rho_hat | MET calls checked | misses | rate | capped (MISSED) "
          "calls | N used |")
        P("|---|---|---|---|---|---|---|---|")
        for r in rs:
            Ns = defaultdict(int)
            for row in r["per_iteration"]:
                for c in row:
                    Ns[c["N"]] += 1
            impl = r.get("rho_impl", "local")
            P(f"| {r['system']} | {r['eps']:g} | "
              f"{'precision.rho_from_output' if impl == 'shipped' else 'local recipe'} | "
              f"{r['checked']} | {r['misses']} | "
              f"{(r['miss_rate'] or 0) * 100:.1f} % | {r.get('capped_calls')} | " +
              ", ".join(f"{k}: {v}" for k, v in sorted(Ns.items())) + " |")
        P(f"\nTotal: {sum(r['misses'] for r in rs)} misses in {sum(r['checked'] for r in rs)} "
          f"MET calls.")
    for f, title in (("md1_davidson.jsonl", "block Davidson (report-only)"),
                     ("md1_cg.jsonl", "CG, 16 right-hand sides (report-only)")):
        p = os.path.join(md, f)
        if not os.path.exists(p):
            continue
        P(f"\n### M-d1 {title}\n")
        P("| schedule | it 1e-7 | it 1e-10 | attainable / eig error | iterations run | phases |")
        P("|---|---|---|---|---|---|")
        for x in open(p):
            r = json.loads(x)
            ph = "; ".join(f"{x_[0][0]}: {x_[1]}x{x_[2]}" for x_ in r["phases"])
            extra = r.get("eig_error", r.get("attainable"))
            P(f"| {r['schedule']} | {r['it_1e7']} | {r['it_1e10']} | {extra:.1e} | "
              f"{r['iters_run']} | {ph if len(ph) < 140 else ph[:137] + '...'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--dstar", type=int, default=None)
    args = ap.parse_args()
    S, Lr = load(args.dir, "S.jsonl"), load(args.dir, "L.jsonl")
    real, multi, batch = load(args.dir, "real.jsonl"), load(args.dir, "multi.jsonl"), \
        load(args.dir, "batch.jsonl")
    b6r = load(args.dir, "b6.jsonl")
    host, h = cg.load_host()
    perf = cg.Perf(host, h)
    P = print
    allr = S + Lr + real + multi + batch
    if allr:
        r0 = allr[0]
        commits = sorted({(x["commit"], ", ".join(x["dirty"]) if "dirty" in x else "not recorded")
                          for x in allr + b6r})
        P(f"Provenance: {r0['lib']} sha256 {r0['lib_sha256'][:16]}..., host API "
          f"{r0['host_api']}; commits (uncommitted study files): " +
          "; ".join(f"{c} ({d or 'none'})" for c, d in commits) + ". Records written before "
          "provenance() recorded the study files' sha256 carry 'not recorded'.")
        P("CPU-seconds in runs per file (duplicates included): " +
          ", ".join(f"{t} {v:.0f}" for t, v in CPU_ALL.items()) +
          f"; peak RSS max {max(x.get('peak_rss_mb', 0) for x in allr):.0f} MB.")
        P(f"CRT precondition: max sum|a'||b'|/H over all sampled outputs of all schemes "
          f"{max(x.get('crt_max_ratio', 0) for x in allr):.4f}; FP64-clamp violations "
          f"{sum(v.get('fp64_violations', 0) for x in allr for v in x.get('keq', {}).values())}; "
          f"sigma engine exact fallbacks {sum(x.get('fallback_rows', 0) for x in allr)}.")
    dstar = args.dstar
    if S:
        d_auto, tab = choose_dstar(S)
        P("\n### d* (Tier S, m = n = 512)\n")
        P("| d | guard rejections (random-structure runs) | median gain decay_opp/opp_nonsep (moduli) |")
        P("|---|---|---|")
        for d, v in tab.items():
            P(f"| {d} | {v['rejections']} / {v['random_runs']} | {v['median_gain']} |")
        P(f"\nRule result: d* = {d_auto}.")
        dstar = d_auto if dstar is None else dstar
    dstar = 3 if dstar is None else dstar
    if real or Lr:
        P("\n### Stage-4 gate (NOTES-C1 §4.5, IR27), applied mechanically\n")
        P(gate_verdict(perf, S, Lr, [x for x in real], multi, batch, dstar))
    for name, recs, schemes in (("Tier S", S, ["keq_d0", "keq_d2", "keq_d3", "keq_d5", "kseg2",
                                                 "kseg4", f"keq_d{dstar}+kseg2", "env", "sc25"]),
                                ("Tier L", Lr, ["keq_d0", f"keq_d{dstar}", "kseg2",
                                                f"keq_d{dstar}+kseg2", "env"])):
        if not recs:
            continue
        P(f"\n### {name}: guard and moduli saved at equal sampled max error "
          f"(elementwise {'/'.join(EW)}; normwise {'/'.join(NW)}; median over seeds)\n")
        P(fixture_table(recs, dstar, schemes))
    gl = [r for r in Lr + real if "keq" in r]
    if gl:
        P("\n### Realized regressions among guard-accepted K-eq runs (Tier L, real)\n")
        P("(1) the pre-registered measure (IR27, gate (ii)): sampled outputs whose error exceeds "
          "fast's at the same N by > 2x (floor u|x|), summed over the N = 4..20 (RN) and 8, 14 "
          "(TRUNC; 'T'), with the N where they occur; it decides gate (ii). Diagnostics only: "
          "(2) systematic, outputs whose median ratio over the informative N exceeds 2; (3) the N "
          "at which the sampled max error exceeds fast's by > 2x where fast's own error in that "
          "metric is below 1.\n")
        P("| case | runs (guard-accepted) | scheme | (1) outputs > 2x | max ratio | (1) at N "
          "(count) | (2) systematic | (3) N |")
        P("|---|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in gl:
            for sname, v in r.get("regress", {}).items():
                groups[(label(r), sname)].append(v)
        for (nm, sname), vs in sorted(groups.items()):
            byN = defaultdict(int)
            for v in vs:
                for N, c in v.get("by_N", {}).items():
                    byN[N] += c
            P(f"| {nm} | {len(vs)} | {sname} | {sum(v['count'] for v in vs)} / "
              f"{sum(v['outputs'] for v in vs)} | {max(v['max_ratio'] for v in vs):.3g} | " +
              (", ".join(f"{N}: {c}" for N, c in sorted(byN.items(),
                                                       key=lambda x: (int(x[0][:-1]), x[0])))
               or "-") +
              f" | {sum(v.get('systematic') or 0 for v in vs)} | "
              f"{sorted({N for v in vs for N in v.get('metric_N', [])}) or '-'} |")
    if Lr:
        acc = [r for r in Lr if r["keq"][f"d{dstar}"]["guard14"] == 1]
        if acc:
            P("\n### Tier L, guard-accepted runs (not gate-admissible: synthetic): [M] net effect "
              "of K-eq d* at equal target, by operand status (caller-cost scenario)\n")
            P(gate_table_agg(perf, acc, dstar, ["invariant", "A", "both"],
                             [("ew", 1e-8), ("ew", 1e-10), ("nw", 1e-10), ("nw", 1e-12)]))
    if real:
        rr = [r for r in real if "nmin" in r and "perf" in r]
        sch = ["fast", "keq_d0", f"keq_d{dstar}", "kseg2", f"keq_d{dstar}+kseg2", "env"]
        P("\n### Real operands: N_min per scheme\n")
        P("Max elementwise (" + "/".join(EW) + "):\n")
        P(nmin_table(rr, sch))
        P("\nElementwise over the outputs with rho <= 1e8 only (diagnostic; " + "/".join(EW) +
          "):\n")
        P(nmin_table(rr, sch, "ew8", EW))
        P("\nNormwise (" + "/".join(NW) + "):\n")
        P(nmin_table(rr, sch, "nw", NW))
        P("\n| case | m x n x k | 2mnk | gate-admissible | guard d0 (G) | guard d* (G) | "
          "worst pair bits d* | median pair bits d* | weighted indices d* | max abs w |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for r in rr:
            g0, g3 = r["keq"]["d0"], r["keq"][f"d{dstar}"]
            P(f"| {label(r)} | {r['m']} x {r['n']} x {r['k']} | {2 * r['m'] * r['n'] * r['k']:.1e} | "
              f"{'yes' if admissible(r) else 'no'} | "
              f"{g0['guard14']} ({g0['G14']}) | {g3['guard14']} ({g3['G14']}) | "
              f"{g3['worst_pair_bits']:+.2f} | {g3['median_pair_bits']:+.2f} | "
              f"{g3['num_weighted']} / {r['k']} | {g3['max_abs_w']} |")
        P("\n### Stage-4 gate (i) on R1-R6: [M] at equal target, K-eq d* with [E] caller costs "
          "(operand status from ccsd.py)\n")
        P(gate_table(perf, rr, dstar, ["real"],
                     [("ew", 1e-8), ("ew", 1e-10), ("nw", 1e-10), ("nw", 1e-12),
                      ("ew8", 1e-8), ("ew8", 1e-10)]))
        P("\n### Q15: K-segmentation against K-equilibration (Tier L guard-accepted and real)\n")
        P(kseg_table([r for r in Lr if r["keq"][f"d{dstar}"]["guard14"] == 1 and r["seed"] == 0]
                     + rr, dstar, [("ew", 1e-10), ("nw", 1e-12), ("nw", 1e-15)]))
    envc, best = [], None
    for r in Lr + [x for x in real if "nmin" in x and "keq" in x]:
        if r["family"] in SEPARABLE_FAMS or not admissible(r):
            continue
        acc = r["keq"][f"d{dstar}"]["guard14"] == 1
        for metric, ts in (("ew", EW), ("nw", NW)):
            for t in ts:
                e = nm_of(r, "env", metric, t)
                # what a caller gets: K-eq d* where its guard holds, otherwise fast mode
                keq = nm_of(r, f"keq_d{dstar}" if acc else "fast", metric, t)
                sv = None if e is None else min(n_of(keq) - e, n_of(nm_of(r, "kseg2", metric, t)) - e)
                if sv is not None:
                    best = sv if best is None else max(best, sv)
                if sv is not None and sv >= 2:
                    envc.append((label(r), r.get("seed"), metric, t))
    P(f"\n### Appendix B revival trigger: non-separable Tier L / real cases of the >= 2048^3 "
      f"class where the envelope rule saves >= 2 moduli beyond both K-seg 2 and what a caller "
      f"gets from K-eq (K-eq d* where its guard holds, otherwise fast mode): {len(envc)}"
      + (f" {envc[:10]}" if envc else "") + f" (largest saving beyond both: {best} moduli)")
    for r in multi + [x for x in real if x["family"] == "R6"]:
        P(f"\n### Several partners: {label(r) if r.get('molecule') else r['family'] + ' ' + str(r['size']) + ' s' + str(r['seed'])}\n")
        P("weights: " + json.dumps(r["weights"]))
        for p, pr in enumerate(r["pairs"]):
            P(f"- pair {p}: ew " + "; ".join(f"{nm} " + "/".join(fmt_n(v["ew"][t]) for t in EW)
                                             for nm, v in pr["nmin"].items()) +
              " | nw " + "; ".join(f"{nm} " + "/".join(fmt_n(v["nw"][t]) for t in NW)
                                   for nm, v in pr["nmin"].items()))
    for r in batch + [x for x in real if x["family"] == "R7"]:
        P(f"\n### Batched: {r.get('molecule') or ''} {r['family']} (b = {r['b']}) s{r.get('seed')}\n")
        P("guard: " + json.dumps(r["guard"]))
        for metric, ts in (("ew", EW), ("nw", NW), ("ew8", EW)):
            if all(metric in v for v in r["nmin"].values()):
                P(f"- {metric} " + "; ".join(f"{nm} " + "/".join(fmt_n(v[metric][t]) for t in ts)
                                            for nm, v in r["nmin"].items()))
    l5 = [r for r in Lr if r.get("l5d") and r["keq"][f"d{dstar}"]["guard14"] == 1]
    if l5:
        P("\n### L5d on guard-accepted Tier L runs: min_moduli (BOUND, nu = sqrt k) for the "
          "weighted call from the plain statistics (cond = the sample's max rho) vs the measured "
          "N_min (elementwise " + "/".join(EW) + "; normwise " + "/".join(NW) + ")\n")
        P("| case | fast bound ew | fast measured ew | K-eq L5d bound ew | K-eq measured ew | "
          "fast bound nw | K-eq L5d bound nw | K-eq measured nw |")
        P("|---|---|---|---|---|---|---|---|")
        for (fam, size), rs in sorted(agg_by_fixture(l5).items()):
            r = rs[0]
            f = r["l5d"]["fast"]
            kq = r["l5d"]["keq"].get(f"d{dstar}", {})
            P(f"| {fam} {size[0]}x{size[1]} s{r['seed']} | " +
              "/".join(fmt_n(f.get(f'ew{t}')) for t in EW) + " | " +
              "/".join(fmt_n(r['nmin']['fast']['ew'][t]) for t in EW) + " | " +
              "/".join(fmt_n(kq.get(f'ew{t}')) for t in EW) + " | " +
              "/".join(fmt_n(r['nmin'][f'keq_d{dstar}']['ew'][t]) for t in EW) + " | " +
              "/".join(fmt_n(f.get(f'nw{t}')) for t in NW) + " | " +
              "/".join(fmt_n(kq.get(f'nw{t}')) for t in NW) + " | " +
              "/".join(fmt_n(r['nmin'][f'keq_d{dstar}']['nw'][t]) for t in NW) + " |")
    b6s = [r for r in S if r.get("b6")] + b6r
    if b6s:
        P("\n### B6 (output-selective refinement) estimate [M]\n")
        P("| case | outputs | target | N_hi | best N_lo | rows refined | ratio to t(N_hi, full) | "
          "rows needing each N |")
        P("|---|---|---|---|---|---|---|---|")
        for r in b6s:
            for key in ("b6", "b6_must"):
                if key not in r:
                    continue
                b = r[key]
                what = ("all" if key == "b6" else f"rho <= 1e8 ({b['excluded_outputs'] * 100:.0f} % "
                        "excluded)") + f", {b['label']}"
                nm = label(r) + ("" if r.get("molecule") else f" s{r['seed']}")
                for t, v in b["targets"].items():
                    if v.get("N_hi") is None:
                        P(f"| {nm} | {what} | {t} | >20 | - | - | - | - |")
                    else:
                        P(f"| {nm} | {what} | {t} | {v['N_hi']} | {v['N_lo']} | "
                          f"{v['rows_refined'] * 100:.1f} % | {v['best_ratio']:.3f} | "
                          f"{v.get('rows_per_N', '-')} |")
    stale = [r for r in S + Lr if r.get("stale")]
    if stale:
        P("\n### Stale weights (seed s's w applied to seed s + 1)\n")
        P("| fixture | runs | guard fresh (acc.) | guard stale (acc.) | median N_min ew fresh | "
          "median N_min ew stale |")
        P("|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in stale:
            for d, v in r["stale"].items():
                if f"keq_dstale{d[1:]}" in r["nmin"]:
                    groups[(r["family"], tuple(r["size"]), d)].append((r, v))
        for (fam, size, d), lst in sorted(groups.items()):
            fr = [x[0]["keq"][d]["guard14"] for x in lst]
            st = [x[1]["guard14"] for x in lst]
            mf = ["/".join(fmt_sv(med([n_of(x[0]['nmin'][f'keq_{d}']['ew'][t]) for x in lst]))
                           .lstrip("+") for t in EW)]
            ms = ["/".join(fmt_sv(med([n_of(x[0]['nmin'][f'keq_dstale{d[1:]}']['ew'][t])
                                       for x in lst])).lstrip("+") for t in EW)]
            P(f"| {fam} {size[0]}x{size[1]} {d} | {len(lst)} | {sum(fr)}/{len(lst)} | "
              f"{sum(st)}/{len(lst)} | {mf[0]} | {ms[0]} |")
    md1_tables(args.dir)


if __name__ == "__main__":
    main()
