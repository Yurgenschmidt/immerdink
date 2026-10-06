#!/usr/bin/env python3
"""Summarize imd_bench JSON lines as the Markdown tables (see docs/BENCHMARKING.md).

    python3 bench/summarize.py bench/results/<run>.jsonl [--N 14]

Tables: (1) every case at the headline N, (2) the N sweep of imd, (3) immerdink phase lower
bounds (MEMORY_MODEL §4) against the measured whole call, (4) measured phases (event marks),
(5) duel (interleaved configurations) and packed (operand reuse) records, if any, and
(6) before / after of imd with --before.
"""
import argparse
import json
import math
from collections import OrderedDict

# RTX 4090 datasheet peaks (boost clock): 1008 GB/s GDDR6X, 660.6 dense INT8 TOPS.
PEAK_BW = 1008e9
PEAK_INT8 = 660.6e12
KCHUNK = 131008  # spec/params.py K_C_PRIME (DESIGN v1.13 §3.5)


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def ms(x):
    return "—" if x is None else (f"{x * 1e3:.3g}" if x * 1e3 < 100 else f"{x * 1e3:.0f}")


def err(x):
    return "—" if x is None else f"{x:.1e}"


def tf(x):
    return "—" if x is None else f"{x:.2f}"


def label(r):
    m = r.get("method", "—")
    if m == "imd":
        v = r["variant"].replace("RN g=N", "").strip()
        return f"imd N={r['N']}" + (f" {v}" if v else "")
    if m in ("packed", "duel-a", "duel-b"):
        return f"{m} {r['variant'].replace(', interleaved', '')}"
    return m


def cases(recs):
    """Records by case. bench/warm_run.sh's clock_state lines (no method, no status) are skipped."""
    out = OrderedDict()
    for r in recs:
        if "method" in r or "status" in r:
            out.setdefault(r["case"], []).append(r)
    return out


def shape(r):
    s = f"{r['m']}×{r['n']}×{r['k']}"
    return s + (f" b={r['b']}" if r.get("b", 1) > 1 else "")


def headline(recs, N):
    print(f"| case | spec | m×n×k | method | median ms | spread % | TFLOPS-eq | max normwise err | max rel err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        ok = [r for r in rs if r.get("status") == "ok"]
        if not ok:
            st = rs[0].get("status", "?")
            print(f"| {name} | | | {st} | | | | | |")
            continue
        first = True
        for r in ok:
            if r["method"] != "bw" and r["N"] != N:
                continue
            if r["method"] == "bw":
                extra = f"{r['gbps']:.0f} GB/s"
                print(f"| {name if first else ''} | {('`' + r['spec'] + '`') if first else ''} | "
                      f"{shape(r) if first else ''} | bw (D2D copy) | {ms(r['median_s'])} | "
                      f"{r['spread_pct']:.1f} | {extra} | | |")
                first = False
                continue
            t = tf(r.get("tflops_eq"))
            print(f"| {name if first else ''} | {('`' + r['spec'] + '`') if first else ''} | "
                  f"{shape(r) if first else ''} | {label(r)} | {ms(r['median_s'])} | "
                  f"{r['spread_pct']:.1f} | {t} | {err(r.get('max_normerr'))} | "
                  f"{err(r.get('max_relerr'))} |")
            first = False
        for r in rs:
            if r.get("status") != "ok":
                print(f"| {name if first else ''} | | | {label(r)} | {r['status'][:90]} | | | | |")
                first = False


def nsweep(recs, names):
    print("| case | method | N | median ms | TFLOPS-eq | max normwise err |")
    print("|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        if name not in names:
            continue
        rows = [r for r in rs if r.get("status") == "ok" and r["method"] == "imd"]
        rows.sort(key=lambda r: (r["N"], r["variant"]))
        for r in rows:
            print(f"| {name} | {label(r)} | {r['N']} | {ms(r['median_s'])} | {tf(r['tflops_eq'])} | "
                  f"{err(r.get('max_normerr'))} |")


def phases(m, n, k, b, N):
    """MEMORY_MODEL §4 lower bounds (g = N, beta = 0, rho = 1), seconds at PEAK_BW/PEAK_INT8,
    plus the byte and op counts."""
    kh, nh = -(-k // 16) * 16, -(-n // 16) * 16
    nch = -(-k // KCHUNK)
    s = 8
    k1 = s * b * (m * k + k * n)
    k23 = s * b * (m * k + k * n) + N * b * (m * kh + nh * kh)
    k4b = N * b * (m * kh + nh * kh) + N * b * m * nh * (2 * nch - 1)
    k4ops = 2 * N * b * m * nh * kh
    k5 = N * b * m * nh + s * b * m * n
    return OrderedDict([("K1 scale", (k1, 0)), ("K2/K3 pack", (k23, 0)), ("K4 GEMM", (k4b, k4ops)),
                        ("K5 CRT", (k5, 0))])


def phase_table(recs, names, N):
    print("| case | N | K1 scale | K2/K3 pack | K4 GEMM | K5 CRT | Σ lower bound | "
          "measured imd | Σ LB / measured |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        if name not in names:
            continue
        for r in rs:
            if r["method"] != "imd" or r.get("status") != "ok" or r["N"] != N:
                continue
            ph = phases(r["m"], r["n"], r["k"], r["b"], N)
            cells, tot = [], 0.0
            for _, (by, ops) in ph.items():
                t = max(by / PEAK_BW, ops / PEAK_INT8)
                tot += t
                cells.append(ms(t))
            meas = r["median_s"]
            print(f"| {name} | {N} | " + " | ".join(cells) + f" | {ms(tot)} | "
                  f"{ms(meas)} | {100 * tot / meas:.0f} % |")


PHASE_GROUPS = OrderedDict([("K0 vec", ["K0 vec"]), ("K1 scale", ["K1 scale A", "K1 scale B"]),
                            ("K2/K3 pack", ["K2 pack A", "K3 pack B"]), ("KL", ["KL A", "KL B"]),
                            ("K4 GEMM", ["K4 gemm"]),
                            ("K5 CRT", ["K5 crt"])])


def measured_phase_table(recs, names, N):
    """Per-phase medians from imd_bench's event marks (field "phases"), each as ms and as the
    MEMORY_MODEL §4 lower bound's share of it (100 % = at the bound)."""
    print("| case | " + " | ".join(PHASE_GROUPS) + " | Σ phases | Σ LB (excl. K0) / Σ |")
    print("|---|" + "---|" * (len(PHASE_GROUPS) + 2))
    for name, rs in cases(recs).items():
        if name not in names:
            continue
        for r in rs:
            if r["method"] != "imd" or r.get("status") != "ok" or r["N"] != N or "phases" not in r:
                continue
            ph, cells = r["phases"], []
            for _, tags in PHASE_GROUPS.items():
                t = sum(ph[x]["ms"] for x in tags if x in ph)
                lb = sum(ph[x]["lb_ms"] for x in tags if x in ph and ph[x].get("lb_ms") is not None)
                cells.append(f"{t:.3g} ({100 * lb / t:.0f} %)" if t > 0 else "—")
            print(f"| {name} | " + " | ".join(cells) + f" | {r['phase_total_ms']:.3g} | "
                  f"{100 * r['phase_lb_ms'] / r['phase_total_ms']:.0f} % |")


def duel_table(recs):
    """method duel: two immerdink configurations interleaved call by call in one process."""
    print("| case | N | A | B | A µs | B µs | t_a / t_b [p10, p90] | bit-identical | "
          "K4 config A / B | plan bytes A / B | launches A / B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    last_a = {}
    for r in recs:
        if r.get("status") != "ok":
            continue
        if r.get("method") == "duel-a":
            last_a[r["case"]] = r
        elif r.get("method") == "duel-b" and r["case"] in last_a:
            ra = last_a.pop(r["case"])
            print(f"| {r['case']} | {r['N']} | {ra['duel_config']} | {r['duel_config']} | "
                  f"{ra['median_s'] * 1e6:.1f} | {r['median_s'] * 1e6:.1f} | "
                  f"{r['ratio_median']:.3f} [{r['ratio_p10']:.3f}, {r['ratio_p90']:.3f}] | "
                  f"{'yes' if r.get('bit_identical') else 'no'} | "
                  f"{ra['k4_config']} / {r['k4_config']} | "
                  f"{ra['plan_device_bytes']} / {r['plan_device_bytes']} | "
                  f"{ra['kernel_launches']} / {r['kernel_launches']} |")


def packed_table(recs):
    """method packed: R x imd_contract vs imd_pack + R x imd_contract_packed (interleaved)."""
    print("| case | N | pack | R | plain R ms | packed R ms | speedup | pack / contract | "
          "bit-identical |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        if r.get("status") != "ok" or r.get("method") != "packed":
            continue
        print(f"| {r['case']} | {r['N']} | {r['pack']} | {r['R']} | {r['plain_R_ms']:.3g} | "
              f"{r['packed_R_ms']:.3g} | {r['speedup_R']:.2f}× | {r['pack_vs_contract']:.2f} | "
              f"{'yes' if r.get('bit_identical') else 'no'} |")


def before_after(before, after, N):
    """imd at N (default configuration), before vs after."""
    def pick(recs):
        out = {}
        for r in recs:
            if (r.get("status") == "ok" and r["method"] == "imd" and r["N"] == N
                    and r["variant"] == "RN g=N"):
                out.setdefault(r["case"], r)
        return out
    b, a = pick(before), pick(after)
    print("| case | m×n×k | imd before ms | imd after ms | speedup | TFLOPS-eq after |")
    print("|---|---|---|---|---|---|")
    for name, r in a.items():
        rb = b.get(name)
        print(f"| {name} | {shape(r)} | {ms(rb['median_s']) if rb else '—'} | {ms(r['median_s'])} | "
              f"{(rb['median_s'] / r['median_s']) if rb else float('nan'):.2f}× | {tf(r['tflops_eq'])} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--N", type=int, default=14)
    ap.add_argument("--before", help="an earlier run's JSONL: adds a before/after table")
    ap.add_argument("--sweep-cases", default="sq4096,sq8192")
    ap.add_argument("--phase-cases", default="sq1024,sq2048,sq4096,sq8192,sq16384,r_lowk,r_bigk,"
                    "r_skinny,r_mm,t_nn,t_batch,t_ikl,t_df,t_cc,t_permout")
    a = ap.parse_args()
    recs = load(a.jsonl)
    meta = next((r for r in recs if r.get("status") == "ok"), {})
    print(f"<!-- {a.jsonl}: commit {meta.get('commit')}, env {meta.get('env')}, {meta.get('gpu')}, "
          f"driver {meta.get('driver')}, CUDA runtime {meta.get('cuda_runtime')} -->")
    rss = [(r["host_maxrss_mib"], r["case"]) for r in recs if r.get("host_maxrss_mib") is not None]
    if rss:
        print(f"<!-- peak host RSS of any bench process: {max(rss)[0]:.0f} MiB (case {max(rss)[1]}) -->")
    cg = [(r["cgroup_peak_mib"], r["case"]) for r in recs if r.get("cgroup_peak_mib") is not None]
    if cg:
        print(f"<!-- peak cgroup (memcap scope) memory of any bench process: {max(cg)[0]:.0f} MiB "
              f"(case {max(cg)[1]}) -->")
    print(f"\n### All cases (N = {a.N})\n")
    headline(recs, a.N)
    print("\n### N sweep\n")
    nsweep(recs, a.sweep_cases.split(","))
    print(f"\n### immerdink phase lower bounds (MEMORY_MODEL §4, N = {a.N}, ms)\n")
    phase_table(recs, a.phase_cases.split(","), a.N)
    if any("phases" in r for r in recs):
        print(f"\n### immerdink measured phases (event marks, N = {a.N}): ms (§4 lower bound as % of it)\n")
        measured_phase_table(recs, a.phase_cases.split(","), a.N)
    if any(r.get("method") == "duel-b" for r in recs):
        print("\n### Duel (interleaved configurations; t_a / t_b > 1: B faster)\n")
        duel_table(recs)
    if any(r.get("method") == "packed" for r in recs):
        print("\n### Packed operand reuse\n")
        packed_table(recs)
    if a.before:
        print(f"\n### Before / after (N = {a.N}; before: {a.before})\n")
        before_after(load(a.before), recs, a.N)


if __name__ == "__main__":
    main()
