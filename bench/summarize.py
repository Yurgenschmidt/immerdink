#!/usr/bin/env python3
"""Summarize imd_bench JSON lines as the Markdown tables (see docs/BENCHMARKING.md).

    python3 bench/summarize.py bench/results/<run>.jsonl [--N 14]

Tables: (1) every case at the headline N, (2) the N sweep of the emulated methods,
(3) immerdink phase lower bounds (MEMORY_MODEL §4) against the measured whole call.
"""
import argparse
import json
import math
from collections import OrderedDict

# RTX 4090 datasheet peaks (boost clock): 1008 GB/s GDDR6X, 660.6 dense INT8 TOPS,
# 1.29 FP64 TFLOPS (1/64 rate).
PEAK_BW = 1008e9
PEAK_INT8 = 660.6e12
KCHUNK = 131056


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
    m = r["method"]
    if m in ("gemmul8", "perm+gemmul8", "imd"):
        return f"{m} N={r['N']}"
    if m in ("emu", "perm+emu"):
        return f"{m} {r['variant'].split()[0]}"
    return m


def cases(recs):
    out = OrderedDict()
    for r in recs:
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
            if r["method"] in ("gemmul8", "perm+gemmul8", "imd") and r["N"] != N:
                continue
            if r["method"] in ("emu", "perm+emu") and not r["variant"].startswith(("dyn", "fix55")):
                continue
            if r["method"] == "bw":
                extra = f"{r['gbps']:.0f} GB/s"
                print(f"| {name if first else ''} | {('`' + r['spec'] + '`') if first else ''} | "
                      f"{shape(r) if first else ''} | bw (D2D copy) | {ms(r['median_s'])} | "
                      f"{r['spread_pct']:.1f} | {extra} | | |")
                first = False
                continue
            if r["method"] == "int8":
                t = f"{r['int8_tops']:.0f} TOPS (×{N}: {ms(N * r['median_s'])} ms)"
            else:
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
        rows = [r for r in rs if r.get("status") == "ok" and r["method"] in ("gemmul8", "imd", "emu", "dgemm")]
        rows.sort(key=lambda r: (r["method"], r["N"], r["variant"]))
        for r in rows:
            n = r["N"] if r["method"] in ("gemmul8", "imd") else (r["variant"].split()[0] if r["method"] == "emu" else "—")
            print(f"| {name} | {r['method']} | {n} | {ms(r['median_s'])} | {tf(r['tflops_eq'])} | "
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
          "N × cuBLASLt int8 | measured imd | Σ LB / measured |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        if name not in names:
            continue
        i8 = next((r for r in rs if r["method"] == "int8" and r.get("status") == "ok"), None)
        bw = next((r for r in rs if r["method"] == "bw" and r.get("status") == "ok"), None)
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
                  f"{ms(N * i8['median_s']) if i8 else '—'} | {ms(meas)} | {100 * tot / meas:.0f} % |")
        _ = bw


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


def before_after(before, after, N):
    """immerdink at N, before vs after, with GEMMul8 (after run) for reference."""
    def pick(recs, method):
        out = {}
        for r in recs:
            if r.get("status") == "ok" and r["method"] == method and r["N"] == N:
                out.setdefault(r["case"], r)
        return out
    b, a, g = pick(before, "imd"), pick(after, "imd"), pick(after, "gemmul8")
    gb = pick(before, "gemmul8")
    print(f"| case | m×n×k | imd before ms | imd after ms | speedup | TFLOPS-eq after | "
          f"GEMMul8 ms (before / after run) | imd / GEMMul8 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in a.items():
        rb = b.get(name)
        gg = g.get(name)
        gbb = gb.get(name)
        print(f"| {name} | {shape(r)} | {ms(rb['median_s']) if rb else '—'} | {ms(r['median_s'])} | "
              f"{(rb['median_s'] / r['median_s']) if rb else float('nan'):.2f}× | {tf(r['tflops_eq'])} | "
              f"{ms(gbb['median_s']) if gbb else '—'} / {ms(gg['median_s']) if gg else '—'} | "
              f"{(r['median_s'] / gg['median_s']) if gg else float('nan'):.2f} |")


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
          f"driver {meta.get('driver')}, cuBLAS {meta.get('cublas')}, cuTENSOR {meta.get('cutensor')} -->")
    rss = [(r["host_maxrss_mib"], r["case"]) for r in recs if r.get("host_maxrss_mib") is not None]
    if rss:
        print(f"<!-- peak host RSS of any bench process: {max(rss)[0]:.0f} MiB (case {max(rss)[1]}) -->")
    cg = [(r["cgroup_peak_mib"], r["case"]) for r in recs if r.get("cgroup_peak_mib") is not None]
    if cg:
        print(f"<!-- peak cgroup (memcap scope) memory of any bench process: {max(cg)[0]:.0f} MiB "
              f"(case {max(cg)[1]}) -->")
    print(f"\n### All cases (N = {a.N} for the Ozaki-II methods)\n")
    headline(recs, a.N)
    print("\n### N sweep\n")
    nsweep(recs, a.sweep_cases.split(","))
    print(f"\n### immerdink phase lower bounds (MEMORY_MODEL §4, N = {a.N}, ms)\n")
    phase_table(recs, a.phase_cases.split(","), a.N)
    if any("phases" in r for r in recs):
        print(f"\n### immerdink measured phases (event marks, N = {a.N}): ms (§4 lower bound as % of it)\n")
        measured_phase_table(recs, a.phase_cases.split(","), a.N)
    if a.before:
        print(f"\n### Before / after (N = {a.N}; before: {a.before})\n")
        before_after(load(a.before), recs, a.N)


if __name__ == "__main__":
    main()
