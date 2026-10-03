#!/usr/bin/env python3
"""Summarize imd_bench JSON lines as the Markdown tables of docs/BENCHMARKS.md.

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
                print(f"| {name if first else ''} | | | bw (D2D copy) | {ms(r['median_s'])} | "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--N", type=int, default=14)
    ap.add_argument("--sweep-cases", default="sq4096,sq8192")
    ap.add_argument("--phase-cases", default="sq1024,sq2048,sq4096,sq8192,sq16384,r_lowk,r_bigk,"
                    "r_skinny,r_mm,t_nn,t_batch,t_ikl,t_df,t_cc,t_permout")
    a = ap.parse_args()
    recs = load(a.jsonl)
    meta = next((r for r in recs if r.get("status") == "ok"), {})
    print(f"<!-- {a.jsonl}: commit {meta.get('commit')}, env {meta.get('env')}, {meta.get('gpu')}, "
          f"driver {meta.get('driver')}, cuBLAS {meta.get('cublas')}, cuTENSOR {meta.get('cutensor')} -->")
    print(f"\n### All cases (N = {a.N} for the Ozaki-II methods)\n")
    headline(recs, a.N)
    print("\n### N sweep\n")
    nsweep(recs, a.sweep_cases.split(","))
    print(f"\n### immerdink phase lower bounds (MEMORY_MODEL §4, N = {a.N}, ms)\n")
    phase_table(recs, a.phase_cases.split(","), a.N)


if __name__ == "__main__":
    main()
