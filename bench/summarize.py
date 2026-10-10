#!/usr/bin/env python3
"""Summarize imd_bench JSON lines as the Markdown tables (see docs/BENCHMARKING.md).

    python3 bench/summarize.py <run>.jsonl [<more>.jsonl ...] [--N 14]   (files are merged)

Tables: (1) every case at the headline N, (2) the N sweep of imd, (3) immerdink phase lower
bounds (MEMORY_MODEL §4) against the measured whole call, (4) measured phases (event marks),
(5) duel (interleaved configurations) and packed (operand reuse) records, if any, and
(6) before / after of imd with --before, (7) the cuTENSOR native / emulated baselines next
to immerdink, if any records of methods cutensor / cutensor-emu are present.
"""
import argparse
import json
import math
from collections import OrderedDict

# RTX 4090 datasheet peaks (boost clock): 1008 GB/s GDDR6X, 660.6 dense INT8 TOPS.
PEAK_BW = 1008e9
PEAK_INT8 = 660.6e12
KCHUNK = 131008  # spec/params.py K_C_PRIME (DESIGN v1.13 §3.5)
K_PAD64_MIN = 96  # spec/params.py K_PAD64_MIN (DESIGN v1.13 §4b: A = 64 iff k > 96)


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


def phases(m, n, k, b, N, k_align=None, k_hat=None):
    """MEMORY_MODEL §4 lower bounds (g = N, beta = 0, rho = 1), seconds at PEAK_BW/PEAK_INT8,
    plus the byte and op counts. k_hat / k_align: the record's own (imd_bench records them,
    also under a forced alignment, --k-align); else the AUTO rule of DESIGN v1.13 §4b."""
    A = k_align or (64 if k > K_PAD64_MIN else 16)
    kh, nh = k_hat or -(-k // A) * A, -(-n // 16) * 16
    nch = -(-kh // KCHUNK)
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
            ph = phases(r["m"], r["n"], r["k"], r["b"], N, r.get("k_align"), r.get("k_hat"))
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


def cutensor_table(recs, N):
    """cuTENSOR native (CUTENSOR_COMPUTE_DESC_64F) and emulated (8XINT8) next to immerdink at N.

    cuTENSOR falls back to native FP64 silently where its emulation does not apply, so "engaged" is
    inferred: the emulated record is flagged "not engaged" when its time is within 5 % of the native
    record's and its error is within 2x of it (same case, same inputs).
    """
    print("| case | m×n×k | immerdink N=%d TFLOPS-eq | err | cuTENSOR native TFLOPS-eq | err | "
          "cuTENSOR emulated TFLOPS-eq | err | emulation | imd / native | imd / emulated |" % N)
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        def pick(m):
            return next((r for r in rs if r.get("method") == m and r.get("status") == "ok"), None)
        nat, emu = pick("cutensor"), pick("cutensor-emu")
        emu_err = next((r for r in rs if r.get("method") == "cutensor-emu" and r.get("status") != "ok"), None)
        if not nat and not emu and not emu_err:
            continue
        imd = next((r for r in rs if r.get("method") == "imd" and r.get("N") == N
                    and r.get("status") == "ok"), None)
        ref = next((r for r in (imd, nat, emu) if r), None)
        if ref is None or "m" not in ref:
            continue
        if emu:
            if nat and abs(emu["median_s"] / nat["median_s"] - 1) <= 0.05 \
                    and emu.get("max_normerr", 0) <= 2 * max(nat.get("max_normerr", 0), 1e-300) \
                    and emu.get("max_normerr", 0) * 2 >= nat.get("max_normerr", 0):
                state = "not engaged (time within 5 % of native, same error)"
            elif nat:
                state = "engaged" if emu["median_s"] < nat["median_s"] else "slower than native"
            else:
                state = "no native record to compare"
        else:
            state = emu_err["status"][:60] if emu_err else "—"
        def cell(r):
            return (tf(r.get("tflops_eq")), err(r.get("max_normerr"))) if r else ("—", "—")
        def ratio(a, b):
            return f"{a['tflops_eq'] / b['tflops_eq']:.1f}×" if a and b else "—"
        ci, cn, ce = cell(imd), cell(nat), cell(emu)
        print(f"| {name} | {shape(ref)} | {ci[0]} | {ci[1]} | {cn[0]} | {cn[1]} | {ce[0]} | {ce[1]} | "
              f"{state} | {ratio(imd, nat)} | {ratio(imd, emu)} |")


def energy_table(recs):
    """Records of the energy methods (bench_mode energy): per case, GFLOPS-eq per watt, J per call,
    average board power and sustained TFLOPS-eq of every configuration (+ the idle baseline)."""
    print("| case | m×n×k | configuration | GFLOPS-eq/W | J / call | avg W | TFLOPS-eq (sustained) | "
          "max normwise err | SM MHz (median) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(recs).items():
        rows = [r for r in rs if r.get("bench_mode") == "energy"]
        idle = next((r for r in rows if r.get("method") == "idle" and r.get("status") == "ok"), None)
        first = True
        for r in rows:
            if r.get("method") == "idle":
                continue
            lab = label(r) if r.get("method") in ("imd",) else (r.get("method") + " " + r.get("variant", "")).strip()
            if r.get("status") != "ok":
                print(f"| {name if first else ''} | | {lab} | {r.get('status', '?')[:80]} | | | | | |")
            else:
                j = r.get("energy_j_per_call")
                jt = "—" if j is None else (f"{j * 1e3:.3g} mJ" if j < 1 else f"{j:.3g} J")
                sm = r.get("sm_mhz_median")
                print(f"| {name if first else ''} | {shape(r) if first else ''} | {lab} | "
                      f"{r.get('gflops_per_w', float('nan')):.1f} | {jt} | {r.get('avg_power_w', float('nan')):.0f} | "
                      f"{r.get('tflops_eq_mean', float('nan')):.1f} | {err(r.get('max_normerr'))} | "
                      f"{'—' if sm is None else f'{sm:.0f}'} |")
            first = False
        if idle:
            print(f"| | | idle (awake) | | | {idle['idle_power_w']:.0f} | | | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--N", type=int, default=14)
    ap.add_argument("--before", help="an earlier run's JSONL: adds a before/after table")
    ap.add_argument("--sweep-cases", default="sq4096,sq8192")
    ap.add_argument("--phase-cases", default="sq1024,sq2048,sq4096,sq8192,sq16384,r_lowk,r_bigk,"
                    "r_skinny,r_mm,t_nn,t_batch,t_ikl,t_df,t_cc,t_permout")
    a = ap.parse_args()
    recs = [r for f in a.jsonl for r in load(f)]
    meta = next((r for r in recs if r.get("status") == "ok"), {})
    print(f"<!-- {' '.join(a.jsonl)}: commit {meta.get('commit')}, env {meta.get('env')}, {meta.get('gpu')}, "
          f"driver {meta.get('driver')}, CUDA runtime {meta.get('cuda_runtime')} -->")
    rss = [(r["host_maxrss_mib"], r["case"]) for r in recs if r.get("host_maxrss_mib") is not None]
    if rss:
        print(f"<!-- peak host RSS of any bench process: {max(rss)[0]:.0f} MiB (case {max(rss)[1]}) -->")
    cg = [(r["cgroup_peak_mib"], r["case"]) for r in recs if r.get("cgroup_peak_mib") is not None]
    if cg:
        print(f"<!-- peak cgroup (memcap scope) memory of any bench process: {max(cg)[0]:.0f} MiB "
              f"(case {max(cg)[1]}) -->")
    if any(r.get("bench_mode") == "energy" for r in recs):
        print("\n### Board energy (NVML energy counter; GFLOPS-eq per watt = 2·m·n·k·b / energy per call)\n")
        energy_table(recs)
        if all(r.get("bench_mode") == "energy" or "method" not in r for r in recs):
            return
    print(f"\n### All cases (N = {a.N})\n")
    headline(recs, a.N)
    print("\n### N sweep\n")
    nsweep(recs, a.sweep_cases.split(","))
    if any(r.get("method") in ("cutensor", "cutensor-emu") for r in recs):
        print("\n### cuTENSOR native / emulated FP64 next to immerdink\n")
        cutensor_table(recs, a.N)
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
