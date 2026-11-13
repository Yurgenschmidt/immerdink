#!/usr/bin/env python3
"""Summarize imd_bench JSON lines as the Markdown tables (see docs/BENCHMARKING.md).

    python3 bench/summarize.py <run>.jsonl [<more>.jsonl ...] [--N 14]   (files are merged)

Tables: (1) every case at the headline N, (2) the N sweep of imd, (3) immerdink phase lower
bounds (MEMORY_MODEL §4) against the measured whole call, (4) measured phases (event marks),
(5) duel (interleaved configurations) and packed (operand reuse) records, if any, and
(6) before / after of imd with --before, (7) the cuTENSOR native / emulated baselines next
to immerdink, if any records of methods cutensor / cutensor-emu are present, and (8) the comparison
of immerdink with the baselines (GEMMul8, cuBLAS FP64 emulation, cuBLAS DGEMM, cuTENSOR native and
emulated) at DGEMM-level accuracy, if any records of methods dgemm / emu / gemmul8 are present.

    python3 bench/summarize.py --only-compare compare.jsonl            # just section (8), also for energy
    python3 bench/summarize.py --only-compare compare_energy.jsonl     # records (GFLOPS-eq per watt)

Section (8) (see "Test 4" in docs/BENCHMARKING.md): per case, the cuBLAS DGEMM record's max normwise error
is the accuracy bar ("DGEMM-level": at or below it, the rule of the evaluation campaigns); for each method
family the fastest (energy: the most efficient) configuration that meets it is shown, with the ratio of
immerdink's best to the best emulator and to DGEMM and cuTENSOR native. The cuTENSOR emulation (8XINT8) is
checked for being engaged: cuTENSOR falls back to native FP64 silently, which shows as an emulated C that is
bit-identical to the native one (the bench's identical-results test: identical_to_native, else the c_hash
of the two records). Every table is robust to missing methods (a baseline that was not built in or failed
gives an error record, listed at the end, and a dash in the table).
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
    base = m.replace("perm+", "")
    pre = "perm+" if m.startswith("perm+") else ""
    if base == "gemmul8":
        return f"{pre}gemmul8-{r.get('g8_mode', 'fast')} N={r.get('N')}"
    if base == "emu":
        return f"{pre}emu {r.get('variant', '')}".strip()
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


def emu_verdict(nat, emu):
    """Was cuTENSOR's emulation (CUTENSOR_COMPUTE_DESC_8XINT8) engaged? -> (code, text), code in
    yes / no / maybe / none. cuTENSOR (>= 2.6) falls back to native FP64 silently where the emulation
    does not apply, so the answer comes from the results, not from the status:
      1. the identical-results test of the bench itself: the emulated record's identical_to_native (its C,
         bit for bit, against a native call on the same inputs, in the same process);
      2. else the c_hash (hash of every bit of C) of the emulated and the native record of the case;
      3. else the older heuristic on time and sampled error (weaker evidence).
    Bit-identical results prove that the emulation did NOT run. Results that differ do not prove that it
    did: asking for 8XINT8 can also select another native algorithm (seen on the RTX 4090: a C different
    from the native plan's but equal to cuBLAS DGEMM's, with the native time and error). So "differs" is
    only "likely engaged" when the time or the error also differ from native; the kernel list of the nsys
    run of the cutensor-emu method (INT8 tensor-core GEMM kernels) is the decisive check. A native result
    that is not run-to-run reproducible (repeat_identical false) cannot show a difference by itself."""
    if emu is None:
        return "none", "—"
    # bench_schema 2: per-record evidence decides (kernel names of one graph-captured call; native twin at the same workspace)
    if isinstance(emu.get("emu_engaged"), bool):
        if emu.get("identical_to_native_same_ws") is True:
            return "no", "NOT engaged (C bit-identical to the native twin at the same workspace)"
        if emu["emu_engaged"]:
            return "yes", "ENGAGED (graph capture: INT8 emulation kernels in the call)"
        return "no", "NOT engaged (graph capture: native FP64 kernels only)"

    def similar():  # time within 5 % and error within a factor 2 of native
        if not nat:
            return False
        te, tn = emu.get("median_s"), nat.get("median_s")
        ee, en = emu.get("max_normerr"), nat.get("max_normerr")
        close_t = te is not None and tn and abs(te / tn - 1) <= 0.05
        close_e = ee is not None and en is not None and en / 2 <= ee <= 2 * max(en, 1e-300)
        return bool(close_t and close_e)

    ident = emu.get("identical_to_native")
    src = "C bit-identical to native"
    if ident is None and nat and nat.get("c_hash") and emu.get("c_hash"):
        ident = nat["c_hash"] == emu["c_hash"]
        src = "C hash equals the native record's"
    if ident is True:
        return "no", f"NOT engaged ({src})"
    if ident is False:
        if nat and nat.get("repeat_identical") is False and similar():
            return "maybe", "inconclusive (native FP64 is not run-to-run reproducible; time and error match it)"
        if similar():
            return "no", ("NOT engaged? (C differs bit for bit from native, but time within 5 % and error comparable: "
                          "another native algorithm; confirm with nsys)")
        return "yes", "likely engaged (C and time or error differ from native; confirm with the nsys kernel list)"
    if not nat:
        return "maybe", "no native record to compare"
    if similar():
        return "no", "not engaged? (time within 5 % of native, same error; no identical-results test in the record)"
    return "yes", "engaged? (time or error differ from native; no identical-results test in the record)"


# Method families of the comparison. dgemm and cutensor are the native references, not candidates.
FAMILIES = [("imd", "immerdink"), ("gemmul8-fast", "GEMMul8 fast"), ("gemmul8-accu", "GEMMul8 accurate"),
            ("emu-fixed", "cuBLAS emu (fixed)"), ("emu-dyn", "cuBLAS emu (dyn)"),
            ("cutensor-emu", "cuTENSOR emu"), ("cutensor", "cuTENSOR FP64"), ("dgemm", "cuBLAS DGEMM")]
EMULATORS = ("gemmul8-fast", "gemmul8-accu", "emu-fixed", "emu-dyn", "cutensor-emu")


def family(r):
    m = r.get("method", "").replace("perm+", "")
    if m == "gemmul8":
        return "gemmul8-" + r.get("g8_mode", "fast")
    if m == "emu":
        return "emu-dyn" if str(r.get("variant", "")).startswith("dyn") else "emu-fixed"
    return m


def cfg_name(r):
    f = family(r)
    if f in ("imd", "gemmul8-fast", "gemmul8-accu"):
        return f"N={r.get('N')}"
    if f in ("emu-fixed",):
        return f"fix{r.get('N')}"
    return ""


def cfg_long(r):
    """Configuration column of the per-configuration table."""
    f = family(r)
    if f.startswith("gemmul8"):
        return f"{f.split('-')[1]} N={r.get('N')}"
    return cfg_name(r) or ("dyn" if f == "emu-dyn" else str(r.get("variant", ""))[:28])


def case_families(rs):
    """Records of one case grouped by family; a bare GEMM record (layout gemm-view...) is dropped where the
    family also has its perm+ record (the permutations are part of the honest comparison)."""
    out = OrderedDict()
    for r in rs:
        if r.get("status") != "ok" or r.get("method") in (None, "idle", "bw"):
            continue
        f = family(r)
        if f in dict(FAMILIES):
            out.setdefault(f, []).append(r)
    for f, lst in out.items():
        if any(str(r["method"]).startswith("perm+") for r in lst):
            out[f] = [r for r in lst if not str(r.get("layout", "")).startswith("gemm-view")]
    mark_native_fallback(out)
    return out


def mark_native_fallback(fams):
    """Flag (r["native_fallback"] = True) the cuBLAS emulation records whose whole-C hash equals one of the
    native records' of the case (cuBLAS DGEMM, cuTENSOR native). The bench sets CUBLAS_EMULATION_STRATEGY_EAGER
    ("emulate whenever possible"); where cuBLAS does not emulate (device, shape, workspace) the call
    presumably runs native FP64 without an error, and a bit-identical C is proof that the emulation did not run
    (a different C does not prove that it did). Such a record is not an emulator result: dgemm_level skips it,
    so it cannot enter "imd / best emulator". Records without a c_hash (older results) stay unflagged."""
    native = {r["c_hash"] for f in ("dgemm", "cutensor") for r in fams.get(f, []) if r.get("c_hash")}
    for f in ("emu-fixed", "emu-dyn"):
        for r in fams.get(f, []):
            if r.get("c_hash") and r["c_hash"] in native:
                r["native_fallback"] = True


def dgemm_level(fams, key):
    """(reference error, source) and, per family, the best record (highest `key`) whose max normwise error is
    at or below the reference and which has no non-finite output. dgemm / cutensor: their own record."""
    ref, src = None, "no DGEMM record"
    for f, name in (("dgemm", "cuBLAS DGEMM"), ("cutensor", "cuTENSOR native")):
        e = [r["max_normerr"] for r in fams.get(f, []) if r.get("max_normerr") is not None]
        if e:
            ref, src = min(e), name
            break
    best = {}
    for f, lst in fams.items():
        if f in ("dgemm", "cutensor"):
            cand = lst
        else:
            cand = [r for r in lst if ref is not None and r.get("max_normerr") is not None
                    and r["max_normerr"] <= ref and not r.get("nonfinite") and not r.get("native_fallback")]
        cand = [r for r in cand if r.get(key) is not None]
        if cand:
            best[f] = max(cand, key=lambda r: r[key])
    return ref, src, best


def _t(r):  # time cell
    return ms(r["median_s"]) + " ms" if r.get("median_s") is not None else ""


def compare_tables(recs, energy=False):
    """The comparison of immerdink with the baselines at DGEMM-level accuracy (section 8 of the docstring)."""
    key = "gflops_per_w" if energy else "tflops_eq"
    unit = "GFLOPS-eq/W" if energy else "TFLOPS-eq"
    rows = [r for r in recs if (r.get("bench_mode") == "energy") == energy]
    cs = cases(rows)
    if not cs:
        print("(no records)")
        return

    def cell(r, f, flag=""):
        if r is None:
            return "—"
        v = r.get(key)
        extra = f"{r['energy_j_per_call']:.3g} J" if energy and r.get("energy_j_per_call") is not None else _t(r)
        c = cfg_name(r)
        sfx = "*" if r.get("contaminated") and not energy else ""  # an energy window has its own criteria
        return f"{v:.3g}{sfx} ({extra}{', ' + c if c else ''}){flag}"

    print(f"Per method, the best configuration at DGEMM-level accuracy (max normwise error at or below the "
          f"cuBLAS DGEMM record's on the same inputs), {unit} with the {'energy per call' if energy else 'time'} "
          "and the configuration." + ("" if energy else " `*`: the record's own spread is above the limit, or the "
          "case's retries did not get its share of noisy records below the limit (record flagged contaminated).") + " \"none\": the method ran but no configuration reached "
          "DGEMM-level accuracy.\n")
    print("| case | m×n×k | DGEMM error | " + " | ".join(n for _, n in FAMILIES) +
          " | imd / best emulator | imd / DGEMM | imd / cuTENSOR |")
    print("|---|---|---|" + "---|" * (len(FAMILIES) + 3))
    errors = OrderedDict()
    verdicts = OrderedDict()
    for name, rs in cs.items():
        fams = case_families(rs)
        for r in rs:
            if r.get("status", "ok") != "ok" and r.get("method") not in (None, "idle"):
                errors.setdefault(name, OrderedDict()).setdefault((r["method"], r["status"][:100]), 0)
                errors[name][(r["method"], r["status"][:100])] += 1
        if not fams:
            continue
        ref, src, best = dgemm_level(fams, key)
        shp = next((shape(r) for lst in fams.values() for r in lst if "m" in r), "")
        verdict = emu_verdict((fams.get("cutensor") or [None])[0], (fams.get("cutensor-emu") or [None])[0])
        verdicts[name] = verdict
        cells = []
        for f, _ in FAMILIES:
            if f not in fams:
                cells.append("—")
            elif f not in best:
                fb = [r for r in fams[f] if r.get("native_fallback")]
                cells.append(f"not engaged ({len(fb)} of {len(fams[f])} configurations = native C)"
                             if fb and f.startswith("emu-") else "none")
            elif f == "cutensor-emu":
                cells.append(cell(best[f], f, {"no": " (not engaged)", "maybe": " (engagement unconfirmed)"}.get(verdict[0], " (likely engaged)")))
            else:
                cells.append(cell(best[f], f))
        emus = [best[f][key] for f in EMULATORS if f in best and not (f == "cutensor-emu" and verdict[0] != "yes")]
        imd = best["imd"][key] if "imd" in best else None

        def ratio(x, y):
            return f"{x / y:.2f}×" if x and y else "—"
        print(f"| {name} | {shp} | {err(ref) if ref is not None else '—'} | " + " | ".join(cells) +
              f" | {ratio(imd, max(emus) if emus else None)} | "
              f"{ratio(imd, best['dgemm'][key]) if 'dgemm' in best else '—'} | "
              f"{ratio(imd, best['cutensor'][key]) if 'cutensor' in best else '—'} |")
    if not energy and any("cutensor-emu" in case_families(rs) for rs in cs.values()):
        print("\n#### Is the cuTENSOR emulation (8XINT8) engaged?\n")
        print("| case | cuTENSOR native error | emulated error | native / emulated time | verdict |")
        print("|---|---|---|---|---|")
        for name, rs in cs.items():
            fams = case_families(rs)
            nat, emu = (fams.get("cutensor") or [None])[0], (fams.get("cutensor-emu") or [None])[0]
            if emu is None:
                continue
            tr = f"{nat['median_s'] / emu['median_s']:.2f}" if nat and emu.get("median_s") else "—"
            print(f"| {name} | {err(nat.get('max_normerr')) if nat else '—'} | {err(emu.get('max_normerr'))} | "
                  f"{tr} | {verdicts[name][1]} |")
    if errors:
        print("\n#### Methods that did not run (error records)\n")
        print("| case | method | error | records |")
        print("|---|---|---|---|")
        for name, d in errors.items():
            for (m, why), n in d.items():
                print(f"| {name} | {m} | {why} | {n} |")


def config_table(recs, energy=False):
    """Every configuration of every case (the long table behind compare_tables), with the DGEMM-level flag."""
    key = "gflops_per_w" if energy else "tflops_eq"
    rows = [r for r in recs if (r.get("bench_mode") == "energy") == energy]
    print(f"| case | method | configuration | {'GFLOPS-eq/W' if energy else 'TFLOPS-eq'} | "
          f"{'J / call' if energy else 'median ms'} | {'avg W' if energy else 'spread %'} | max normwise err | "
          "max rel err | DGEMM-level |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in cases(rows).items():
        fams = case_families(rs)
        ref, _, _ = dgemm_level(fams, key)
        first = True
        for f, _n in FAMILIES:
            for r in sorted(fams.get(f, []), key=lambda r: (r.get("N") or 0, str(r.get("variant")))):
                ok = ref is not None and r.get("max_normerr") is not None and r["max_normerr"] <= ref \
                    and not r.get("nonfinite")
                lvl = "native reference" if f in ("dgemm", "cutensor") else ("yes" if ok else "no")
                if r.get("native_fallback"):
                    lvl = "no (not engaged: C identical to native)"
                v = r.get(key)
                j = r.get("energy_j_per_call")
                tcell = f"{j:.3g}" if energy and j is not None else (ms(r["median_s"]) if r.get("median_s") else "—")
                flag = "*" if r.get("contaminated") and not energy else ""
                last = f"{r.get('avg_power_w', float('nan')):.0f}" if energy else f"{r.get('spread_pct', float('nan')):.1f}"
                print(f"| {name if first else ''} | {r['method']} | {cfg_long(r)} | "
                      f"{'—' if v is None else f'{v:.3g}'}{flag} | {tcell} | "
                      f"{last} | {err(r.get('max_normerr'))} | "
                      f"{err(r.get('max_relerr'))} | {lvl} |")
                first = False


def cutensor_table(recs, N):
    """cuTENSOR native (CUTENSOR_COMPUTE_DESC_64F) and emulated (8XINT8) next to immerdink at N.

    cuTENSOR falls back to native FP64 silently where its emulation does not apply, so "engaged" is
    inferred from the results (emu_verdict): bit-identical to the native C = not engaged; a different C
    with the native time and error = probably another native algorithm; a different C and time = likely
    engaged (confirm with the nsys kernel list).
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
            state = emu_verdict(nat, emu)[1]
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
    ap.add_argument("--only-compare", action="store_true",
                    help="print only the comparison tables (immerdink vs the baselines at DGEMM-level accuracy)")
    ap.add_argument("--no-configs", action="store_true",
                    help="omit the per-configuration table behind the comparison")
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
    has_energy = any(r.get("bench_mode") == "energy" for r in recs)
    has_cmp = any(r.get("method") and family(r) in ("dgemm", "emu-dyn", "emu-fixed", "gemmul8-fast", "gemmul8-accu")
                  for r in recs)

    def compare_section():
        for en in (False, True):
            if not any((r.get("bench_mode") == "energy") == en and r.get("method") not in (None, "idle")
                       for r in recs):
                continue
            print(f"\n### immerdink against the baselines at DGEMM-level accuracy "
                  f"({'board energy' if en else 'throughput'})\n")
            compare_tables(recs, energy=en)
            if not a.no_configs:
                print(f"\n### Every configuration ({'board energy' if en else 'throughput'})\n")
                config_table(recs, energy=en)

    if a.only_compare:
        compare_section()
        return
    if has_energy:
        print("\n### Board energy (NVML energy counter; GFLOPS-eq per watt = 2·m·n·k·b / energy per call)\n")
        energy_table(recs)
        if all(r.get("bench_mode") == "energy" or "method" not in r for r in recs):
            if has_cmp:
                compare_section()
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
    if has_cmp:
        compare_section()


if __name__ == "__main__":
    main()
