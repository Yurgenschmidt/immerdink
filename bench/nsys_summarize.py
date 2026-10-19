#!/usr/bin/env python3
"""Summarize an nsys profile of one imd_bench run: kernels by category, INT8 engagement (issue 11).

Python 3.6+, standard library only. Input: the sqlite export of `nsys profile -t cuda --export sqlite`
(bench/profile_nsys.sh does that). Subcommands:

  summarize --db run.sqlite --tag T --case C --method M [--N n] [--variant v] [--gpu NAME]
            --outdir DIR [--nsys-rep path] [--nsys-version V] [--notes TEXT]
      writes DIR/<tag>/summary.json and adds the tag to DIR/index.json
  missing --outdir DIR --nsys-note TEXT --entries FILE
      nsys (or its export) is not available: writes DIR/index.json and a summary.json per planned tag
      with engaged_int8 null and the note; FILE has one "tag<TAB>case<TAB>method<TAB>N<TAB>variant" per line
  table DIR             markdown table of DIR/index.json and the summaries (profile.md of run_metrics.sh)
  classify NAME...      print the category of kernel names (documentation / debugging)

What is counted. imd_bench times each call as  cudaEventRecord(e0); run(); cudaEventRecord(e1);
cudaEventSynchronize(e1)  (bench/imd_bench.cu time_it), so a "timed window" is e0 .. the synchronize
that returns after e1. The summary counts only GPU activity (kernels, memcpy, memset) that starts inside
the timed windows, never the setup, the accuracy reference, the warm-up or the untimed identity checks
(for cutensor-emu that is a native cuTENSOR call outside the windows). The first window is dropped
(first-call effects; its start is also the least certain) when there are at least 2. Per kernel name: time_ms is the mean total time per timed
call (windows averaged), count the mean launches per call, share_pct the share of the GPU activity time of
a call (sum of kernel + memcpy + memset durations, not the span). wall_ms is the median over the windows
of (last activity end - e0 record start), inflated slightly by the profiler. If no window is found (a
method that does not use time_it) the whole process is summarized and the notes say so.

Categories (CATEGORY_RULES: first match wins; matched on the kernel key, see kernel_key: the function name
without template arguments):
  memcpy_memset         memcpy / memset activities ("[CUDA memcpy ...]", "[CUDA memset]")
  int8_gemm             INT8 tensor-core GEMM: cuBLAS/cuBLASLt/CUTLASS names with imma, i8816, i16832, s8s8 /
                        s8_s32, int8, igemm; immerdink's k4_kernel (the INT8 GEMM mainloop)
  fp64_gemm             FP64 GEMM / contraction: dgemm, d884 / d8816 / d16816 (FP64 MMA), f64 / fp64;
                        immerdink's k_direct (direct FP64 mode)
  permute               tensor permutation / transposition / elementwise (cuTENSOR permute, transpose)
  conversion_pack_crt   the Ozaki-II conversions around the GEMMs: scaling, residue / slice / pack, CRT
                        reconstruction (immerdink kf_scale_pack, k0_vec, k1_*, k23_*, k5_*, k45_*, kl_*;
                        GEMMul8 and cuBLAS-emulation helper kernels)
  other                 everything else
engaged_int8: true when int8_gemm is at least ENGAGED_MIN_SHARE (10 %) of the GPU activity time of a timed
call, false when it is below that (with the exact share in the notes), null when there is no GPU activity
(or no profile). It is decisive for cutensor-emu: native FP64 fall-back shows only fp64_gemm kernels.
"""
import argparse
import json
import os
import re
import sqlite3
import statistics
import sys

ENGAGED_MIN_SHARE = 0.10
NAME_MAX = 400  # longer demangled names (CUTLASS template types) are cut in the summaries

CATEGORIES = ("int8_gemm", "fp64_gemm", "conversion_pack_crt", "permute", "memcpy_memset", "other")

# (category, regex on the kernel KEY); first match wins. The key (kernel_key) is the qualified function name
# without its template arguments, plus the first template argument of cutlass::Kernel / Kernel2 wrappers (which
# carries the real kernel name): matching the whole demangled name would hit type names in the template
# arguments (oz2_int8_dgemm, TransposePitchLinearThreadMap, ...). Order matters: the library's own helper kernels
# first, then FP64 GEMM, INT8 GEMM, permutations, the generic conversion words last.
CATEGORY_RULES = [
    ("memcpy_memset", r"^\[cuda mem(cpy|set)"),
    # immerdink's own kernels (src/): k4_kernel is the INT8 GEMM mainloop, k_direct the FP64 direct mode
    ("int8_gemm", r"\bk4_kernel|\bk4_[a-z0-9_]*kernel"),
    ("fp64_gemm", r"\bk_direct"),
    ("conversion_pack_crt", r"\b(kf_scale_pack|k0_vec|k1_scale|k1_tiled|k23_pack|k23_zero_rows|k5_crt|k5_beta_only|k45_|kl_)"),
    # helper kernels of GEMMul8 and of the cuBLAS emulation (their names carry gemm words, e.g. oz2_int8_dgemm)
    ("conversion_pack_crt", r"^gemmul8::|^prolog_|inverse_scaling|undo_scaling|reduce_nmoduli|^sft_|minmax|"
                            r"unfused_epilogue|quantize"),
    # FP64 GEMM / contraction (cuBLAS, CUTLASS, cuTENSOR native)
    ("fp64_gemm", r"dgemm|dgett|d884|d8816|d16816|f64f64|f64_|_f64|fp64|gemm_d_|dtrsm|dsyrk"),
    # INT8 tensor-core GEMM of the libraries; nvjet kernels carry their type code in the 3rd token
    # (nvjet_sm89_biu_mma_...: an i in the code = int8; seen as the INT8 GEMM of the cuBLAS emulation)
    ("int8_gemm", r"imma|i8816|i8832|i16832|i16864|igemm|s8s8|s8_s8|i8i8|i8i32|s8_s32|_s8_|gemm_s8|s32_s8|int8|8xint8|"
                  r"^nvjet_sm\d+[a-z]?_[a-z]*i[a-z]*_"),
    ("permute", r"permut|transpos|elementwise|tensor_copy|\bcopy_kernel|strided_copy"),
    ("conversion_pack_crt", r"gemmul8|ozaki|oz2|crt|residu|modul|scal(e|ing)|slice|extract|split|convert|pack|reduc|"
                            r"inverse|accumul|emulat|round|trunc|cast"),
]
_COMPILED = [(c, re.compile(p)) for c, p in CATEGORY_RULES]
_WRAPPER = re.compile(r"cutlass::Kernel\d*$")


def kernel_key(name):
    """Lower-cased matching key of a kernel name (see CATEGORY_RULES)."""
    n = name.strip()
    for junk in ("<unnamed>::", "(anonymous namespace)::"):
        n = n.replace(junk, "")
    if n.startswith("void "):
        n = n[5:]
    m = re.match(r"[\w:~]+", n)
    base = m.group(0) if m else n
    key = base
    if _WRAPPER.search(base) and n[len(base):len(base) + 1] == "<":
        m2 = re.match(r"[\w:]+", n[len(base) + 1:])
        if m2:
            key = base + " " + m2.group(0)
    return key.lower()


def classify(name):
    """Category of one kernel (or memcpy / memset activity) name."""
    if name.startswith("[CUDA mem"):
        return "memcpy_memset"
    key = kernel_key(name)
    for cat, rx in _COMPILED:
        if rx.search(key):
            return cat
    return "other"


def _tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def _strings(c):
    return dict(c.execute("select id, value from StringIds"))


def _rows(c, tables, table, cols):
    """Select cols from an optional table; a missing table or column gives no rows."""
    if table not in tables:
        return []
    have = {r[1] for r in c.execute("pragma table_info(%s)" % table)}
    sel = [x if x in have else "NULL" for x in cols]
    return list(c.execute("select %s from %s" % (", ".join(sel), table)))


COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 10: "PtoP"}


def load_activities(db):
    """-> (activities, runtime calls, gpu name). activity = (start, end, name, kind)."""
    c = sqlite3.connect(db)
    tables = _tables(c)
    names = _strings(c)
    acts = []
    for s, e, dn, sn in _rows(c, tables, "CUPTI_ACTIVITY_KIND_KERNEL", ["start", "end", "demangledName", "shortName"]):
        acts.append((s, e, names.get(dn) or names.get(sn) or "?", "kernel"))
    for s, e, kind in _rows(c, tables, "CUPTI_ACTIVITY_KIND_MEMCPY", ["start", "end", "copyKind"]):
        acts.append((s, e, "[CUDA memcpy %s]" % COPY_KINDS.get(kind, "?"), "memcpy"))
    for s, e in _rows(c, tables, "CUPTI_ACTIVITY_KIND_MEMSET", ["start", "end"]):
        acts.append((s, e, "[CUDA memset]", "memset"))
    acts.sort()
    rt = []
    for s, e, n, tid in _rows(c, tables, "CUPTI_ACTIVITY_KIND_RUNTIME", ["start", "end", "nameId", "globalTid"]):
        rt.append((tid or 0, s, e, names.get(n, "")))
    rt.sort()
    gpu = None
    for (n,) in _rows(c, tables, "TARGET_INFO_GPU", ["name"]):
        gpu = n
        break
    c.close()
    return acts, rt, gpu


def find_windows(rt):
    """Timed windows [(e0 start, sync end)] from the runtime-API calls (see the module docstring).
    A window ends at a cudaEventSynchronize whose previous call (same thread) is a cudaEventRecord (e1) and
    starts at the first cudaEventRecord after the previous such synchronize. The first window may start at an
    unrelated earlier event record (setup), so summarize_db drops it."""
    out = []
    by_tid = {}
    for r in rt:
        by_tid.setdefault(r[0], []).append(r)
    for calls in by_tid.values():
        calls.sort(key=lambda r: r[1])
        prev_end_idx = -1
        for i, (_, s, e, n) in enumerate(calls):
            if not n.startswith("cudaEventSynchronize") or i == 0 or not calls[i - 1][3].startswith("cudaEventRecord"):
                continue
            # e0: first cudaEventRecord after prev_end_idx, other than e1 itself
            j0 = None
            for j in range(prev_end_idx + 1, i - 1):
                if calls[j][3].startswith("cudaEventRecord"):
                    j0 = j
                    break
            if j0 is None:  # no e0 between the previous window and this e1: not a bench window
                prev_end_idx = i
                continue
            out.append((calls[j0][1], e))
            prev_end_idx = i
    out.sort()
    return out


def summarize_db(db, drop_first=True):
    """-> dict(kernels, wall_ms, windows, mode, activity_ms, gpu)"""
    acts, rt, gpu = load_activities(db)
    wins = find_windows(rt)
    mode = "timed windows"
    if not acts:
        return dict(kernels=[], wall_ms=None, windows=0, mode="no GPU activity", activity_ms=0.0, gpu=gpu)
    if wins:
        use = wins[1:] if (drop_first and len(wins) >= 2) else wins
        per = []
        walls = []
        for lo, hi in use:
            inside = [a for a in acts if lo <= a[0] <= hi]
            if not inside:
                continue
            per.append(inside)
            walls.append((max(a[1] for a in inside) - lo) / 1e6)
    else:
        mode = "whole process (no timed window found)"
        per = [acts]
        walls = [(acts[-1][1] - acts[0][0]) / 1e6]
    n = max(len(per), 1)
    agg = {}
    for inside in per:
        for s, e, name, kind in inside:
            a = agg.setdefault(name, [0, 0])
            a[0] += e - s
            a[1] += 1
    total = sum(v[0] for v in agg.values())
    kernels = []
    for name, (t, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        kernels.append(dict(name=name if len(name) <= NAME_MAX else name[:NAME_MAX - 3] + "...", category=classify(name), time_ms=round(t / n / 1e6, 4),
                            share_pct=round(100.0 * t / total, 2) if total else 0.0,
                            count=round(cnt / n, 2) if cnt % n else cnt // n))
    return dict(kernels=kernels, wall_ms=round(statistics.median(walls), 4) if walls else None,
                windows=len(per), mode=mode, activity_ms=round(total / n / 1e6, 4), gpu=gpu)


def category_shares(kernels):
    tot = sum(k["time_ms"] for k in kernels)
    sh = {c: 0.0 for c in CATEGORIES}
    for k in kernels:
        sh[k["category"]] += k["time_ms"]
    return {c: (v / tot if tot else 0.0) for c, v in sh.items()}


def bench_status(runlog):
    """-> (list of status strings of the bench records that are not ok, number of ok records)."""
    bad, ok = [], 0
    try:
        lines = open(runlog).read().splitlines()
    except (OSError, TypeError):
        return bad, ok
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        st = d.get("status")
        if st is None:
            continue
        if st == "ok":
            ok += 1
        else:
            bad.append(str(st)[:200])
    return bad, ok


def make_summary(tag, case, method, N, variant, db, gpu_cli=None, nsys_version=None, extra_notes="", runlog=None):
    r = summarize_db(db)
    sh = category_shares(r["kernels"])
    int8 = sh["int8_gemm"]
    if not r["kernels"]:
        engaged = None
    else:
        engaged = int8 >= ENGAGED_MIN_SHARE
    notes = ["%s, %d window(s); GPU activity %.3f ms per call" % (r["mode"], r["windows"], r["activity_ms"]),
             "shares of GPU activity time: " + ", ".join("%s %.1f%%" % (c, 100 * sh[c]) for c in CATEGORIES if sh[c] > 0)]
    if method == "cutensor-emu":
        if engaged:
            notes.append("cuTENSOR emulation ENGAGED: INT8 tensor-core GEMM kernels ran (%.1f%% of the call)" % (100 * int8))
        elif engaged is False:
            notes.append("cuTENSOR emulation NOT engaged: no significant INT8 GEMM kernel time (%.1f%%); "
                         "it ran native FP64 (%.1f%% fp64_gemm)" % (100 * int8, 100 * sh["fp64_gemm"]))
    if sh["other"] > 0.2:
        notes.append("%.0f%% of the time is in kernels classified 'other': check the name patterns (nsys_summarize.py)"
                     % (100 * sh["other"]))
    bad, nok = bench_status(runlog)
    if bad:
        notes.append("imd_bench reported: " + " | ".join(bad))
    if extra_notes:
        notes.append(extra_notes)
    return dict(tag=tag, case=case, method=method, N=N, variant=variant, gpu=gpu_cli or r["gpu"],
                wall_ms=r["wall_ms"], kernels=r["kernels"], engaged_int8=engaged,
                int8_share_pct=round(100 * int8, 2) if r["kernels"] else None, windows=r["windows"],
                bench_status=("ok" if nok and not bad else (bad[0] if bad else "no record")),
                nsys_version=nsys_version, notes="; ".join(notes))


def _update_index(outdir, entry):
    p = os.path.join(outdir, "index.json")
    idx = []
    if os.path.exists(p):
        try:
            idx = json.load(open(p))
        except ValueError:
            idx = []
    idx = [e for e in idx if e.get("tag") != entry["tag"]]
    idx.append(entry)
    with open(p, "w") as f:
        json.dump(idx, f, indent=1)
        f.write("\n")


def _write_summary(outdir, summ, nsys_rep):
    d = os.path.join(outdir, summ["tag"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
        f.write("\n")
    _update_index(outdir, dict(tag=summ["tag"], case=summ["case"], method=summ["method"], N=summ["N"],
                               variant=summ.get("variant"), nsys_rep=nsys_rep or None,
                               summary="%s/summary.json" % summ["tag"], engaged_int8=summ["engaged_int8"]))


def _N(v):
    return None if v in (None, "", "-", "null") else int(v)


def profile_table(d):
    """Markdown table of a profile directory (index.json + <tag>/summary.json)."""
    try:
        idx = json.load(open(os.path.join(d, "index.json")))
    except (OSError, ValueError):
        return "(no profile/index.json)"
    cols = ["int8_gemm", "fp64_gemm", "conversion_pack_crt", "permute", "memcpy_memset", "other"]
    out = ["| tag | wall (ms) | INT8 engaged | " + " | ".join(c + " %" for c in cols) + " | top kernel |",
           "|---|---:|:---:|" + "---:|" * len(cols) + "---|"]
    for e in idx:
        try:
            sm = json.load(open(os.path.join(d, e["summary"])))
        except (OSError, ValueError, KeyError):
            out.append("| %s | - | - |" % e.get("tag") + " - |" * (len(cols) + 1))
            continue
        ks = sm.get("kernels") or []
        sh = category_shares(ks) if ks else {c: 0.0 for c in CATEGORIES}
        eng = {True: "yes", False: "NO", None: "n/a"}[sm.get("engaged_int8")]
        top = re.sub(r"\s+", " ", ks[0]["name"])[:70] if ks else (sm.get("notes") or "")[:70]
        wall = "%.3f" % sm["wall_ms"] if sm.get("wall_ms") is not None else "-"
        out.append("| %s | %s | %s | %s | %s |" % (sm["tag"], wall, eng, " | ".join("%.1f" % (100 * sh[c]) for c in cols),
                                                  top.replace("|", "/")))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True  # (the required= keyword needs Python 3.7)
    s = sub.add_parser("summarize")
    s.add_argument("--db", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--case", required=True)
    s.add_argument("--method", required=True)
    s.add_argument("--N", default=None)
    s.add_argument("--variant", default=None)
    s.add_argument("--gpu", default=None)
    s.add_argument("--outdir", required=True)
    s.add_argument("--nsys-rep", default=None)
    s.add_argument("--nsys-version", default=None)
    s.add_argument("--notes", default="")
    s.add_argument("--runlog", default=None, help="the imd_bench --out file of the profiled run (JSON records)")
    m = sub.add_parser("missing")
    m.add_argument("--outdir", required=True)
    m.add_argument("--nsys-note", required=True)
    m.add_argument("--entries", required=True)
    m.add_argument("--gpu", default=None)
    t = sub.add_parser("table")
    t.add_argument("dir")
    c = sub.add_parser("classify")
    c.add_argument("names", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "classify":
        for n in a.names:
            print("%-20s %s" % (classify(n), n))
        return 0
    if a.cmd == "table":
        print(profile_table(a.dir))
        return 0
    os.makedirs(a.outdir, exist_ok=True)
    if getattr(a, "variant", None) in ("", "-"):
        a.variant = None
    if a.cmd == "summarize":
        try:
            summ = make_summary(a.tag, a.case, a.method, _N(a.N), a.variant or None, a.db, a.gpu, a.nsys_version,
                                a.notes, a.runlog)
        except (sqlite3.Error, OSError) as e:
            summ = dict(tag=a.tag, case=a.case, method=a.method, N=_N(a.N), variant=a.variant or None, gpu=a.gpu,
                        wall_ms=None, kernels=[], engaged_int8=None, int8_share_pct=None, windows=0,
                        nsys_version=a.nsys_version, notes="could not read the nsys export: %s" % e)
        _write_summary(a.outdir, summ, a.nsys_rep)
        print("%s: %s, int8 %s%%, wall %s ms" % (a.tag, "engaged_int8=%s" % summ["engaged_int8"],
                                                 summ["int8_share_pct"], summ["wall_ms"]))
        return 0
    # missing
    idx = []
    for line in open(a.entries):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        f += [""] * (5 - len(f))
        summ = dict(tag=f[0], case=f[1], method=f[2], N=_N(f[3]), variant=(f[4] if f[4] not in ("", "-") else None), gpu=a.gpu, wall_ms=None,
                    kernels=[], engaged_int8=None, int8_share_pct=None, windows=0, nsys_version=None,
                    notes=a.nsys_note)
        _write_summary(a.outdir, summ, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
