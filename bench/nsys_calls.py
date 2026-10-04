#!/usr/bin/env python3
"""Per-call breakdown of the bench's timed calls from an nsys sqlite export (A5).

    bench/nsys_calls.py run.sqlite [--skip 5] [--json]

The bench times each call as  cudaEventRecord(e0); run(); cudaEventRecord(e1);
cudaEventSynchronize(e1)  on an idle GPU. For every such window this collects the launches
(kernels and memsets, by correlation id) and reports, as medians over the windows of each
method (imd if any immerdink kernel ran, else other, e.g. the bw copy), in µs:
  host    host time inside run()            (e1 record start - e0 record end)
  pre     host time before the first launch (first launch API start - e0 record end)
  lat0    e0 record end -> first kernel start on the GPU
  span    first kernel start -> last kernel end
  kern    sum of kernel (and memset) durations; gaps = span - kern
  wall    e0 record start -> max(last kernel end, e1 record end): ~ the event time
and the median duration of each kernel by position. GPU context switches inside a window
(GPU_CONTEXT_SWITCH_EVENTS, nsys --gpuctxsw=true) are counted per window ("ctxsw").
"""
import argparse
import json
import sqlite3
import statistics
import sys

IMD = ("kf_scale_pack", "k4_kernel", "k5_crt", "k0_vec", "k1_scale", "k1_tiled", "k23_pack",
       "k23_zero_rows", "kl_", "k5_beta_only", "k_direct", "k45_")


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=5, help="windows to skip per method (warm)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seq", type=int, default=0, help="print wall/lat0 of the first SEQ calls")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    names = dict(c.execute("select id, value from StringIds"))
    # Correlation ids are per process: key launches and activities by (process, id).
    rt = [(s, e, names[n], (t >> 24, corr)) for s, e, n, corr, t in
          c.execute("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by globalTid, start")]
    gpu = {}
    for s, e, corr, n, gx, gy, gz, bx, p in c.execute(
            "select start, end, correlationId, shortName, gridX, gridY, gridZ, blockX, globalPid from CUPTI_ACTIVITY_KIND_KERNEL"):
        gpu[(p >> 24, corr)] = (s, e, names[n], f"{gx}x{gy}x{gz}/{bx}")
    for s, e, corr, p in c.execute("select start, end, correlationId, globalPid from CUPTI_ACTIVITY_KIND_MEMSET"):
        gpu[(p >> 24, corr)] = (s, e, "memset", "")
    ctx = [t for (t,) in c.execute("select timestamp from GPU_CONTEXT_SWITCH_EVENTS order by timestamp")] \
        if c.execute("select count(*) from sqlite_master where name='GPU_CONTEXT_SWITCH_EVENTS'").fetchone()[0] else []

    calls = {}
    i = 0
    while i < len(rt):
        if not rt[i][2].startswith("cudaEventRecord"):
            i += 1
            continue
        j = i + 1
        while j < len(rt) and not rt[j][2].startswith("cudaEventRecord"):
            j += 1
        if j + 1 >= len(rt) or not rt[j + 1][2].startswith("cudaEventSynchronize"):
            i = j
            continue
        e0, e1 = rt[i], rt[j]
        ks = sorted(gpu[r[3]] for r in rt[i + 1:j] if r[3] in gpu)
        launches = [r for r in rt[i + 1:j] if r[3] in gpu]
        i = j + 2
        if not ks:
            continue
        method = "imd" if any(k[2].startswith(IMD) for k in ks) else "other"
        span = ks[-1][1] - ks[0][0]
        kern = sum(k[1] - k[0] for k in ks)
        w = dict(host=(e1[0] - e0[1]) / 1e3, pre=(launches[0][0] - e0[1]) / 1e3,
                 lat0=(ks[0][0] - e0[1]) / 1e3, span=span / 1e3, kern=kern / 1e3,
                 gaps=(span - kern) / 1e3, wall=(max(ks[-1][1], e1[1]) - e0[0]) / 1e3,
                 ctxsw=sum(1 for t in ctx if e0[0] <= t <= max(ks[-1][1], e1[1])),
                 seq=[(k[2], k[3]) for k in ks], kt=[(k[1] - k[0]) / 1e3 for k in ks],
                 gap_list=[(ks[q + 1][0] - ks[q][1]) / 1e3 for q in range(len(ks) - 1)])
        calls.setdefault((method, tuple(s for s, _ in w["seq"])), []).append(w)

    if a.seq:
        for (method, seq), ws in calls.items():
            print(method, len(seq), " ".join(f"{w['wall']:.0f}/{w['lat0']:.0f}" for w in ws[:a.seq]))
    out = []
    for (method, seq), ws in calls.items():
        ws = ws[a.skip:] if len(ws) > a.skip + 5 else ws
        r = dict(method=method, calls=len(ws), launches=len(seq))
        for k in ("host", "pre", "lat0", "span", "kern", "gaps", "wall"):
            r[k] = round(med([w[k] for w in ws]), 2)
        r["ctxsw_windows"] = sum(1 for w in ws if w["ctxsw"])
        r["kernels"] = [(seq[q], ws[0]["seq"][q][1], round(med([w["kt"][q] for w in ws]), 2))
                        for q in range(len(seq))]
        r["gap_each"] = [round(med([w["gap_list"][q] for w in ws]), 2) for q in range(len(seq) - 1)]
        out.append(r)
    if a.json:
        json.dump(out, sys.stdout, indent=1)
        print()
        return
    for r in out:
        print(f"{r['method']:8s} calls {r['calls']:4d} launches {r['launches']}: wall {r['wall']:.1f}  "
              f"host {r['host']:.1f} pre {r['pre']:.1f} lat0 {r['lat0']:.1f}  span {r['span']:.1f} = "
              f"kern {r['kern']:.1f} + gaps {r['gaps']:.1f}  (ctxsw in {r['ctxsw_windows']} calls)")
        print("    " + ", ".join(f"{n}[{g}] {t}" for n, g, t in r["kernels"]))
        print("    gaps " + " ".join(str(g) for g in r["gap_each"]))


if __name__ == "__main__":
    main()
