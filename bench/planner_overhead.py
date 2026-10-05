#!/usr/bin/env python3
"""Per-call cost of the planner queries from Python (DESIGN §6e). CPU only, no GPU.

    python3 bench/planner_overhead.py [--host-lib build/libimmerdink_host.so] [--floor PATH]
                                      [--cases ring,df3] [-N 8,14,17] [--json]

Run single-threaded on a quiet machine (the numbers are meaningless under load).

For each ehrenfest-like case (rank-4 CC tensors, DF 3-index, GEMM views) and N it prints
µs per call of
  floor      the C function in a C loop (bench/planner_floor.c, imd_planner_floor; ns -> µs)
  host.*     the host.py wrappers (workspace_size, lowering, plan_bytes, pack_desc_for,
             predict, predict_pack, rank), and their breakdown for predict:
               parse    spec + extents -> host.Desc (host._operands)
               structs  the ctypes descriptors, options, request and output (host._Call ...)
               call     the ctypes call with prebuilt arguments (C included)
               decode   imd_perf_t -> host.Prediction
  device     the host queries of the device API: immerdink.workspace_size on Layouts (the
             call ehrenfest makes before a contraction) and the host-side part of
             Plan creation (Layouts -> descriptors -> C structs -> imd_workspace_size)
  q.*        planner.Query (descriptor structs built once; lazy results; the shipped
             calibration loaded once and passed); q.predict.time_s / q.rank[0] include
             reading a field of the result
  pl.*       planner.Planner(spec, extents) lookups (query cache), memo off
  memo.*     planner.Planner with an opt-in Memo: an identical query (hit)
Times: median over --rounds rounds of --batch calls each, methods interleaved per round.
"""

import argparse
import ctypes as C
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from immerdink import host  # noqa: E402

try:
    from immerdink import planner  # noqa: E402
except ImportError:  # before DESIGN §6e
    planner = None

# name -> (spec, extents): ehrenfest-like shapes (o occupied, v virtual, Q auxiliary)
CASES = {
    "gemm16": ("ik,kj->ij", dict(i=16, j=16, k=16)),
    "gemm512": ("ik,kj->ij", dict(i=512, j=512, k=512)),
    "ring": ("kjbc,kiac->abij", dict(a=64, b=64, c=64, i=32, j=32, k=32)),
    "ppl": ("abef,efij->abij", dict(a=48, b=48, e=48, f=48, i=16, j=16)),
    "df3": ("Qia,Qjb->iajb", dict(Q=256, i=16, j=16, a=64, b=64)),
    "dftr": ("Qma,mi->Qia", dict(Q=256, m=80, a=64, i=16)),
}


def floor_lines(cases, Ns):
    out = []
    for name in cases:
        spec, ext = CASES[name]
        for N in Ns:
            descs = host.contraction(spec, ext)
            parts = [f"{name}@{N}", str(N), "0", "0"]
            for d in descs:
                parts += [str(len(d.extent))] + [str(x) for x in d.extent + d.stride + d.label]
            out.append(" ".join(parts))
    return "\n".join(out) + "\n"


FLOOR_COLS = ["workspace_size", "lowering", "plan_bytes", "pack_desc_for", "predict_null",
              "predict", "predict_pack", "rank", "min_moduli"]


def run_floor(path, cases, Ns):
    if not path or not os.path.exists(path):
        return {}
    r = subprocess.run([path], input=floor_lines(cases, Ns), capture_output=True, text=True,
                       check=True)
    out = {}
    for line in r.stdout.splitlines():
        f = line.split()
        out[f[0]] = {c: float(v) / 1e3 for c, v in zip(FLOOR_COLS, f[1:])}
    return out


def methods(spec, ext, N, h, cal):
    """name -> zero-argument callable."""
    lib = h
    descs = host.contraction(spec, ext)
    m = {}
    m["host.workspace_size"] = lambda: host.workspace_size(spec, ext, N=N, lib=lib)
    m["host.lowering"] = lambda: host.lowering(spec, ext, N=N, lib=lib)
    m["host.plan_bytes"] = lambda: host.plan_bytes(spec, ext, N=N, lib=lib)
    m["host.pack_desc_for"] = lambda: host.pack_desc_for(spec, ext, operand=0, N=N, lib=lib)
    m["host.predict"] = lambda: host.predict(spec, ext, N=N, lib=lib)
    m["host.predict(calib)"] = lambda: host.predict(spec, ext, N=N, calib=cal, lib=lib)
    pa = host.pack_desc_for(spec, ext, operand=0, N=N, lib=lib)
    m["host.predict_pack"] = lambda: host.predict_pack(pa, calib=cal, lib=lib)
    m["host.rank"] = lambda: host.rank(spec, ext, N=N, calib=cal, lib=lib)

    # breakdown of host.predict (calib passed)
    call = host._Call(descs, N, 0, "RN", "EMULATE", "GENERATE")
    req = host._PerfRequest(C.sizeof(host._PerfRequest), 0, -1, 0, -1.0)
    out = host._Perf()
    out.struct_size = C.sizeof(host._Perf)
    f = h.lib.imd_perf_predict
    args = call.args() + [None, C.byref(req), C.byref(cal._raw), C.byref(out)]
    st = f(*args)

    def structs():
        c = host._Call(descs, N, 0, "RN", "EMULATE", "GENERATE")
        q = host._PerfRequest(C.sizeof(host._PerfRequest), 0, -1, 0, -1.0)
        r = host._Perf()
        r.struct_size = C.sizeof(host._Perf)
        return c.args(), q, r

    m["  predict: parse"] = lambda: host._operands(spec, ext, None)
    m["  predict: structs"] = structs
    m["  predict: call"] = lambda: f(*args)
    m["  predict: decode"] = lambda: host._prediction(out, st)

    # device API host queries (Layouts: no array, no GPU)
    try:
        from immerdink import device
        lays = [device.Layout(d.extent) for d in descs]

        def dev_ws():   # immerdink.workspace_size without loading libimmerdink
            ds = device._descs(spec, [device._layout(x, n) for x, n in zip(lays, "ABC")])
            return host.workspace_size(ds, N=N, lib=lib)

        fws = h.lib["imd_workspace_size"]   # libimmerdink's declaration takes _capi structs
        fws.restype = C.c_int

        def plan_host():  # Plan._create's host-side part
            ds = device._descs(spec, [device._layout(x, n) for x, n in zip(lays, "ABC")])
            tens = [device._CTensor(d) for d in ds]
            opts = device._options(N, 0, "RN", "EMULATE", "GENERATE", False)
            ws = C.c_size_t(0)
            fws(*(C.byref(t.t) for t in tens), C.byref(opts), C.byref(ws))
            return ws.value

        m["device.workspace_size"] = dev_ws
        m["device.Plan host part"] = plan_host
    except Exception as e:  # noqa: BLE001
        print(f"(device API skipped: {e})", file=sys.stderr)

    if planner is not None:
        q = planner.Query(spec, ext, lib=lib)
        m["q.workspace_size"] = lambda: q.workspace_size(N)
        m["q.lowering"] = lambda: q.lowering(N)
        m["q.plan_bytes"] = lambda: q.plan_bytes(N)
        m["q.pack_desc_for"] = lambda: q.pack_desc_for(0, N)
        m["q.predict"] = lambda: q.predict(N)
        m["q.predict.time_s"] = lambda: q.predict(N).time_s
        m["q.predict_pack"] = lambda: q.predict_pack(pa)
        req = planner.Request(beta=True)
        m["q.predict(request)"] = lambda: q.predict(N, request=req)
        m["q.rank"] = lambda: q.rank(N)
        m["q.rank[0]"] = lambda: q.rank(N)[0].num_moduli
        acc = planner.Accuracy(target=1e-12)                 # v8 (DESIGN §6f)
        m["host.min_moduli"] = lambda: host.min_moduli(spec, ext, accuracy=acc, lib=lib)
        m["q.min_moduli"] = lambda: q.min_moduli(acc)
        m["q.min_moduli.N"] = lambda: q.min_moduli(acc).num_moduli
        pl = planner.Planner(lib=lib)
        m["pl.predict"] = lambda: pl.predict(spec, ext, N=N)
        m["pl.workspace_size"] = lambda: pl.workspace_size(spec, ext, N=N)
        m["pl.rank"] = lambda: pl.rank(spec, ext, N=N)
        from immerdink.device import Layout
        lays2 = [Layout(d.extent) for d in descs]
        m["pl.query_arrays.ws"] = lambda: pl.query_arrays(spec, *lays2).workspace_size(N)
        mp = planner.Planner(lib=lib, memo=planner.Memo(1024))
        m["memo.predict"] = lambda: mp.predict(spec, ext, N=N)
        m["memo.rank"] = lambda: mp.rank(spec, ext, N=N)
        m["memo.workspace_size"] = lambda: mp.workspace_size(spec, ext, N=N)
        m["memo.q.predict"] = (lambda qq: lambda: qq.predict(N))(mp.query(spec, ext))
        m["memo.min_moduli"] = lambda: mp.min_moduli(spec, ext, accuracy=acc)
    return m


def bench(ms, batch, rounds):
    names = list(ms)
    samples = {n: [] for n in names}
    for fn in ms.values():   # warm up
        for _ in range(3):
            fn()
    clock = time.perf_counter_ns
    for _ in range(rounds):
        for n in names:
            fn = ms[n]
            t0 = clock()
            for _ in range(batch):
                fn()
            samples[n].append((clock() - t0) / batch / 1e3)
    return {n: statistics.median(v) for n, v in samples.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", default=None)
    ap.add_argument("--floor", default=None, help="imd_planner_floor (default: next to the lib)")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("-N", default="8,14,17")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    h = host.load(a.host_lib)
    cal = host.Calibration.default(h)
    cases = a.cases.split(",")
    Ns = [int(x) for x in a.N.split(",")]
    floor_path = a.floor or os.path.join(os.path.dirname(h.path), "imd_planner_floor")
    floors = run_floor(floor_path, cases, Ns)
    results = {}
    for name in cases:
        spec, ext = CASES[name]
        for N in Ns:
            key = f"{name}@{N}"
            r = bench(methods(spec, ext, N, h, cal), a.batch, a.rounds)
            results[key] = {"python_us": r, "floor_us": floors.get(key, {})}
            if not a.json:
                fl = floors.get(key, {})
                print(f"\n{key}  {spec}  {ext}")
                print("  C floor (µs): " + "  ".join(f"{c} {v:.2f}" for c, v in fl.items()))
                for n, v in r.items():
                    print(f"  {n:<24} {v:8.2f} µs")
    if a.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    elif len(results) > 1:
        summary(results)


# entry point -> (host.py method, fast method, memo-hit method, C floor column)
SUMMARY = [("workspace_size", "host.workspace_size", "q.workspace_size", "memo.workspace_size",
            "workspace_size"),
           ("lowering", "host.lowering", "q.lowering", None, "lowering"),
           ("plan_bytes", "host.plan_bytes", "q.plan_bytes", None, "plan_bytes"),
           ("pack_desc_for", "host.pack_desc_for", "q.pack_desc_for", None, "pack_desc_for"),
           ("predict (calib None)", "host.predict", "q.predict", "memo.predict", "predict"),
           ("predict_pack", "host.predict_pack", "q.predict_pack", None, "predict_pack"),
           ("rank (one N, all g)", "host.rank", "q.rank", "memo.rank", "rank"),
           ("min_moduli (§6f)", "host.min_moduli", "q.min_moduli", "memo.min_moduli",
            "min_moduli"),
           ("device workspace_size", "device.workspace_size", "pl.query_arrays.ws", None,
            "workspace_size")]


def summary(results):
    def rng(vals):
        vals = [v for v in vals if v is not None]
        return f"{min(vals):6.2f}-{max(vals):<6.2f}" if vals else f"{'n/a':^13}"

    print(f"\nSummary over {len(results)} case x N points (µs per call, min-max):")
    print(f"  {'entry point':<22} {'C floor':>13} {'host.py':>13} {'planner':>13} {'memo hit':>13}")
    for name, old, new, memo, fl in SUMMARY:
        col = [[r["python_us"].get(k) if k else None for r in results.values()]
               for k in (old, new, memo)]
        flo = [r["floor_us"].get(fl) for r in results.values()]
        print(f"  {name:<22} {rng(flo):>13} {rng(col[0]):>13} {rng(col[1]):>13} {rng(col[2]):>13}")
    print("  (predict's C floor is with the calibration passed (copied and validated per "
          "call); with calib = NULL the C library uses its parsed-once shipped calibration: "
          + rng([r["floor_us"].get("predict_null") for r in results.values()]).strip() + " µs)")


if __name__ == "__main__":
    main()
