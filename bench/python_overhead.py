#!/usr/bin/env python3
"""Per-call host overhead of the Python device API (DESIGN §6c.7).

    flock <gpu.lock> tools/memcap.sh python3 bench/python_overhead.py [--lib build/libimmerdink.so]

A tiny contraction (default "ik,kj->ij" 16^3, N = 8) is enqueued in batches of --batch calls
(the GPU queue never fills; a device synchronization between batches is not timed) by:
  ctypes      lib.imd_contract(...) with prebuilt integer arguments: the bare C call from Python
  ptr         Plan.contract_ptr (integers; the C checks only)
  cai         Plan.contract with __cuda_array_interface__ objects (layout checks included)
  dlpack      Plan.contract with DLPack producers (hand-built capsules; producer cost included)
  einsum      immerdink.contract with a cached plan (CAI objects)
Methods are interleaved round by round; the median and minimum host time per call are
reported (µs), and the overhead over `ctypes`. Host time only: the call returns after
enqueuing its kernels. JSON with --json.
"""

import argparse
import ctypes as C
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), os.path.join(ROOT, "tests")]
import immerdink as imd  # noqa: E402
from test_python_device import Cai, Cuda, DLPackArray  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=None)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("-N", type=int, default=8)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    lib = imd.load_library(a.lib)
    cu = Cuda()
    n = a.size
    spec = "ik,kj->ij"
    p = [cu.malloc(8 * n * n) for _ in range(3)]
    for x in p:
        cu.rt.cudaMemset(x, 0, 8 * n * n)
    cai = [Cai(x, (n, n)) for x in p]
    dl = [DLPackArray(x, (n, n), [n, 1]) for x in p]
    plan = imd.Plan(spec, *cai, N=a.N)
    nws = plan.workspace_bytes
    ws = cu.malloc(nws)
    W = (ws, nws)
    cache = imd.PlanCache()
    imd.contract(spec, *cai, N=a.N, workspace=W, plan_cache=cache)
    f, h = lib.lib.imd_contract, plan._h

    def m_ctypes():
        f(h, 1.0, p[0], p[1], 0.0, p[2], ws, nws, 0)

    def m_ptr():
        plan.contract_ptr(1.0, p[0], p[1], 0.0, p[2], ws, nws, 0)

    def m_cai():
        plan.contract(cai[0], cai[1], cai[2], workspace=W)

    def m_dlpack():
        plan.contract(dl[0], dl[1], dl[2], workspace=W)

    def m_einsum():
        imd.contract(spec, cai[0], cai[1], cai[2], N=a.N, workspace=W, plan_cache=cache)

    methods = {"ctypes": m_ctypes, "ptr": m_ptr, "cai": m_cai, "dlpack": m_dlpack,
               "einsum": m_einsum}
    for fn in methods.values():   # warm up
        for _ in range(a.batch):
            fn()
        cu.sync()
    per = {k: [] for k in methods}
    rng = list(range(a.batch))
    for r in range(a.rounds):
        order = list(methods.items())
        if r % 2:
            order.reverse()
        for name, fn in order:
            t0 = time.perf_counter_ns()
            for _ in rng:
                fn()
            per[name].append((time.perf_counter_ns() - t0) / a.batch / 1e3)
            cu.sync()
    base = statistics.median(per["ctypes"])
    rows = []
    for name, v in per.items():
        med = statistics.median(v)
        rows.append({"method": name, "median_us": round(med, 3), "min_us": round(min(v), 3),
                     "overhead_us": round(med - base, 3)})
    if a.json:
        print(json.dumps({"spec": spec, "size": n, "N": a.N, "batch": a.batch,
                          "rounds": a.rounds, "rows": rows}))
    else:
        print(f"{spec} {n}^3 N={a.N}: host time per call, {a.rounds} rounds x {a.batch} calls")
        print(f"{'method':<8} {'median us':>10} {'min us':>8} {'vs ctypes':>10}")
        for r in rows:
            print(f"{r['method']:<8} {r['median_us']:>10.2f} {r['min_us']:>8.2f} "
                  f"{r['overhead_us']:>+10.2f}")
    cu.free_all()


if __name__ == "__main__":
    main()
