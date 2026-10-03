#!/usr/bin/env python3
"""Many live plans: per-plan memory and creation time in each vec_mode (DESIGN v1.2).

    LD_PRELOAD=build/tests/libimd_alloc_shim.so \\
        tools/run_bench.sh python3 bench/plans_probe.py [--lib build/libimmerdink.so] [--plans 32]

For each shape and mode, creates --plans plans of the same contraction (all alive at
once), then reports per plan: imd_plan_device_bytes, the device memory the driver actually
took (cudaMemGetInfo delta / plans, so allocation granularity included), the CUDA
allocations plan creation made (pinned host, device, managed: counted by the
tests/alloc_shim.c LD_PRELOAD shim), and the median creation time. JSON lines on stdout.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import harness as hz  # noqa: E402

I, J, K = 1, 2, 3
SHAPES = [  # name, m, n, k (matrix ik,jk->ij)
    ("sq1024", 1024, 1024, 1024),
    ("r_mm", 1000, 64, 50000),
    ("tall", 4_000_000, 64, 64),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--plans", type=int, default=32)
    ap.add_argument("--N", type=int, default=14)
    a = ap.parse_args()
    lib, cuda = hz.Library(a.lib), hz.Cuda()
    shim = os.environ.get("LD_PRELOAD", "")
    if "alloc_shim" in shim:
        lib.load_alloc_shim(shim)
    for name, m, n, k in SHAPES:
        A = hz.make_tensor([m, k], [k, 1], [I, K])
        B = hz.make_tensor([n, k], [k, 1], [J, K])
        C = hz.make_tensor([m, n], [n, 1], [I, J])
        for vec, vname in ((hz.VEC_GENERATE, "generate"), (hz.VEC_RESIDENT, "resident")):
            st, ws = lib.workspace_size(A, B, C, a.N, 0, hz.RN, hz.EMULATE, vec)
            assert st == "SUCCESS", st
            cuda.sync()
            free0, _ = cuda.mem_info()
            allocs0 = lib.alloc_count()
            plans, times = [], []
            for _ in range(a.plans):
                t0 = time.perf_counter()
                st, p = lib.plan_create(A, B, C, a.N, 0, hz.RN, hz.EMULATE, vec)
                times.append(time.perf_counter() - t0)
                assert st == "SUCCESS", st
                plans.append(p)
            free1, _ = cuda.mem_info()
            allocs = None if allocs0 is None else lib.alloc_count() - allocs0
            dev = [lib.plan_device_bytes(p)[1] for p in plans]
            for p in plans:
                lib.plan_destroy(p)
            print(json.dumps({
                "shape": name, "m": m, "n": n, "k": k, "N": a.N, "vec_mode": vname,
                "plans": a.plans, "workspace_bytes": ws,
                "plan_device_bytes": dev[0],
                "device_bytes_per_plan_measured": (free0 - free1) / a.plans,
                "cuda_allocs_per_plan": None if allocs is None else allocs / a.plans,
                # the only CUDA allocation a plan may make is the resident cudaMalloc
                "pinned_host_bytes_per_plan": 0 if allocs == a.plans * (vec == hz.VEC_RESIDENT) else None,
                "plan_create_ms_median": 1e3 * statistics.median(times),
            }), flush=True)


if __name__ == "__main__":
    main()
