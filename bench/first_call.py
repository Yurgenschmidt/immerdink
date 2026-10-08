#!/usr/bin/env python3
"""The first call at a new N: what a precision-phase switch costs the first time (DESIGN v1.19
§6g.6; NOTES-C1 IR10), next to re-packing an operand at that N.

    QUIET_GATE=1 tools/run_quiet.sh python3 bench/first_call.py --sweep --out first_call.jsonl
    python3 bench/first_call.py --N 10 --size 1024 --mode plain      # one fresh-process run
    python3 bench/first_call.py --table first_call.jsonl               # the tables, no GPU

--sweep runs one FRESH process per (shape, N, mode) and prints a table (TFLOPS-eq first, time
alongside); a single run prints one JSON line. Each line records its provenance: the
library's path and sha256, libimmerdink_host's sha256 and host API version
(imd_host_api_version of the library actually loaded), the git commit and whether tracked
files differ from it. In each process:
  1. warm-up at a base N (14, or 8 when N = 14): the CUDA context and every kernel a call at
     the base N launches are loaded, so what follows is a switch to an N never launched in
     the process, not process start-up (the plain mode also reports the process's very first
     plan and contract at the base N, "cold_process");
  2. plain mode: the first imd_plan_create at N and a second one (host time), the first
     imd_contract at N (host enqueue time, GPU time between CUDA events, and enqueue to
     completion), then the steady state (median of --reps event-timed calls); the first and
     steady imd_pack of the operand A at N (the re-pack a phase switch needs for a packed
     operand);
  3. graph mode: the contract at N captured (cudaStreamCaptureModeGlobal) before any launch
     at N (a cold capture), instantiated, then the first replay and the steady replay
     (median of --reps), each GPU time between events and host time to completion. NOTES-C1
     IR10's bar: a plan-time preload is needed if the cold capture fails or the first replay
     costs > 2x the steady replay.
"ik,kj->ij" m = n = k = --size, GENERATE vectors, g = N, RN; data (u - 1/2) exp(4 g).
TFLOPS-eq = 2 m n k / time (FP64 work the call replaces). Under CUDA_MODULE_LOADING=LAZY
(the default; set explicitly). Run on a quiet machine under the GPU lock (tools/run_quiet.sh).
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

os.environ["CUDA_MODULE_LOADING"] = os.environ.get("IMD_BENCH_MODULE_LOADING", "LAZY")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "tests")]
import harness as hz  # noqa: E402


class Rt:
    """The CUDA runtime calls this benchmark needs (libcudart of libimmerdink)."""

    def __init__(self):
        rt = self.rt = C.CDLL("libcudart.so.13")
        for name, args in {
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t], "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaDeviceSynchronize": [], "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamSynchronize": [C.c_void_p],
            "cudaEventCreate": [C.POINTER(C.c_void_p)], "cudaEventRecord": [C.c_void_p, C.c_void_p],
            "cudaEventSynchronize": [C.c_void_p],
            "cudaEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "cudaStreamBeginCapture": [C.c_void_p, C.c_int],
            "cudaStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "cudaGraphInstantiate": [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            "cudaGraphLaunch": [C.c_void_p, C.c_void_p],
        }.items():
            getattr(rt, name).argtypes = args
            getattr(rt, name).restype = C.c_int

    def ok(self, err, what):
        if err:
            raise RuntimeError(f"{what}: CUDA error {err}")

    def malloc(self, n):
        p = C.c_void_p()
        self.ok(self.rt.cudaMalloc(C.byref(p), max(int(n), 1)), "cudaMalloc")
        return p.value

    def event(self):
        e = C.c_void_p()
        self.ok(self.rt.cudaEventCreate(C.byref(e)), "cudaEventCreate")
        return e.value

    def elapsed_ms(self, a, b):
        self.ok(self.rt.cudaEventSynchronize(b), "cudaEventSynchronize")
        ms = C.c_float()
        self.ok(self.rt.cudaEventElapsedTime(C.byref(ms), a, b), "cudaEventElapsedTime")
        return ms.value


def run_one(lib_path, N, size, mode, reps):
    import numpy as np
    L = hz.Library(lib_path)
    lib = L.lib
    rt = Rt()
    m = n = k = size
    i, j, kk = 1, 2, 3
    tA, tB, tC = (hz.make_tensor(e, hz.dense(e), lab) for e, lab in
                  (([m, k], [i, kk]), ([k, n], [kk, j]), ([m, n], [i, j])))
    rng = np.random.default_rng(7)
    out = {"N": N, "size": size, "mode": mode, "flop": 2 * m * n * k,
           "module_loading": os.environ.get("CUDA_MODULE_LOADING")}
    s = C.c_void_p()
    rt.ok(rt.rt.cudaStreamCreate(C.byref(s)), "stream")
    s = s.value
    ev = [rt.event() for _ in range(2)]
    dev = {}
    for name, e in (("A", m * k), ("B", k * n), ("C", m * n)):
        dev[name] = rt.malloc(8 * e)
        x = ((rng.random(e) - 0.5) * np.exp(4.0 * rng.standard_normal(e))).astype(np.float64)
        rt.ok(rt.rt.cudaMemcpy(dev[name], x.ctypes.data, 8 * e, 1), "H2D")
    h = L.handle()

    def plan_at(Nx):
        p = C.c_void_p()
        t0 = time.perf_counter()
        st = lib.imd_plan_create(h, C.byref(tA), C.byref(tB), C.byref(tC),
                                 C.byref(hz.Options(Nx, 0, hz.RN, hz.EMULATE, 0, 0)), C.byref(p))
        dt = time.perf_counter() - t0
        if st:
            raise RuntimeError(f"imd_plan_create N={Nx}: {hz.STATUS[st]}")
        return p, dt

    def ws_at(Nx):
        return L.workspace_size(tA, tB, tC, Nx)[1]

    base = 8 if N == 14 else 14
    nws = max(ws_at(N), ws_at(base))
    ws = rt.malloc(nws)

    def contract(p, timed=True):
        """(host enqueue s, GPU ms, host enqueue-to-completion s)."""
        rt.ok(rt.rt.cudaEventRecord(ev[0], s), "record")
        t0 = time.perf_counter()
        st = lib.imd_contract(p, 1.0, dev["A"], dev["B"], 0.0, dev["C"], ws, nws, s)
        t1 = time.perf_counter()
        if st:
            raise RuntimeError(f"imd_contract: {hz.STATUS[st]}")
        rt.ok(rt.rt.cudaEventRecord(ev[1], s), "record")
        gpu = rt.elapsed_ms(ev[0], ev[1])
        return t1 - t0, gpu, time.perf_counter() - t0

    # 1. the process's first plan and contract (at the base N), then warm the base N
    pb_, t_plan0 = plan_at(base)
    first0 = contract(pb_)
    if mode == "plain":
        out["cold_process"] = {"N": base, "plan_create_s": t_plan0, "enqueue_s": first0[0],
                               "gpu_ms": first0[1], "total_s": first0[2]}
    for _ in range(3):
        contract(pb_)
    rt.ok(rt.rt.cudaDeviceSynchronize(), "sync")

    # 2. the switch to N (never launched in this process)
    p, t_plan = plan_at(N)
    p2, t_plan2 = plan_at(N)
    out["plan_create_first_s"], out["plan_create_again_s"] = t_plan, t_plan2
    if mode == "plain":
        first = contract(p)
        steady = [contract(p) for _ in range(reps)]
        out["contract_first"] = {"enqueue_s": first[0], "gpu_ms": first[1], "total_s": first[2]}
        out["contract_steady"] = {"enqueue_s": statistics.median(x[0] for x in steady),
                                  "gpu_ms": statistics.median(x[1] for x in steady),
                                  "total_s": statistics.median(x[2] for x in steady)}
        # re-pack cost: imd_pack of A at N (its pack kernels: first launch, then steady)
        klist = (C.c_int32 * 1)(kk)
        od = hz.PackDesc(C.sizeof(hz.PackDesc), N, hz.RN, hz.ORDER_AUTO, C.pointer(tA), 1,
                         klist, 0, None, 0, None)
        st, hp = L.pack_create(od)
        if st != "SUCCESS":
            raise RuntimeError(f"imd_pack_create: {st}")
        pbytes, pws = L.pack_bytes(hp)
        dp, dpw = rt.malloc(pbytes), rt.malloc(pws)

        def pack():
            rt.ok(rt.rt.cudaEventRecord(ev[0], s), "record")
            t0 = time.perf_counter()
            st = lib.imd_pack(hp, dev["A"], dp, pbytes, dpw, pws, s)
            t1 = time.perf_counter()
            if st:
                raise RuntimeError(f"imd_pack: {hz.STATUS[st]}")
            rt.ok(rt.rt.cudaEventRecord(ev[1], s), "record")
            return t1 - t0, rt.elapsed_ms(ev[0], ev[1]), time.perf_counter() - t0
        pf = pack()
        ps = [pack() for _ in range(reps)]
        out["pack_bytes"] = pbytes
        out["pack_first"] = {"enqueue_s": pf[0], "gpu_ms": pf[1], "total_s": pf[2]}
        out["pack_steady"] = {"enqueue_s": statistics.median(x[0] for x in ps),
                              "gpu_ms": statistics.median(x[1] for x in ps),
                              "total_s": statistics.median(x[2] for x in ps)}
        L.pack_destroy(hp)
    else:
        g, ge = C.c_void_p(), C.c_void_p()
        rt.ok(rt.rt.cudaStreamBeginCapture(s, 0), "begin capture")
        t0 = time.perf_counter()
        st = lib.imd_contract(p, 1.0, dev["A"], dev["B"], 0.0, dev["C"], ws, nws, s)
        t_cap = time.perf_counter() - t0
        end = rt.rt.cudaStreamEndCapture(s, C.byref(g))
        out["capture_status"] = hz.STATUS[st]
        out["end_capture_error"] = end
        if st or end:
            out["capture_ok"] = False
            return out
        t0 = time.perf_counter()
        rt.ok(rt.rt.cudaGraphInstantiate(C.byref(ge), g, 0), "instantiate")
        out["capture_ok"] = True
        out["capture_s"], out["instantiate_s"] = t_cap, time.perf_counter() - t0

        def replay():
            rt.ok(rt.rt.cudaEventRecord(ev[0], s), "record")
            t0 = time.perf_counter()
            rt.ok(rt.rt.cudaGraphLaunch(ge, s), "graph launch")
            rt.ok(rt.rt.cudaEventRecord(ev[1], s), "record")
            gpu = rt.elapsed_ms(ev[0], ev[1])
            return gpu, time.perf_counter() - t0
        first = replay()
        steady = [replay() for _ in range(reps)]
        out["replay_first"] = {"gpu_ms": first[0], "total_s": first[1]}
        out["replay_steady"] = {"gpu_ms": statistics.median(x[0] for x in steady),
                                "total_s": statistics.median(x[1] for x in steady)}
        out["first_over_steady_gpu"] = first[0] / out["replay_steady"]["gpu_ms"]
        out["first_over_steady_total"] = first[1] / out["replay_steady"]["total_s"]
    return out


def tflops(flop, seconds):
    return flop / seconds / 1e12 if seconds > 0 else float("nan")


def table(rows):
    """The sweep's tables. Every call time runs from just before the enqueue to completion
    (total_s), the same measure for the steady call (median) and the first one (DESIGN §6g.6
    publishes these); the GPU time between events follows in brackets."""
    def tm(x):
        return (f"{tflops(r['flop'], x['total_s']):>6.1f} {1e3 * x['total_s']:>7.3f} "
                f"({x['gpu_ms']:>7.3f})")

    print("plain: switch to a never-launched N (TFLOPS-eq first, time alongside; ms, GPU-event "
          "ms in brackets; extra = first - steady, us)")
    print(f"{'size':>5} {'N':>3} | {'plan 1st':>8} {'plan 2nd':>8} | {'steady TF':>9} {'ms':>7} "
          f"{'(GPU)':>9} | {'1st TF':>6} {'ms':>7} {'(GPU)':>9} | {'extra':>6} {'enqueue':>7} | "
          f"{'re-pack':>7} {'pack 1st':>8}")
    for r in rows:
        if r["mode"] != "plain":
            continue
        f, st = r["contract_first"], r["contract_steady"]
        print(f"{r['size']:>5} {r['N']:>3} | {1e3 * r['plan_create_first_s']:>8.3f} "
              f"{1e3 * r['plan_create_again_s']:>8.3f} | {'':>2}{tm(st)} | {tm(f)} | "
              f"{1e6 * (f['total_s'] - st['total_s']):>+6.1f} "
              f"{1e6 * (f['enqueue_s'] - st['enqueue_s']):>+7.1f} | "
              f"{r['pack_steady']['gpu_ms']:>7.3f} {1e3 * r['pack_first']['total_s']:>8.3f}")
    print("\ngraph: cold capture at a never-launched N, first vs steady replay")
    print(f"{'size':>5} {'N':>3} | {'capture':>7} | {'steady TF':>9} {'ms':>7} {'(GPU)':>9} | "
          f"{'1st TF':>6} {'ms':>7} {'(GPU)':>9} | {'1st/steady':>10} {'(GPU)':>6}")
    for r in rows:
        if r["mode"] != "graph":
            continue
        if not r.get("capture_ok"):
            print(f"{r['size']:>5} {r['N']:>3} | FAILED ({r.get('capture_status')}, "
                  f"end {r.get('end_capture_error')})")
            continue
        f, st = r["replay_first"], r["replay_steady"]
        print(f"{r['size']:>5} {r['N']:>3} | {'ok':>7} | {'':>2}{tm(st)} | {tm(f)} | "
              f"{r['first_over_steady_total']:>10.2f} ({r['first_over_steady_gpu']:>4.2f})")


def provenance(lib_path):
    """What produced a record: the library (path, sha256), libimmerdink_host next to it (its
    sha256 and host API, imd_host_api_version), and the source commit (git HEAD; "dirty" if
    tracked files differ from it)."""
    def sha(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def git(*args):
        r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    lib = os.path.realpath(lib_path)
    host_lib = os.path.join(os.path.dirname(lib), "libimmerdink_host.so")
    status = git("status", "--porcelain", "--untracked-files=no")
    host_api = None
    if os.path.exists(host_lib):
        f = C.CDLL(host_lib).imd_host_api_version
        f.restype, f.argtypes = C.c_uint32, []
        host_api = int(f())
    return {"lib": lib, "lib_sha256": sha(lib),
            "host_lib_sha256": sha(host_lib) if os.path.exists(host_lib) else None,
            "host_api": host_api, "commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--mode", choices=("plain", "graph"), default="plain")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--sweep", action="store_true",
                    help="one fresh process per (size, N, mode): sizes 1024 4096, N 6 8 10 14")
    ap.add_argument("--sizes", default="1024,4096")
    ap.add_argument("--Ns", default="6,8,10,14")
    ap.add_argument("--out", help="append the JSON lines here (--sweep)")
    ap.add_argument("--table", metavar="JSONL",
                    help="print the sweep's tables from these records (no GPU work)")
    a = ap.parse_args()
    if a.table:
        with open(a.table) as f:
            table([json.loads(line) for line in f if line.strip()])
        return 0
    if not a.sweep:
        print(json.dumps(run_one(a.lib, a.N, a.size, a.mode, a.reps) | provenance(a.lib)))
        return 0
    rows, prov = [], provenance(a.lib)
    for size in map(int, a.sizes.split(",")):
        for N in map(int, a.Ns.split(",")):
            for mode in ("plain", "graph"):
                r = subprocess.run([sys.executable, os.path.abspath(__file__), "--lib", a.lib,
                                    "--N", str(N), "--size", str(size), "--mode", mode,
                                    "--reps", str(a.reps)], capture_output=True, text=True)
                if r.returncode:
                    print(r.stdout, r.stderr, file=sys.stderr)
                    return 1
                rec = json.loads(r.stdout.strip().splitlines()[-1]) | prov
                rec["date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                rows.append(rec)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
    table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
