#!/usr/bin/env python3
"""NOTES-C1 §4.3 M-d2: precision schedules timed on the GPU against their own [M] prediction
band (pre-registered protocol; DESIGN v1.19 §6g.9-§6g.10).

    python3 bench/accuracy/md2_time.py predict [--out PRED.json]                 # CPU only
    QUIET_GATE=1 tools/run_quiet.sh python3 bench/accuracy/md2_time.py run SYSTEM \\
        [--data normal|decay_opp] --out RECORDS.jsonl [--md2-cleared REASON]        # GPU
    python3 bench/accuracy/md2_time.py judge RECORDS.jsonl [--out SUMMARY.json]   # CPU only

ORDERING (binding, NOTES-C1 §5.9 "M-d2 ordering"). M-d2 runs only after M-c's MC-3 path
closes. `run` refuses the M-d2 systems (ppl, df, sq4096, jacobi) unless --md2-cleared names
why the ordering is met (recorded in every record); the tooling checks `smoke` (512^3) and
`smoke_t` (a small PPL-like tensor case), not M-d2 systems, run without it. If the
calibration changes (MC-3), the bars below are re-derived with `predict` and committed
BEFORE any M-d2 run (IR23); `judge` refuses records whose calibration is not the registered
one.

The protocol (pre-registered before any M-d2 measurement):
- Systems (NOTES-C1 §4.3): the o = 10, v = 96 PPL view ijcd,abcd->ijab (GEMM view
  100 x 9216 x 9216; invariant operand B, the integrals), DF acQ,bdQ->acbd 64^4, Q = 2032
  (invariant B), 4096^3 ik,kj->ij (invariant A; the squares' spec of the cited [M] figures,
  NOTES-C1 §5.9 MC-4), all GATED; the Jacobi view abcd,cdij->abij (1600 x 100 x 1600,
  invariant A, the matrix W) reported, not gated.
- Configuration per level: PrecisionPlan.configs() over levels (6, 8, 9, 10, 14), in-line,
  GENERATE, RN (M-b's view of the §2.4 schedules; it picks g = N at every level on these
  shapes). The packed runs pack the invariant operand with the in-line plan's own
  pack_spec (bit-identical to the in-line call; checked per level before timing).
- Runs (one rep = each once, in an order rotated by the rep index):
    S14 = 5 x N6 + 4 x N8 + 4 x N10 + 3 x N14 (NOTES-C1 §2.4), F14 = 16 x N14   [gated pair]
    S9  = 6 x N6 + 10 x N9, F9 = 16 x N9, F8 = 16 x N8                           [reported]
    P14 = S14 with the invariant operand packed into ONE buffer of the largest pack_bytes,
          packed before each phase (the initial pack and the three re-packs at the
          switches), PF14 = one pack at N = 14, then 16 packed calls                [reported]
  Every call: alpha = 1, beta = 0, the same operands (an iteration's operands are fixed in
  shape; their values do not change the GEMM work), one CUDA stream, the raw entry points
  Plan.contract_ptr / Pack.pack_ptr (the checked calls run in the bit-exact check).
- Timing: GPU events. Before each run a cudaMemsetAsync of a scratch buffer (--primer-mib,
  default 1024) keeps the GPU busy while the host enqueues the run, so the run's calls
  follow each other on the GPU without host gaps (queued_ahead records whether the whole
  run was enqueued before the primer finished); then an event before the first call and one
  after each call and each pack. A run's time is first event to last; per-call and per-pack
  times are consecutive differences.
- Conditions: one fresh process per (system, data), under the quiet gate, the GPU lock and
  memcap (tools/run_quiet.sh with QUIET_GATE=1), FREE only (ehrenfest busy check before
  each chunk, chunks <= 5 min); stock clocks, no clock lock (DESIGN §6c.5's regime, as M-c);
  a warm-up of >= 3 s of the system's own in-line N = 14 call (bench/warm_run.sh's 3 s),
  then one call at every level, in-line and packed, before the timed reps; --reps 15;
  if the worst spread (max - min) / median of the S14 and F14 totals exceeds 5 % (warm_run's
  BENCH_MAX_SPREAD), the reps are repeated once (2 attempts, M-c's BENCH_RETRIES) and the
  attempt with the smaller worst spread is kept. Clock state (nvidia-smi) after the warm-up
  and after the reps.
- Data: --data normal ((u - 1/2) exp(0.5 g), the bench default's distribution, seed 1) and
  --data decay_opp (bench/accuracy/c2_fixtures.py's family on the GEMM view, K in the spec's
  mode order; [KT26]'s data-dependent INT8 throughput). Both gated.
- The gated statistic: r = median over reps of the S14 totals / median of the F14 totals.
  Band: r_bar x (1 -/+ p90), p90 = DESIGN §6c.5's 90th-percentile target of the call class
  (>= 1 ms: 15 %; every level of every gated system is predicted >= 1 ms), and r <= 0.8.
  A system whose levels' predicted (DEFAULT/COLD) calls span classes takes the largest p90
  among them (the Jacobi view: < 100 us at N = 6, 8, 100 us - 1 ms at 9-14, so 30 %).
  r_bar = the registered bar (SYSTEMS' "bar": 0.750 / 0.664 / 0.673, NOTES-C1 §4.3, M-b;
  [M] shipped calibration, DEFAULT regime, COLD, GENERATE, g = N). A system passes when both data sets
  are inside; M-d2 passes when all three systems do. The Jacobi view's band (predicted 0.786,
  < 100 µs class, p90 30 %) is reported only.
- Reported: per phase of S14 measured vs predicted (DEFAULT/COLD and BURST/REPEAT), TFLOPS-eq
  first with time alongside; the paired per-rep S14/F14 ratios; S9/F9 and t14/t8, t14/t9
  (the gain against fixed N = 8-9); P14/PF14, P14/F14 and each re-pack's time against
  imd_perf_predict_pack; the bit-exact check; queued_ahead; spreads; clocks.
TFLOPS-eq = 2 m n k (the GEMM view; FP64 work the call replaces) / time.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "python"), HERE]

LEVELS = (6, 8, 9, 10, 14)
RUNS = {                         # name -> ([(iterations, N)], packed)
    "S14": ([(5, 6), (4, 8), (4, 10), (3, 14)], False),
    "F14": ([(16, 14)], False),
    "S9": ([(6, 6), (10, 9)], False),
    "F9": ([(16, 9)], False),
    "F8": ([(16, 8)], False),
    "P14": ([(5, 6), (4, 8), (4, 10), (3, 14)], True),
    "PF14": ([(16, 14)], True),
}
ORDER = ("S14", "F14", "S9", "F9", "F8", "P14", "PF14")

# name: spec, extents, invariant operand, GEMM view (m, n, k), B's layout (K modes first:
# "kn", else "nk"), registered bar (None: reported only), the class's p90 (DESIGN §6c.5)
SYSTEMS = {
    "ppl": dict(spec="ijcd,abcd->ijab", ext=dict(i=10, j=10, a=96, b=96, c=96, d=96),
                inv="B", mnk=(100, 9216, 9216), b_layout="nk", bar=0.750, p90=0.15),
    "df": dict(spec="acQ,bdQ->acbd", ext=dict(a=64, b=64, c=64, d=64, Q=2032),
               inv="B", mnk=(4096, 4096, 2032), b_layout="nk", bar=0.664, p90=0.15),
    "sq4096": dict(spec="ik,kj->ij", ext=dict(i=4096, k=4096, j=4096),
                   inv="A", mnk=(4096, 4096, 4096), b_layout="kn", bar=0.673, p90=0.15),
    "jacobi": dict(spec="abcd,cdij->abij", ext=dict(a=40, b=40, c=40, d=40, i=10, j=10),
                   inv="A", mnk=(1600, 100, 1600), b_layout="kn", bar=None, predicted=0.786,
                   p90=0.30),
    # tooling checks only, not M-d2 systems: a square with A invariant, and a small PPL-like
    # tensor contraction with B invariant (GEMM view 100 x 576 x 576)
    "smoke": dict(spec="ik,kj->ij", ext=dict(i=512, k=512, j=512),
                  inv="A", mnk=(512, 512, 512), b_layout="kn", bar=None, p90=None),
    "smoke_t": dict(spec="ijcd,abcd->ijab", ext=dict(i=10, j=10, a=24, b=24, c=24, d=24),
                    inv="B", mnk=(100, 576, 576), b_layout="nk", bar=None, p90=None),
}
MD2_SYSTEMS = ("ppl", "df", "sq4096", "jacobi")
GATED = ("ppl", "df", "sq4096")
CAP = 0.8
CLASSES = ((1e-3, ">= 1 ms", 0.15), (1e-4, "100 us - 1 ms", 0.20), (0.0, "< 100 us", 0.30))
# The calibration the bars were derived with (src/host/calib/rtx4090_sm89.calib, shipped
# since v1.18; NOTES-C1 §5.9 MC-1: unchanged after M-c).
REGISTERED_CALIB = "d31f62ca5e8d8893bed427f5a6e1a7905e1acf008ab1f4743cc2dcbb2a60e933"
MAX_SPREAD, ATTEMPTS, WARMUP_S, REPS, PHI, SEED = 5.0, 2, 3.0, 15, 0.5, 1


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def provenance(lib_path, host_lib_path):
    status = git("status", "--porcelain", "--untracked-files=no")
    calib = os.path.join(ROOT, "src", "host", "calib", "rtx4090_sm89.calib")
    out = {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
           "host_lib": os.path.realpath(host_lib_path), "host_lib_sha256": sha256(host_lib_path),
           "calib_file_sha256": sha256(calib),
           "driver_sha256": sha256(os.path.abspath(__file__))}
    if lib_path:
        out |= {"lib": os.path.realpath(lib_path), "lib_sha256": sha256(lib_path)}
    return out


def cls(t):
    return next((name, p) for lo, name, p in CLASSES if t >= lo)


def flops(name):
    m, n, k = SYSTEMS[name]["mnk"]
    return 2.0 * m * n * k


# ------------------------------------------------------------------------------------------
# [M] predictions (CPU)

def predictions(name, host_lib):
    """Per level: PrecisionPlan's configuration and the [M] times; the runs' predicted totals;
    the predicted ratios. DEFAULT/COLD is the bars' regime; BURST/REPEAT §6c.5's."""
    from immerdink import planner, precision
    s = SYSTEMS[name]
    pl = planner.Planner(lib=host_lib)
    q = pl.query(s["spec"], s["ext"])
    cf = precision.PrecisionPlan(pl, {name: (s["spec"], s["ext"])}, levels=LEVELS,
                                 vec_modes=("GENERATE",), monotone=False).configs()[name]
    op = 0 if s["inv"] == "A" else 1
    lv = {}
    for N in LEVELS:
        ch = cf[N]
        g, vec = ch.residue_group, ch.vec_mode
        spec = q.pack_desc_for(op, N, g, vec)
        packing = (spec, None) if op == 0 else (None, spec)
        pk = q.predict_pack(spec)
        lv[N] = dict(N=N, g=g, vec_mode=vec, status=ch.status,
                     t_cold=ch.time_s,                                   # = rank's, in-line
                     t_gN_cold=q.predict(N).time_s,                      # M-b's g = N view
                     t_burst_repeat=q.predict(N, g, vec, regime="BURST", cache="REPEAT").time_s,
                     t_packed_cold=q.predict(N, g, vec, packing=packing).time_s,
                     pack_cold=pk.time_s, pack_bytes=pk.pack_bytes,
                     workspace_bytes=ch.workspace_bytes)
        lv[N]["class"] = cls(lv[N]["t_cold"])[0]

    def tot(run, key):
        steps, packed = RUNS[run]
        t = sum(n * lv[N][key if not packed else "t_packed_cold"] for n, N in steps)
        if packed:
            t += sum(lv[N]["pack_cold"] for _, N in steps)
        return t
    totals = {k: {r: tot(r, k) for r in RUNS} for k in ("t_cold", "t_gN_cold", "t_burst_repeat")}
    rat = {k: dict(S14_F14=v["S14"] / v["F14"], S9_F9=v["S9"] / v["F9"],
                   P14_PF14=v["P14"] / v["PF14"], P14_F14=v["P14"] / v["F14"])
           for k, v in totals.items()}
    return dict(system=name, spec=s["spec"], ext=s["ext"], mnk=s["mnk"], invariant=s["inv"],
                flops_eq=flops(name), levels={str(N): v for N, v in lv.items()},
                totals=totals, ratios=rat)


def band(name):
    s = SYSTEMS[name]
    r = s["bar"] if s["bar"] is not None else s.get("predicted")
    if r is None:
        return None
    lo, hi = r * (1 - s["p90"]), r * (1 + s["p90"])
    if name in GATED:
        hi = min(hi, CAP)
    return dict(ratio=r, p90=s["p90"], lo=lo, hi=hi, cap=CAP if name in GATED else None,
                gated=name in GATED)


def cmd_predict(a):
    from immerdink import host
    hl = host.load(a.host_lib)
    out = dict(provenance=provenance(None, a.host_lib), systems={})
    ok = True
    print("M-d2 predictions [M] (shipped calibration; levels 6, 8, 9, 10, 14; in-line; "
          "GENERATE). TFLOPS-eq first, time alongside.")
    for name in MD2_SYSTEMS + ("smoke", "smoke_t"):
        p = predictions(name, hl)
        b = band(name)
        p["band"] = b
        out["systems"][name] = p
        print(f"\n{name}: {p['spec']} {p['ext']} (GEMM view {p['mnk']}), invariant "
              f"{p['invariant']}")
        print(f"  {'N':>3} {'g':>3} {'TF COLD':>8} {'ms':>8} | {'TF B/R':>7} {'ms':>8} | "
              f"{'packed ms':>9} {'pack ms':>8} {'class':>14}")
        for N in LEVELS:
            v = p["levels"][str(N)]
            fe = p["flops_eq"]
            print(f"  {N:>3} {v['g']:>3} {fe / v['t_cold'] * 1e-12:>8.2f} "
                  f"{1e3 * v['t_cold']:>8.3f} | {fe / v['t_burst_repeat'] * 1e-12:>7.2f} "
                  f"{1e3 * v['t_burst_repeat']:>8.3f} | {1e3 * v['t_packed_cold']:>9.3f} "
                  f"{1e3 * v['pack_cold']:>8.3f} {v['class']:>14}")
        r = p["ratios"]
        print(f"  S14/F14: {r['t_cold']['S14_F14']:.4f} (rank's choice, DEFAULT/COLD), "
              f"{r['t_gN_cold']['S14_F14']:.4f} (g = N), {r['t_burst_repeat']['S14_F14']:.4f} "
              f"(BURST/REPEAT); S9/F9 {r['t_cold']['S9_F9']:.4f}; P14/PF14 "
              f"{r['t_cold']['P14_PF14']:.4f}; P14/F14 {r['t_cold']['P14_F14']:.4f}")
        if b:
            print(f"  band: {b['ratio']:.3f} x (1 -/+ {b['p90']:.2f}) = [{b['lo']:.4f}, "
                  f"{b['hi']:.4f}]{' (capped at 0.8)' if b['cap'] else ''}"
                  f"{'' if b['gated'] else ', reported only'}")
        s = SYSTEMS[name]
        if name in GATED or name == "jacobi":
            reg = s["bar"] if s["bar"] is not None else s["predicted"]
            got = {k: round(r[k]["S14_F14"], 3) for k in ("t_cold", "t_gN_cold")}
            same = all(abs(v - reg) < 5e-4 for v in got.values())
            classes = {p["levels"][str(N)]["class"] for N in LEVELS}
            p90 = max(cls(p["levels"][str(N)]["t_cold"])[1] for N in LEVELS)
            p["registered_check"] = dict(registered=reg, derived=got, same=same,
                                         classes=sorted(classes), p90=p90,
                                         p90_same=p90 == s["p90"])
            print(f"  registered {reg:.3f}: derived {got} -> {'same' if same else 'DIFFERENT'}; "
                  f"classes {sorted(classes)} -> p90 {p90:.2f} "
                  f"{'as registered' if p90 == s['p90'] else 'DIFFERENT'}")
            ok &= same and p90 == s["p90"]
    calib_ok = out["provenance"]["calib_file_sha256"] == REGISTERED_CALIB
    out["registered_calib"] = REGISTERED_CALIB
    out["calib_is_registered"] = calib_ok
    out["bars_reproduced"] = bool(ok)
    print(f"\ncalibration {'is' if calib_ok else 'is NOT'} the registered one; bars " +
          ("reproduced" if ok else
           "NOT reproduced: re-derive and commit them before M-d2 (IR23)"))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if ok and calib_ok else 1


# ------------------------------------------------------------------------------------------
# The GPU run

class Rt:
    """The CUDA runtime calls this benchmark needs (libcudart of libimmerdink)."""

    def __init__(self):
        rt = self.rt = C.CDLL("libcudart.so.13")
        for fn, args in {
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t], "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaMemsetAsync": [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
            "cudaDeviceSynchronize": [], "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamSynchronize": [C.c_void_p],
            "cudaEventCreate": [C.POINTER(C.c_void_p)],
            "cudaEventRecord": [C.c_void_p, C.c_void_p],
            "cudaEventSynchronize": [C.c_void_p],
            "cudaEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "cudaGetErrorString": [C.c_int],
        }.items():
            getattr(rt, fn).argtypes = args
            getattr(rt, fn).restype = C.c_int
        rt.cudaGetErrorString.restype = C.c_char_p

    def ok(self, err, what):
        if err:
            raise RuntimeError(f"{what}: {self.rt.cudaGetErrorString(err).decode()}")

    def malloc(self, n):
        p = C.c_void_p()
        self.ok(self.rt.cudaMalloc(C.byref(p), max(int(n), 1)), "cudaMalloc")
        return p.value

    def upload(self, p, arr):
        self.ok(self.rt.cudaMemcpy(p, arr.ctypes.data, arr.nbytes, 1), "cudaMemcpy H2D")

    def download(self, arr, p):
        self.ok(self.rt.cudaMemcpy(arr.ctypes.data, p, arr.nbytes, 2), "cudaMemcpy D2H")

    def stream(self):
        s = C.c_void_p()
        self.ok(self.rt.cudaStreamCreate(C.byref(s)), "cudaStreamCreate")
        return s.value

    def event(self):
        e = C.c_void_p()
        self.ok(self.rt.cudaEventCreate(C.byref(e)), "cudaEventCreate")
        return e.value

    def record(self, e, s):
        self.ok(self.rt.cudaEventRecord(e, s), "cudaEventRecord")

    def ms(self, a, b):
        self.ok(self.rt.cudaEventSynchronize(b), "cudaEventSynchronize")
        x = C.c_float()
        self.ok(self.rt.cudaEventElapsedTime(C.byref(x), a, b), "cudaEventElapsedTime")
        return x.value

    def sync(self, s):
        self.ok(self.rt.cudaStreamSynchronize(s), "cudaStreamSynchronize")


def clock_state():
    q = ("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active,"
         "clocks_event_reasons.applications_clocks_setting,clocks_event_reasons.sw_power_cap,"
         "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown")
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:  # noqa: BLE001  (a record field, not a failure)
        return {"error": repr(e)}


def operands(name, data, seed):
    """(A, B) as dense host arrays of the spec's operand shapes."""
    import numpy as np
    s = SYSTEMS[name]
    lhs = s["spec"].split("->")[0].split(",")
    shapes = [tuple(s["ext"][c] for c in t) for t in lhs]
    m, n, k = s["mnk"]
    if data == "normal":
        out = []
        for x, sh in enumerate(shapes):
            r = np.random.default_rng([seed, x])
            size = math.prod(sh)
            out.append(((r.random(size) - 0.5) * np.exp(PHI * r.standard_normal(size)))
                       .reshape(sh))
        return out
    import c2_fixtures
    A, Bt = c2_fixtures.synthetic("decay_opp", m, n, k, seed)     # (m, k), (n, k)
    B = Bt if s["b_layout"] == "nk" else np.ascontiguousarray(Bt.T)
    return [np.ascontiguousarray(A).reshape(shapes[0]), B.reshape(shapes[1])]


def spread(xs):
    return 100.0 * (max(xs) - min(xs)) / statistics.median(xs)


def cmd_run(a):
    import numpy as np
    import immerdink as imd
    from immerdink import planner, precision
    name = a.system
    if name in MD2_SYSTEMS and not a.md2_cleared:
        print(f"{name} is an M-d2 system: M-d2 waits until M-c's MC-3 path closes (NOTES-C1 "
              "§5.9 'M-d2 ordering', binding). Pass --md2-cleared REASON once it has.",
              file=sys.stderr)
        return 2
    s = SYSTEMS[name]
    lib = imd.load_library(a.lib)
    hl = lib.host
    rec = dict(kind="md2_run", system=name, data=a.data, spec=s["spec"], ext=s["ext"],
               mnk=s["mnk"], invariant=s["inv"], flops_eq=flops(name), md2_cleared=a.md2_cleared,
               date=time.strftime("%Y-%m-%dT%H:%M:%S"), host_api=lib.version,
               protocol=dict(levels=LEVELS, runs={k: v[0] for k, v in RUNS.items()},
                             reps=a.reps, primer_mib=a.primer_mib, warmup_s=a.warmup_s,
                             max_spread_pct=MAX_SPREAD, attempts=a.attempts, seed=a.seed,
                             phi=PHI if a.data == "normal" else None))
    rec |= provenance(lib.path, os.path.join(os.path.dirname(lib.path), "libimmerdink_host.so"))
    pl = planner.Planner(lib=hl)
    cf = precision.PrecisionPlan(pl, {name: (s["spec"], s["ext"])}, levels=LEVELS,
                                 vec_modes=("GENERATE",), monotone=False).configs()[name]
    rt = Rt()
    st = rt.stream()
    A, B = operands(name, a.data, a.seed)
    lhs, rhs = s["spec"].split("->")
    sa, sb = (tuple(s["ext"][c] for c in t) for t in lhs.split(","))
    cshape = tuple(s["ext"][c] for c in rhs)
    assert A.shape == sa and B.shape == sb, (A.shape, sa, B.shape, sb)
    dA, dB, dC = rt.malloc(A.nbytes), rt.malloc(B.nbytes), rt.malloc(8 * math.prod(cshape))
    rt.upload(dA, np.ascontiguousarray(A))
    rt.upload(dB, np.ascontiguousarray(B))
    del A, B
    xA, xB, xC = imd.DeviceArray(dA, sa), imd.DeviceArray(dB, sb), imd.DeviceArray(dC, cshape)
    op = 0 if s["inv"] == "A" else 1
    X, dX = (xA, dA) if op == 0 else (xB, dB)
    plans, pplans, packs = {}, {}, {}
    for N in LEVELS:
        opts = cf[N].options()
        plans[N] = imd.Plan(s["spec"], xA, xB, xC, lib=lib, **opts)
        packs[N] = imd.Pack(plans[N].pack_spec(op), lib=lib)
        pk = (packs[N], None) if op == 0 else (None, packs[N])
        pplans[N] = imd.Plan(s["spec"], xA, xB, xC, lib=lib, packs=pk, **opts)
    nws = max(max(p.workspace_bytes for p in plans.values()),
              max(p.workspace_bytes for p in pplans.values()))
    npk = max(p.pack_bytes for p in packs.values())
    npw = max(p.workspace_bytes for p in packs.values())
    ws, pbuf, pws = rt.malloc(nws), rt.malloc(npk), rt.malloc(npw)
    primer_bytes = a.primer_mib << 20
    scratch = rt.malloc(primer_bytes)
    rec["levels"] = {str(N): dict(N=N, g=cf[N].residue_group, vec_mode=cf[N].vec_mode,
                                  workspace_bytes=plans[N].workspace_bytes,
                                  packed_workspace_bytes=pplans[N].workspace_bytes,
                                  pack_bytes=packs[N].pack_bytes,
                                  pack_workspace_bytes=packs[N].workspace_bytes)
                     for N in LEVELS}
    rec["buffers"] = dict(workspace=nws, packed=npk, pack_workspace=npw, primer=primer_bytes)
    wsa = imd.DeviceArray(ws, (nws,), dtype="u1")
    pwa = imd.DeviceArray(pws, (npw,), dtype="u1")

    # the bit-exact check: the checked calls, in-line against packed (PackedBuffer), per level
    pb = imd.PackedBuffer(pbuf, npk)
    nC = math.prod(cshape)
    ref, got = np.empty(nC), np.empty(nC)
    rec["bitexact_packed_vs_inline"] = {}
    for N in LEVELS:
        plans[N].contract(xA, xB, xC, workspace=wsa, stream=st)
        rt.sync(st)
        rt.download(ref, dC)
        packs[N].pack(X, pb, workspace=pwa, stream=st)
        pplans[N].contract(xA, xB, xC, workspace=wsa, stream=st,
                           packed=(pb, None) if op == 0 else (None, pb))
        rt.sync(st)
        rt.download(got, dC)
        rec["bitexact_packed_vs_inline"][str(N)] = bool(np.array_equal(ref.view(np.uint64),
                                                                       got.view(np.uint64)))

    def contract(N, packed):
        if packed:
            pr = (pbuf, packs[N].pack_bytes)
            pplans[N].contract_ptr(1.0, dA, dB, 0.0, dC, ws, nws, st,
                                   packed=(pr, None) if op == 0 else (None, pr))
        else:
            plans[N].contract_ptr(1.0, dA, dB, 0.0, dC, ws, nws, st)

    def pack(N):
        packs[N].pack_ptr(dX, pbuf, packs[N].pack_bytes, pws, npw, st)

    # warm-up: the system's own in-line N = 14 call for >= warmup_s, then every level once
    t0, nw = time.perf_counter(), 0
    while True:
        for _ in range(8):
            contract(14, False)
        nw += 8
        rt.sync(st)
        if time.perf_counter() - t0 >= a.warmup_s:
            break
    for N in LEVELS:
        contract(N, False)
        pack(N)
        contract(N, True)
    rt.sync(st)
    rec["warmup"] = dict(seconds=time.perf_counter() - t0, calls=nw)
    rec["clock_after_warmup"] = clock_state()

    ev = [rt.event() for _ in range(2 + max(sum(c for c, _ in v[0]) + len(v[0])
                                            for v in RUNS.values()))]

    def one(run):
        steps, packed = RUNS[run]
        rt.record(ev[0], st)
        rt.ok(rt.rt.cudaMemsetAsync(scratch, 0, primer_bytes, st), "cudaMemsetAsync")
        rt.record(ev[1], st)
        h0 = time.perf_counter()
        i, marks = 2, []
        for count, N in steps:
            if packed:
                pack(N)
                rt.record(ev[i], st)
                marks.append(("pack", N))
                i += 1
            for _ in range(count):
                contract(N, packed)
                rt.record(ev[i], st)
                marks.append(("call", N))
                i += 1
        enq = time.perf_counter() - h0
        rt.sync(st)
        primer = rt.ms(ev[0], ev[1])
        steps_ms = [rt.ms(ev[j - 1], ev[j]) for j in range(2, i)]
        calls = [(N, t) for (kind, N), t in zip(marks, steps_ms) if kind == "call"]
        packs_ms = [(N, t) for (kind, N), t in zip(marks, steps_ms) if kind == "pack"]
        return dict(total_ms=rt.ms(ev[1], ev[i - 1]), primer_ms=primer, enqueue_ms=1e3 * enq,
                    queued_ahead=1e3 * enq < primer, calls=calls, packs=packs_ms)

    attempts = []
    for at in range(a.attempts):
        reps = []
        for r in range(a.reps):
            order = ORDER[r % len(ORDER):] + ORDER[:r % len(ORDER)]
            reps.append({run: one(run) for run in order} | {"order": list(order)})
        worst = max(spread([x[run]["total_ms"] for x in reps]) for run in ("S14", "F14"))
        attempts.append(dict(reps=reps, worst_spread_pct=worst))
        if worst <= MAX_SPREAD:
            break
    best = min(attempts, key=lambda x: x["worst_spread_pct"])
    rec["attempts"] = [dict(worst_spread_pct=x["worst_spread_pct"]) for x in attempts]
    rec["kept_attempt"] = attempts.index(best)
    rec["contaminated"] = best["worst_spread_pct"] > MAX_SPREAD
    rec["reps"] = best["reps"]
    rec["clock_after_reps"] = clock_state()
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    med = {run: statistics.median(x[run]["total_ms"] for x in rec["reps"]) for run in RUNS}
    fe = rec["flops_eq"]
    print(f"{name} {a.data}: " + ", ".join(
        f"{run} {16 * fe / med[run] * 1e-9:.1f} TF-eq ({med[run]:.3f} ms)" for run in ORDER) +
        f"; S14/F14 {med['S14'] / med['F14']:.4f}; bit-exact "
        f"{all(rec['bitexact_packed_vs_inline'].values())}; worst spread "
        f"{best['worst_spread_pct']:.2f} %")
    return 0


# ------------------------------------------------------------------------------------------
# The judge (CPU)

def cmd_judge(a):
    from immerdink import host
    recs = []
    with open(a.records) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("kind") == "md2_run":
                    recs.append(d)
    hl = host.load(a.host_lib)
    preds, rows, verdict = {}, [], {}
    for r in recs:
        name = r["system"]
        if name in MD2_SYSTEMS and r.get("calib_file_sha256") != REGISTERED_CALIB:
            print(f"{name} {r['data']}: calibration {r.get('calib_file_sha256')} is not the "
                  "registered one; re-derive and commit the bars first (IR23)", file=sys.stderr)
            return 2
        if name not in preds:
            preds[name] = predictions(name, hl)
        p = preds[name]
        reps = r["reps"]
        med = {run: statistics.median(x[run]["total_ms"] for x in reps) for run in RUNS}
        ratio = med["S14"] / med["F14"]
        paired = [x["S14"]["total_ms"] / x["F14"]["total_ms"] for x in reps]
        b = band(name)
        inside = None if b is None else b["lo"] <= ratio <= b["hi"]
        per_level = {}
        for N in LEVELS:      # every in-line call at N over the reps and runs
            ts = [t for x in reps for run in ("S14", "F14", "S9", "F9", "F8")
                  for n_, t in x[run]["calls"] if n_ == N]
            per_level[N] = statistics.median(ts)
        phases = []
        for count, N in RUNS["S14"][0]:
            ph = statistics.median(sum(t for n_, t in x["S14"]["calls"] if n_ == N) for x in reps)
            lv = p["levels"][str(N)]
            phases.append(dict(N=N, calls=count, ms=ph, tflops_eq=count * r["flops_eq"] / ph * 1e-9,
                               pred_cold_ms=1e3 * count * lv["t_cold"],
                               pred_burst_repeat_ms=1e3 * count * lv["t_burst_repeat"]))
        repack = []
        for N in (6, 8, 10, 14):
            ts = [t for x in reps for n_, t in x["P14"]["packs"] if n_ == N]
            repack.append(dict(N=N, ms=statistics.median(ts),
                               pred_ms=1e3 * p["levels"][str(N)]["pack_cold"]))
        row = dict(system=name, data=r["data"], commit=r["commit"], lib_sha256=r["lib_sha256"],
                   host_api=r["host_api"], median_ms=med,
                   tflops_eq={run: 16 * r["flops_eq"] / med[run] * 1e-9 for run in RUNS},
                   S14_F14=ratio, paired_median=statistics.median(paired),
                   paired_min=min(paired), paired_max=max(paired),
                   pred_S14_F14=dict(cold=p["ratios"]["t_cold"]["S14_F14"],
                                     burst_repeat=p["ratios"]["t_burst_repeat"]["S14_F14"]),
                   band=b, inside=inside,
                   S9_F9=med["S9"] / med["F9"], pred_S9_F9=p["ratios"]["t_cold"]["S9_F9"],
                   t14_t8=per_level[14] / per_level[8], t14_t9=per_level[14] / per_level[9],
                   P14_PF14=med["P14"] / med["PF14"], P14_F14=med["P14"] / med["F14"],
                   pred_P14_PF14=p["ratios"]["t_cold"]["P14_PF14"],
                   per_level_ms=per_level, phases=phases, repack=repack,
                   bitexact=all(r["bitexact_packed_vs_inline"].values()),
                   queued_ahead=sum(x[run]["queued_ahead"] for x in reps for run in RUNS) /
                   (len(reps) * len(RUNS)),
                   worst_spread_pct=r["attempts"][r["kept_attempt"]]["worst_spread_pct"],
                   contaminated=r["contaminated"], clock=r.get("clock_after_reps"))
        rows.append(row)
        if b is not None and b["gated"]:
            verdict.setdefault(name, []).append(inside)
    for row in rows:
        print(f"\n## {row['system']} ({row['data']}; commit {row['commit'][:7]}, lib "
              f"{row['lib_sha256'][:8]}…, host API {row['host_api']})")
        print("run   | TFLOPS-eq | ms (median of the reps)")
        for run in ORDER:
            print(f"{run:5s} | {row['tflops_eq'][run]:9.2f} | {row['median_ms'][run]:.3f}")
        b = row["band"]
        print(f"S14/F14 {row['S14_F14']:.4f} (paired median {row['paired_median']:.4f}, "
              f"{row['paired_min']:.4f}-{row['paired_max']:.4f}); predicted "
              f"{row['pred_S14_F14']['cold']:.4f} (DEFAULT/COLD), "
              f"{row['pred_S14_F14']['burst_repeat']:.4f} (BURST/REPEAT)" +
              ("" if b is None else f"; band [{b['lo']:.4f}, {b['hi']:.4f}] -> "
               f"{'INSIDE' if row['inside'] else 'OUTSIDE'}"
               f"{'' if b['gated'] else ' (reported)'}"))
        print("phase of S14 | TFLOPS-eq | ms | predicted ms DEFAULT/COLD (err) | "
              "BURST/REPEAT (err)")
        for ph in row["phases"]:
            print(f"{ph['calls']} x N{ph['N']:<2}      | {ph['tflops_eq']:9.2f} | "
                  f"{ph['ms']:.3f} | {ph['pred_cold_ms']:.3f} "
                  f"({100 * (ph['pred_cold_ms'] / ph['ms'] - 1):+.1f} %) | "
                  f"{ph['pred_burst_repeat_ms']:.3f} "
                  f"({100 * (ph['pred_burst_repeat_ms'] / ph['ms'] - 1):+.1f} %)")
        print(f"S9/F9 {row['S9_F9']:.4f} (pred {row['pred_S9_F9']:.4f}); t14/t8 "
              f"{row['t14_t8']:.3f}, t14/t9 {row['t14_t9']:.3f}; P14/PF14 {row['P14_PF14']:.4f} "
              f"(pred {row['pred_P14_PF14']:.4f}), P14/F14 {row['P14_F14']:.4f}")
        print("re-packs (ms, predicted): " + ", ".join(
            f"N{x['N']} {x['ms']:.3f} ({x['pred_ms']:.3f})" for x in row["repack"]))
        print(f"bit-exact packed = in-line: {row['bitexact']}; queued ahead "
              f"{100 * row['queued_ahead']:.0f} % of runs; worst spread "
              f"{row['worst_spread_pct']:.2f} %{' CONTAMINATED' if row['contaminated'] else ''}")
    final = {}
    for name in GATED:
        v = verdict.get(name, [])
        final[name] = ("PASS" if len(v) == 2 and all(v) else
                       "FAIL" if any(x is False for x in v) else "INCOMPLETE")
    md2 = ("PASS" if all(x == "PASS" for x in final.values()) else
           "FAIL" if any(x == "FAIL" for x in final.values()) else "INCOMPLETE")
    print(f"\nper system: {final}; M-d2: {md2}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(rows=rows, systems=final, md2=md2,
                           provenance=provenance(None, a.host_lib)), f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    host_default = os.path.join(ROOT, "build", "libimmerdink_host.so")
    p = sub.add_parser("predict", help="the [M] predictions, bars and bands (CPU)")
    p.add_argument("--host-lib", default=host_default)
    p.add_argument("--out")
    r = sub.add_parser("run", help="time one system on the GPU (one fresh process)")
    r.add_argument("system", choices=sorted(SYSTEMS))
    r.add_argument("--data", choices=("normal", "decay_opp"), default="normal")
    r.add_argument("--lib", default=os.path.join(ROOT, "build", "libimmerdink.so"))
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--attempts", type=int, default=ATTEMPTS)
    r.add_argument("--warmup-s", type=float, default=WARMUP_S)
    r.add_argument("--primer-mib", type=int, default=1024)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--md2-cleared", default=None,
                   help="why M-d2's ordering (NOTES-C1 §5.9) is met; required for the M-d2 systems")
    j = sub.add_parser("judge", help="evaluate run records against the bands (CPU)")
    j.add_argument("records")
    j.add_argument("--host-lib", default=host_default)
    j.add_argument("--out")
    a = ap.parse_args()
    return {"predict": cmd_predict, "run": cmd_run, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
