#!/usr/bin/env python3
"""GPU calibration: soak the GPU under a sustained dense load until its clock and temperature are
stable, report the steady state and recommend a clock lock (see bench/calibrate.sh).

The load is imd_bench running a 4096^3 contraction at N = 14 back to back, started through
tools/run_bench.sh (machine-wide lock, quiet gate, shared GPU lock), while nvidia-smi is sampled at
1 Hz. The state is STABLE when, over the last 20 s of the loaded run (and at least 30 s after the
load began), the SM clock range is <= 30 MHz and the temperature changed by <= 1 C. The load is then
stopped. Recommended lock = the lowest SM clock of the last 30 s minus 30 MHz (two steps of
margin below the worst sustained dip), floored to a 15 MHz step (same rule as the 4090 study).
"""
import argparse, json, math, os, signal, subprocess, sys, time

FIELDS = ("clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks.max.sm,"
          "clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,"
          "clocks_event_reasons.hw_slowdown,clocks_event_reasons.applications_clocks_setting,"
          "pstate")


def gpu_id():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v else "0"


def sample(gid):
    try:
        out = subprocess.run(["nvidia-smi", "-i", gid, f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=10).stdout.strip()
    except Exception:
        return None
    f = [x.strip() for x in out.split(",")]
    if len(f) < 10:
        return None
    def num(x):
        try:
            return float(x)
        except ValueError:
            return math.nan
    act = lambda x: x.lower().startswith("active")
    return {"t": time.time(), "sm": num(f[0]), "power": num(f[1]), "temp": num(f[2]), "util": num(f[3]),
            "sm_max": num(f[4]), "cap": act(f[5]), "thermal": act(f[6]), "hw": act(f[7]),
            "appclk": act(f[8]), "pstate": f[9]}


def rng(v):
    return max(v) - min(v) if v else math.nan


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if s else math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="path of imd_bench")
    ap.add_argument("--runner", required=True, help="tools/run_bench.sh")
    ap.add_argument("--out", required=True, help="output directory (calibration.json)")
    ap.add_argument("--max-seconds", type=float, default=180, help="timeout of the loaded phase (default 180)")
    ap.add_argument("--spread-mhz", type=float, default=30)
    ap.add_argument("--spread-c", type=float, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gid = gpu_id()

    first = sample(gid)
    if first is None:
        print("calibrate: nvidia-smi gives no sample for GPU " + gid, file=sys.stderr)
        return 1
    idle = {"sm": first["sm"], "sm_max": first["sm_max"], "temp": first["temp"], "power": first["power"],
            "pstate": first["pstate"], "appclk_reason": first["appclk"]}

    # The load detector below is "utilization >= 80 %": it cannot tell our load from someone else's. So first make
    # sure nobody else is on the GPU, or calibration would measure a foreign workload (and report its clock/power
    # as ours while our own runner waits at the quiet gate).
    pre = []
    for _ in range(6):
        t = sample(gid)
        if t is not None:
            pre.append(t["util"])
        time.sleep(0.5)
    busy = sum(pre) / len(pre) if pre else 0.0
    if busy >= 5.0:
        msg = (f"calibrate: GPU {gid} is already busy before our load (utilization {busy:.0f} %, idle clock "
               f"{idle['sm']:.0f} MHz, {idle['pstate']}): another process is using it (shared GPU?). "
               "Calibration would measure that workload, not ours; not calibrating.")
        print(msg, file=sys.stderr, flush=True)
        with open(os.path.join(a.out, "calibration.json"), "w") as fh:
            json.dump({"valid": False, "gpu_busy_before_load_pct": busy, "idle": idle, "reason": msg}, fh, indent=2)
        return 5

    cmd = [a.runner, a.bench, "--name", "calibrate", "--spec", "ik,jk->ij", "--ext", "i=4096,j=4096,k=4096",
           "--methods", "imd", "--N", "14", "--reps", "10000000", "--warmup", "2", "--max-seconds",
           str(a.max_seconds + 600), "--phase-reps", "0", "--samples", "16", "--out", "/dev/null"]
    env = dict(os.environ, QUIET_LOG=os.path.join(a.out, "calibrate.quiet"))
    print(f"calibrate: sustained dense load (4096^3, N=14) on GPU {gid}; idle SM clock {idle['sm']:.0f} MHz "
          f"(max {idle['sm_max']:.0f}), {idle['temp']:.0f} C", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    samples, t_load = [], None
    stable_at, rc = None, None
    try:
        while True:
            time.sleep(1.0)
            rc = proc.poll()
            if rc is not None:
                break
            s = sample(gid)
            if s is None:
                continue
            if t_load is None:
                if s["util"] >= 80:
                    t_load = s["t"]
                else:
                    continue
            samples.append(s)
            el = s["t"] - t_load
            if len(samples) >= 30:
                w20 = samples[-20:]
                if rng([x["sm"] for x in w20]) <= a.spread_mhz and rng([x["temp"] for x in w20]) <= a.spread_c:
                    stable_at = el
                    break
            if el >= a.max_seconds:
                break
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    if rc == 3:
        print("calibrate: the machine never got quiet (tools/run_bench.sh exit 3); see calibrate.quiet", file=sys.stderr)
        return 3
    if len(samples) < 5:
        print(f"calibrate: the load did not run long enough to sample (runner exit {rc})", file=sys.stderr)
        return 1

    win = samples[-30:]
    sm = [x["sm"] for x in win]
    steady_med, steady_min, steady_max = median(sm), min(sm), max(sm)
    share = lambda key, v: sum(1 for x in v if x[key]) / len(v)
    res = {
        "stable": stable_at is not None,
        "seconds_to_stable": stable_at,
        "loaded_seconds": samples[-1]["t"] - t_load,
        "gpu": gid,
        "idle": idle,
        "steady_sm_mhz": {"median": steady_med, "min": steady_min, "max": steady_max, "window_s": len(win)},
        "power_w": {"mean": sum(x["power"] for x in win) / len(win), "max": max(x["power"] for x in win)},
        "temp_c": {"first": win[0]["temp"], "last": win[-1]["temp"], "max": max(x["temp"] for x in win)},
        "share_sw_power_cap_window": share("cap", win),
        "share_sw_power_cap_run": share("cap", samples),
        "share_thermal_slowdown_run": share("thermal", samples),
        "share_hw_slowdown_run": share("hw", samples),
        "sm_max_mhz": win[-1]["sm_max"],
        "spread_limits": {"sm_mhz": a.spread_mhz, "temp_c": a.spread_c},
    }
    # locked? an idle GPU sits at a low clock (P8) unless its clock is locked (or the applications-
    # clocks event reason is active); under load a lock holds the clock to one value.
    locked = bool(idle["appclk_reason"]) or (idle["sm"] >= 0.5 * idle["sm_max"] and abs(steady_med - idle["sm"]) <= 15)
    res["locked"] = locked
    low = steady_min - 30
    rec = int(math.floor(low / 15.0) * 15)
    clean = (steady_max - steady_min <= 15 and res["share_sw_power_cap_window"] == 0
             and res["share_thermal_slowdown_run"] == 0 and res["share_hw_slowdown_run"] == 0)
    if locked:
        if clean:
            verdict = f"lock holds: SM {steady_med:.0f} MHz in every sample, no power-cap or thermal slowdown"
            res["recommended_lock_mhz"] = int(round(steady_med))
        else:
            verdict = (f"lock does NOT hold cleanly (SM {steady_min:.0f}-{steady_max:.0f} MHz, power-cap share "
                       f"{res['share_sw_power_cap_window']:.0%}): step down 15-30 MHz")
            res["recommended_lock_mhz"] = int(math.floor((min(steady_min, idle['sm']) - 30) / 15.0) * 15)
    else:
        verdict = "clocks are NOT locked"
        res["recommended_lock_mhz"] = rec
    res["verdict"] = verdict
    if res["recommended_lock_mhz"] and (not locked or not clean):
        f = res["recommended_lock_mhz"]
        res["lock_command"] = f"sudo nvidia-smi -i {gid} -lgc {f},{f}"
        res["unlock_command"] = f"sudo nvidia-smi -i {gid} -rgc"
    with open(os.path.join(a.out, "calibration.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("calibration:")
    print(f"  stable: {'yes' if res['stable'] else 'NO (timeout)'}"
          + (f", after {stable_at:.0f} s under load" if stable_at is not None else f", {res['loaded_seconds']:.0f} s under load"))
    print(f"  steady SM clock (last {len(win)} s): median {steady_med:.0f}, min {steady_min:.0f}, max {steady_max:.0f} MHz"
          f" (max boost {res['sm_max_mhz']:.0f})")
    print(f"  power {res['power_w']['mean']:.0f} W mean / {res['power_w']['max']:.0f} W max; temperature "
          f"{res['temp_c']['first']:.0f} -> {res['temp_c']['last']:.0f} C")
    print(f"  clock events over the run: power cap {res['share_sw_power_cap_run']:.0%}, thermal slowdown "
          f"{res['share_thermal_slowdown_run']:.0%}, HW slowdown {res['share_hw_slowdown_run']:.0%}")
    print(f"  clocks locked: {'yes' if locked else 'no'}; {verdict}")
    if "lock_command" in res:
        print(f"  RECOMMENDED LOCK: {res['recommended_lock_mhz']} MHz   {res['lock_command']}   (undo: {res['unlock_command']})")
        print("    (must be a supported graphics clock: if the command refuses it, take the nearest lower value of")
        print("     nvidia-smi -q -d SUPPORTED_CLOCKS)")
    if not res["stable"]:
        print("  WARNING: no stable state within the timeout; numbers taken now carry run-to-run clock drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
