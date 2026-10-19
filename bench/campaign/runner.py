#!/usr/bin/env python3
"""The campaign runner: executes one manifest on one GPU, on the 4090 and on the cluster alike.

    runner.py run MANIFEST --out DIR --src SRC --build BUILD [--class C] [--copy-to DEST]
                  [--wall-min M | --deadline-epoch T] [--reserve-min 6] [--sync-min 10]
                  [--smoke] [--only ID,ID] [--retry-failed] [--no-pin-check] [--allow-lint-errors]
    runner.py sync DIR DEST        copy a result folder (DONE.json last in each experiment, COMPLETE.json last of all)

What it guarantees (the layout is in results.py):
  * resume: run the same command again after a kill; experiments with a valid DONE.json are not redone, the one
    that was running is wiped and redone, nothing else is touched. A folder that belongs to another manifest,
    decisions file, class or source commit is refused.
  * time: an experiment starts only if it can finish before the deadline minus the copy-back reserve; its own
    time limit is cut to what is left. P1 / P2 need the manifest's margin on top of their estimate.
  * copy-back: after every experiment, every --sync-min minutes, on SIGUSR1 (Slurm --signal=B:USR1@300) and at the end.
  * SIGTERM: the running experiment's process group is killed, the folder is synced, exit 143.
  * every experiment records the GPU clocks / power before and after (the cluster cannot lock clocks).
Experiment kinds: probe (built in: env, clocks, tools, limits), cmd (argv with {src} {build} {out} {class} {exp}),
sweep (bench/run_sweep.sh on a sweep file of the pinned source).
"""
import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest as mf          # noqa: E402
import results as rs           # noqa: E402

LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_launch.py")
SMI_QUERY = ("name,uuid,driver_version,vbios_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,"
             "temperature.gpu,utilization.gpu,memory.used,memory.total,clocks_event_reasons.active")
TOOLS = [("nvcc", "--version"), ("ptxas", "--version"), ("cuobjdump", "--version"), ("nvdisasm", "--version"),
         ("nsys", "--version"), ("ncu", "--version"), ("compute-sanitizer", "--version"), ("gcc", "--version"),
         ("g++", "--version"), ("cmake", "--version"), ("git", "--version"), ("rsync", "--version")]
PIP = ("numpy", "pytest")
CUDART_ATTRS = {"multiprocessor_count": 16, "l2_cache_bytes": 38, "shared_per_block_optin": 97,
                "shared_per_sm": 81, "shared_per_block": 8, "regs_per_block": 12, "cc_major": 75, "cc_minor": 76,
                "max_blocks_per_sm": 106, "mem_bus_width": 37, "sm_clock_khz": 13, "mem_clock_khz": 36}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def run_text(argv, timeout=20):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: {e}"


def smi(query=SMI_QUERY):
    """One nvidia-smi sample as a dict (the first GPU of CUDA_VISIBLE_DEVICES when it is a plain index)."""
    if not shutil.which("nvidia-smi"):
        return {"error": "no nvidia-smi"}
    args = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    if cv.isdigit():
        args.insert(1, f"--id={cv}")
    out = run_text(args)
    if out.startswith("error") or not out.strip():
        return {"error": out}
    row = [x.strip() for x in out.splitlines()[0].split(",")]
    keys = query.split(",")
    return dict(zip(keys, row)) if len(row) == len(keys) else {"error": out}


def gpu_name():
    return os.environ.get("CAMPAIGN_GPU_NAME") or smi("name").get("name", "")


# ---------------------------------------------------------------- probes
def probe_env(out):
    info = {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("SLURM_JOB", "SLURM_NODE", "CUDA_VISIBLE", "CONDA_DEFAULT", "IMMERDINK"))},
            "gpu": smi(), "nvidia_smi_L": run_text(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else None}
    rs.write_json_atomic(os.path.join(out, "env.json"), info)


def probe_clocks(out):
    info = {"sample": smi()}
    if shutil.which("nvidia-smi"):
        info["q_clock_performance"] = run_text(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE"])
        info["clock_lock_possible"] = "no (a cluster job has no root): clocks are recorded, never set"
    rs.write_json_atomic(os.path.join(out, "clocks.json"), info)


def probe_tools(out):
    info = {}
    for tool, flag in TOOLS:
        path = shutil.which(tool)
        text = run_text([path, flag]).splitlines() if path else []
        info[tool] = {"path": path, "version": (text[-1] if tool in ("nvcc", "ptxas", "cuobjdump", "nvdisasm") else (text or [""])[0]),
                      "output": text[:6]} if path else None
    py = {}
    for pkg in PIP:
        try:
            m = __import__(pkg)
            py[pkg] = getattr(m, "__version__", "?")
        except ImportError:
            py[pkg] = None
    info["python_packages"] = py
    rs.write_json_atomic(os.path.join(out, "tools.json"), info)


def probe_limits(out):
    info = {}
    lib = None
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            info["libcudart"] = name
            break
        except OSError:
            continue
    if lib is None:
        info["error"] = "no libcudart on the library path"
    else:
        dev = 0
        for key, attr in CUDART_ATTRS.items():
            v = ctypes.c_int(-1)
            rc = lib.cudaDeviceGetAttribute(ctypes.byref(v), attr, dev)
            info[key] = v.value if rc == 0 else f"cudaError {rc}"
    rs.write_json_atomic(os.path.join(out, "limits.json"), info)


PROBES = {"env": probe_env, "clocks": probe_clocks, "tools": probe_tools, "limits": probe_limits}


# ---------------------------------------------------------------- copy-back
def sync_tree(src, dest, lock=None):
    """Copy a result folder to `dest`. Each file goes through a temp name and a rename; within an experiment folder
    DONE.json goes last, and COMPLETE.json goes last of all, so a copy cut short never looks finished."""
    if lock:
        lock.acquire()
    try:
        os.makedirs(dest, exist_ok=True)
        late = []                                   # DONE.json files, then COMPLETE.json
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            os.makedirs(os.path.join(dest, rel), exist_ok=True)
            dirs.sort()
            for fn in sorted(files):
                if fn.startswith("."):
                    continue
                sp = os.path.join(root, fn)
                if fn == rs.DONE or (fn == rs.COMPLETE and rel == "."):
                    late.append((fn == rs.COMPLETE, sp, os.path.join(dest, rel, fn)))
                    continue
                _copy_if_changed(sp, os.path.join(dest, rel, fn))
        for _, sp, dp in sorted(late):              # DONE.json (False) before COMPLETE.json (True)
            _copy_if_changed(sp, dp)
    finally:
        if lock:
            lock.release()


def _copy_if_changed(sp, dp):
    try:
        st = os.stat(sp)
        if os.path.exists(dp):
            dt = os.stat(dp)
            if dt.st_size == st.st_size and dt.st_mtime >= st.st_mtime:
                return
        tmp = os.path.join(os.path.dirname(dp), f".{os.path.basename(dp)}.tmp")
        shutil.copyfile(sp, tmp)
        shutil.copystat(sp, tmp)
        os.replace(tmp, dp)
    except FileNotFoundError:
        pass                                        # vanished between listing and copying: next sync


# ---------------------------------------------------------------- the run
class Terminated(Exception):
    pass


class Run:
    def __init__(self, a):
        a.src, a.build, a.manifest = os.path.abspath(a.src), os.path.abspath(a.build), os.path.abspath(a.manifest)
        a.copy_to = os.path.abspath(a.copy_to) if a.copy_to else None
        self.a = a      # (experiments run in their own folders: every path they receive must be absolute)
        self.out = os.path.abspath(a.out)
        self.m = mf.load(a.manifest)
        self.child = None
        self.sync_lock = threading.Lock()
        self.want_sync = threading.Event()
        self.terminated = False
        self.cls = None
        self.t0 = time.time()
        if a.deadline_epoch is not None:
            self.deadline = a.deadline_epoch
        elif os.environ.get("SLURM_JOB_END_TIME") and a.wall_min is None:
            self.deadline = float(os.environ["SLURM_JOB_END_TIME"])
        else:
            self.deadline = self.t0 + 60.0 * (a.wall_min if a.wall_min is not None else self.m.wall_limit_min or 240)

    # -- helpers
    def remaining_min(self):
        return (self.deadline - time.time()) / 60.0 - self.a.reserve_min

    def sync(self):
        if self.a.copy_to:
            try:
                sync_tree(self.out, self.a.copy_to, self.sync_lock)
            except OSError as e:
                log(f"copy-back failed: {e}")

    def expdir(self, eid):
        return os.path.join(self.out, "exp", eid)

    def effective(self, e):
        if self.a.smoke and isinstance(e.get("smoke"), dict):
            e = {**e, **e["smoke"]}
        return e

    def sub(self, s, e):
        return (s.replace("{src}", self.a.src).replace("{build}", self.a.build).replace("{out}", self.expdir(e["id"]))
                 .replace("{class}", self.cls).replace("{exp}", e["id"]))

    # -- setup
    def setup(self):
        a, m = self.a, self.m
        findings = mf.lint(m, a.src)
        for f in findings:
            log(f)
        if any(f.level == "error" for f in findings) and not a.allow_lint_errors:
            log("manifest lint (G1) failed; refusing to run")
            return 2
        self.cls = a.cls or m.classify(gpu_name())
        if self.cls is None or self.cls not in m.classes:
            log(f"cannot tell the GPU class (device {gpu_name()!r}); give --class")
            return 2
        src_commit = run_text(["git", "-C", a.src, "rev-parse", "HEAD"])
        dirty = run_text(["git", "-C", a.src, "status", "--porcelain", "--untracked-files=no"])
        pinned = m.data["campaign"]["immerdink_commit"]
        pin_ok = src_commit == pinned and not dirty
        if not pin_ok and not a.no_pin_check:
            log(f"source checkout {src_commit[:12]}{' (dirty)' if dirty else ''} is not the pinned commit {pinned[:12]}; refusing")
            return 3
        lock = {"campaign": m.id, "manifest_sha256": m.sha256,
                "decisions_sha256": m.data["campaign"]["decisions_sha256"], "class": self.cls,
                "gpu_name": gpu_name(), "gpu_name_overridden": bool(os.environ.get("CAMPAIGN_GPU_NAME")),
                "pinned_commit": pinned, "src_commit": src_commit, "src_dirty": bool(dirty), "pin_check": "ok" if pin_ok else "skipped",
                "cutlass_pinned": m.data["campaign"]["cutlass_commit"]}
        lp = os.path.join(self.out, rs.LOCK)
        os.makedirs(os.path.join(self.out, "exp"), exist_ok=True)
        if os.path.exists(lp):
            old = rs.read_json(lp)
            keys = ("campaign", "manifest_sha256", "decisions_sha256", "class", "src_commit", "src_dirty")
            diff = [k for k in keys if old.get(k) != lock[k]]
            if diff:
                log(f"{self.out} belongs to another campaign state ({', '.join(diff)} differ); refusing to resume into it")
                return 4
            lock["created"] = old.get("created")
            rs.append_journal(self.out, "resume", pid=os.getpid())
        else:
            lock["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            rs.append_journal(self.out, "start", pid=os.getpid(), cls=self.cls)
        rs.write_json_atomic(lp, lock)
        shutil.copyfile(m.path, os.path.join(self.out, "MANIFEST.toml"))
        shutil.copyfile(m.decisions_path(), os.path.join(self.out, "DECISIONS.toml"))
        ev = os.path.join(self.out, "env.json")
        if not os.path.exists(ev):
            probe_env(self.out)
        else:
            n = len([f for f in os.listdir(self.out) if f.startswith("env-resume")])
            tmp_out = os.path.join(self.out, f"env-resume-{n + 1}.d")
            os.makedirs(tmp_out, exist_ok=True)
            probe_env(tmp_out)
            os.replace(os.path.join(tmp_out, "env.json"), os.path.join(self.out, f"env-resume-{n + 1}.json"))
            os.rmdir(tmp_out)
        return 0

    # -- one experiment
    def argv_of(self, e):
        kind = e["kind"]
        if kind == "cmd":
            return [self.sub(x, e) for x in e["argv"]]
        sweepout = os.path.join(self.expdir(e["id"]), e.get("outputs", ["sweep.jsonl"])[0])
        return ["bash", os.path.join(self.a.src, "bench", "run_sweep.sh"), os.path.join(self.a.src, e["file"]), sweepout]

    def run_experiment(self, e):
        eid = e["id"]
        d = self.expdir(eid)
        if os.path.exists(d):
            shutil.rmtree(d)                       # incomplete (no valid DONE.json): wipe and redo
        os.makedirs(d)
        rs.append_journal(self.out, "exp-start", id=eid)
        t_start = time.time()
        clocks0 = smi("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active")
        status, reason, rc, argv = "ok", "", None, None
        if e["kind"] == "probe":
            try:
                PROBES[e["probe"]](d)
            except Exception as ex:                # a probe must never end the campaign
                status, reason = "failed", f"probe raised {type(ex).__name__}: {ex}"
        else:
            argv = self.argv_of(e)
            env = dict(os.environ)
            env.update({k: self.sub(str(v), e) for k, v in e.get("env", {}).items()})
            env["IMMERDINK_BENCH_BUILD"] = self.a.build          # bench/build.sh, run_sweep.sh, profile_nsys.sh build / find imd_bench here
            if e["kind"] == "sweep":
                if e.get("cases"):
                    env["CASES"] = e["cases"]
            limit_min = min(e.get("timeout_min", 2.0 * e["est_min"]), max(self.remaining_min(), 0.05))
            with open(os.path.join(d, "stdout.log"), "wb") as so, open(os.path.join(d, "stderr.log"), "wb") as se:
                try:
                    self.child = subprocess.Popen([sys.executable, "-I", LAUNCH] + argv, cwd=d, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                                  start_new_session=True)
                except OSError as ex:
                    status, reason = "failed", f"cannot start: {ex}"
                    self.child = None
                if self.child is not None:
                    try:
                        rc = self.child.wait(timeout=limit_min * 60.0)
                    except subprocess.TimeoutExpired:
                        self.kill_child()
                        rc = self.child.wait()
                        status, reason = "failed", f"timeout after {limit_min:.1f} min"
                    finally:
                        was_terminated = self.terminated
                        self.child = None
                    if was_terminated:
                        raise Terminated()
                    ok_codes = e.get("ok_exit_codes", [0, 4] if e["kind"] == "sweep" else [0])
                    if status == "ok" and rc not in ok_codes:
                        status, reason = "failed", f"exit code {rc}"
        if status == "ok":
            for o in e.get("outputs", []):
                p = os.path.join(d, o)
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    status, reason = "failed", f"expected output {o} is missing or empty"
                    break
        clocks1 = smi("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active")
        done = {"id": eid, "status": status, "reason": reason, "rc": rc, "argv": argv, "kind": e["kind"],
                "priority": e["priority"], "started": t_start, "ended": time.time(),
                "seconds": time.time() - t_start, "est_min": e["est_min"], "clocks_before": clocks0,
                "clocks_after": clocks1, "files": rs.file_table(d)}
        rs.write_json_atomic(os.path.join(d, rs.DONE), done)
        rs.append_journal(self.out, "exp-done", id=eid, status=status, reason=reason, seconds=done["seconds"])
        log(f"{eid}: {status}{(' (' + reason + ')') if reason else ''} in {done['seconds'] / 60.0:.1f} min")
        self.sync()
        return status

    def kill_child(self):
        c = self.child
        if c is None or c.poll() is not None:
            return
        try:
            os.killpg(c.pid, signal.SIGTERM)
            try:
                c.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # -- the loop
    def run(self):
        a = self.a
        rc = self.setup()
        if rc:
            return rc
        plan = self.m.plan(self.cls)
        if a.only:
            plan = [e for e in plan if e["id"] in a.only]
        plan = [self.effective(e) for e in plan if not a.smoke or isinstance(e.get("smoke"), dict) or e.get("smoke") is True]
        log(f"class {self.cls}: {len(plan)} experiment(s), deadline in {(self.deadline - time.time()) / 60:.0f} min")
        signal.signal(signal.SIGUSR1, lambda *_: self.want_sync.set())
        signal.signal(signal.SIGTERM, self.on_term)
        stop = threading.Event()
        th = threading.Thread(target=self.sync_loop, args=(stop,), daemon=True)
        th.start()
        states, skipped = {}, {}
        try:
            for e in plan:
                if self.terminated:
                    raise Terminated()
                eid = e["id"]
                state, _ = rs.verify_exp(self.expdir(eid)) if os.path.exists(self.expdir(eid)) else ("incomplete", [])
                if state == "ok" or (state == "failed" and not a.retry_failed):
                    states[eid] = state
                    rs.append_journal(self.out, "exp-skip-done", id=eid, state=state)
                    continue
                miss = [n for n in e.get("needs", []) if states.get(n) != "ok"]
                if miss:
                    skipped[eid] = f"needs {', '.join(miss)}, which did not finish ok"
                    rs.append_journal(self.out, "exp-skipped", id=eid, reason=skipped[eid])
                    log(f"{eid}: skipped, {skipped[eid]}")
                    continue
                need = e["est_min"] * (1.0 if e["priority"] == "P0" else 1.0 + self.m.margin)
                if self.remaining_min() < need:
                    skipped[eid] = f"not enough time: {self.remaining_min():.1f} min left before the copy-back reserve, needs {need:.1f}"
                    rs.append_journal(self.out, "exp-skipped", id=eid, reason=skipped[eid])
                    log(f"{eid}: skipped, {skipped[eid]}")
                    continue
                states[eid] = self.run_experiment(e)
        except Terminated:
            rs.append_journal(self.out, "terminated")
            log("terminated; syncing")
            self.sync()
            stop.set()
            return 143
        stop.set()
        counts = {}
        for s in states.values():
            counts[s] = counts.get(s, 0) + 1
        rs.write_json_atomic(os.path.join(self.out, rs.COMPLETE),
                             {"campaign": self.m.id, "class": self.cls, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              "states": states, "skipped": skipped, "counts": counts,
                              "planned": [e["id"] for e in plan]})
        rs.append_journal(self.out, "complete", counts=counts)
        self.sync()
        log(f"complete: {counts}, skipped {len(skipped)}")
        return 0

    def sync_loop(self, stop):
        last = time.time()
        while not stop.is_set():
            if self.want_sync.wait(timeout=5.0) or time.time() - last >= 60.0 * self.a.sync_min:
                self.want_sync.clear()
                self.sync()
                last = time.time()

    def on_term(self, *_):
        self.terminated = True
        self.kill_child()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("--out", required=True)
    r.add_argument("--src", required=True)
    r.add_argument("--build", required=True)
    r.add_argument("--class", dest="cls")
    r.add_argument("--copy-to")
    r.add_argument("--wall-min", type=float)
    r.add_argument("--deadline-epoch", type=float)
    r.add_argument("--reserve-min", type=float, default=6.0)
    r.add_argument("--sync-min", type=float, default=10.0)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--only", type=lambda s: s.split(","))
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--no-pin-check", action="store_true")
    r.add_argument("--allow-lint-errors", action="store_true")
    s = sub.add_parser("sync")
    s.add_argument("src")
    s.add_argument("dest")
    a = p.parse_args(argv)
    if a.cmd == "sync":
        sync_tree(a.src, a.dest)
        return 0
    return Run(a).run()


if __name__ == "__main__":
    sys.exit(main())
