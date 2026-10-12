#!/usr/bin/env python3
"""The offline gates of the campaign kit: everything that must pass before a run is submitted (docs/CAMPAIGN.md).

    gates.py g1 MANIFEST... [--src DIR]                  manifest lint: schema, pins, decisions hash, budget, no placeholders
    gates.py g2 --build-dir DIR                          architecture gate: sm_89, sm_90a, sm_120a compile checks and facts
    gates.py g3 MANIFEST --src DIR --build DIR [...]     runner rehearsal on this machine's GPU: kill -9, resume, copy-back
    gates.py g4 [--python PY]                            analyzer + runner + manifest + statistics tests, golden test on the issue-11 folders
    gates.py g5 FOLDER                                   bit identity of every variant record in a result folder
    gates.py g6 MANIFEST --logs DIR... [--src DIR]       time estimates against the issue-11 logs
    gates.py g7 MANIFEST                                 the login-node dry run (run it ON the PACE login node)

G3 uses the real machine: it takes the GPU like any benchmark (the experiments go through tools/run_bench.sh: quiet gate and
shared GPU lock), so check that the machine is quiet first. It kills the runner with SIGKILL while a chosen experiment runs,
resumes, and checks the result folder, the copy and the expected verdicts.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an      # noqa: E402
import manifest as mf     # noqa: E402
import results as rs      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TESTS = ("stats", "manifest", "runner", "analyze", "archgate")


def say(gate, ok, msg):
    print(f"{gate}: {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def g1(a):
    ok = True
    for path in a.manifests:
        m = mf.load(path)
        F = mf.lint(m, a.src)
        for f in F:
            print("  ", f)
        n = sum(f.level == "error" for f in F)
        ok &= say("G1", n == 0, f"{os.path.basename(path)}: {n} error(s), {len(F) - n} warning(s)")
    return ok


def g2(a):
    env = dict(os.environ)
    r1 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "archgate.py"), "facts"], env=env)
    r2 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "archgate.py"), "check", "--build-dir", a.build_dir], env=env)
    return say("G2", r1.returncode == 0 and r2.returncode == 0, "toolchain facts and per-architecture compile table")


def find_python(arg):
    for c in (arg, os.environ.get("PY312"), shutil.which("python3.12"), os.path.expanduser("~/miniforge3/envs/campaign-py312/bin/python")):
        if c and os.path.exists(c):
            return c
    return None


def g4(a):
    ok = True
    pys = [sys.executable]
    p312 = find_python(a.python)
    if p312:
        pys.append(p312)
    else:
        print("  note: no Python 3.12 found (--python / PY312): the cluster env runs 3.12")
    for py in pys:
        for t in TESTS:
            r = subprocess.run([py, "-I", os.path.join(HERE, f"test_{t}.py")], capture_output=True, text=True,
                               env={**os.environ, "PATH": os.environ["PATH"]})
            last = (r.stderr.strip().splitlines() or ["?"])[-1]
            ok &= say("G4", r.returncode == 0, f"test_{t} under {py}: {last}")
    golden = os.environ.get("CAMPAIGN_GOLDEN_DIR", os.path.expanduser("~/repos/generalWork/immerdink/scratch/runs/20261011-180837"))
    if os.path.isdir(golden):
        jobs = sorted(os.path.join(golden, d) for d in os.listdir(golden) if "-immerdink-" in d)
        for j in jobs:
            checked, diffs = an.golden(j)
            ok &= say("G4", bool(checked) and not diffs, f"issue-11 tables regenerated from {os.path.basename(j)}: {len(checked)} file(s), {len(diffs)} difference line(s)")
    else:
        print("  note: the issue-11 folders are not here; the golden comparison is skipped")
    return ok


def g5(a):
    n, bad = an.bitcheck(a.folder)
    for x in bad:
        print("   NOT IDENTICAL", x)
    return say("G5", not bad and n > 0, f"{n} record(s) checked")


def g6(a):
    import estimates
    rows, bad = estimates.check(a.manifest, a.logs, a.src)
    for eid, est, implied, note in rows:
        print(f"   {eid}: est {est:g} min, logs imply {'-' if implied is None else f'{implied:.1f} min'}: {note}")
    return say("G6", bad == 0, f"{bad} estimate(s) too low")


def g7(a):
    here = os.path.abspath(a.manifest)
    sub = os.path.join(os.path.dirname(os.path.dirname(here)), "submit_campaign.sh")
    if not os.path.exists(sub):
        return say("G7", False, f"no submit_campaign.sh next to the campaigns folder ({sub})")
    r = subprocess.run(["bash", sub, here], env={**os.environ, "DRY_RUN": "1"})
    return say("G7", r.returncode == 0, "DRY_RUN=1 submit_campaign.sh (run on the login node: environment, pins, CUTLASS, GEMMul8)")


def wait_for(pred, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def g3(a):
    """Rehearsal on this machine. Returns True when the kill -9 / resume / copy-back / verdict expectations all hold."""
    work = os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)
    out, copy = os.path.join(work, "out"), os.path.join(work, "copy")
    os.makedirs(work)
    py = find_python(a.python) or sys.executable
    runner = [py, "-I", os.path.join(HERE, "runner.py"), "run", a.manifest, "--out", out, "--src", a.src, "--build", a.build,
              "--copy-to", copy, "--sync-min", "0.05"]
    ok = True
    m = mf.load(a.manifest)
    victim = a.victim
    if victim is None:
        victim = next(e["id"] for e in m.experiments if e["kind"] in ("sweep", "cmd"))
    print(f"  runner under {py}; victim of the kill: {victim}")
    p = subprocess.Popen(runner, stdout=subprocess.DEVNULL, stderr=open(os.path.join(work, "run1.log"), "w"))
    jp = os.path.join(out, "journal.jsonl")

    def started():
        return os.path.exists(jp) and any(json.loads(l).get("id") == victim and json.loads(l)["event"] == "exp-start"
                                          for l in open(jp).read().splitlines() if l.strip())

    got = wait_for(started, 120)
    time.sleep(a.kill_after)
    p.send_signal(signal.SIGKILL)
    p.wait()
    ok &= say("G3", got and not os.path.exists(os.path.join(out, rs.COMPLETE)), f"runner killed with SIGKILL while {victim} ran (exit {p.returncode})")
    time.sleep(1.0)
    st = rs.verify_exp(os.path.join(out, "exp", victim))[0]
    ok &= say("G3", st == "incomplete", f"the victim has no valid DONE.json after the kill ({st})")
    states_before = {e["id"]: rs.verify_exp(os.path.join(out, "exp", e["id"]))[0] for e in m.experiments if os.path.isdir(os.path.join(out, "exp", e["id"]))}
    r = subprocess.run(runner, capture_output=True, text=True)
    open(os.path.join(work, "run2.log"), "w").write(r.stderr)
    ok &= say("G3", r.returncode == 0, f"resume ran to the end (exit {r.returncode})")
    events = [json.loads(l) for l in open(jp).read().splitlines() if l.strip()]
    starts = {}
    for e in events:
        if e["event"] == "exp-start":
            starts[e["id"]] = starts.get(e["id"], 0) + 1
    redone = [k for k, v in starts.items() if v > 1]
    ok &= say("G3", redone == [victim], f"only the victim ran twice (ran more than once: {redone})")
    ok &= say("G3", all(states_before.get(k) in ("ok", "failed") for k in states_before if k != victim),
              "every experiment finished before the kill kept its DONE.json")
    comp = os.path.exists(os.path.join(out, rs.COMPLETE))
    ok &= say("G3", comp, "COMPLETE.json written")
    cls = rs.read_json(os.path.join(out, rs.LOCK))["class"]
    intact = all(rs.verify_exp(os.path.join(out, "exp", e["id"]))[0] == "ok" for e in m.plan(cls))
    ok &= say("G3", intact, "all experiments verify (size and sha256 of every file)")
    same = subprocess.run(["diff", "-r", "-q", "--exclude=journal.jsonl", out, copy], capture_output=True, text=True)
    ok &= say("G3", same.returncode == 0, "the copy-back folder equals the run folder" + ("" if same.returncode == 0 else f": {same.stdout.strip()[:200]}"))
    dfile = m.decisions_path()
    vs, info = an.verdicts(out, decisions_file=dfile)
    print(an.render(vs, info))
    if a.expect:
        want = dict(x.split("=") for x in a.expect.split(","))
        got_v = {v["id"]: v["verdict"] for v in vs}
        for k, v in want.items():
            ok &= say("G3", got_v.get(k) == v, f"verdict {k} = {got_v.get(k)} (expected {v})")
    n, bad = an.bitcheck(out)
    ok &= say("G5", not bad and n > 0, f"bit identity: {n} record(s), {len(bad)} problem(s)")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="gate", required=True)
    x = s.add_parser("g1"); x.add_argument("manifests", nargs="+"); x.add_argument("--src")
    x = s.add_parser("g2"); x.add_argument("--build-dir", required=True)
    x = s.add_parser("g3"); x.add_argument("manifest"); x.add_argument("--src", required=True); x.add_argument("--build", required=True)
    x.add_argument("--work", required=True); x.add_argument("--victim"); x.add_argument("--kill-after", type=float, default=2.0)
    x.add_argument("--python"); x.add_argument("--expect", help="decision=VERDICT,decision=VERDICT")
    x = s.add_parser("g4"); x.add_argument("--python")
    x = s.add_parser("g5"); x.add_argument("folder")
    x = s.add_parser("g6"); x.add_argument("manifest"); x.add_argument("--logs", nargs="+", required=True); x.add_argument("--src")
    x = s.add_parser("g7"); x.add_argument("manifest")
    a = p.parse_args(argv)
    return 0 if globals()[a.gate](a) else 1


if __name__ == "__main__":
    sys.exit(main())
