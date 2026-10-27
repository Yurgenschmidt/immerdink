#!/usr/bin/env python3
"""Smoke test of imd_bench's --seq / --crt (method imd; issue 21; docs/NOTES-21.md "Pack emitter and finish").

    python3 bench/test_seq_crt.py --bench build/bench/imd_bench [--no-seq-twins] [--keep]

Checks, on a tiny shape (the GPU is used under the caller's GPU lock; tools/test.sh holds it):
  * the usage text names --seq and --crt;
  * invalid values are refused before any GPU work: --seq 2, 6, 8, 256 / 257 / 272 (klpersist without klload), 512, x and --crt foo exit non-zero;
  * for every sequence set in {default, 0, 1, 7} x CRT variant {default, int, f64}: one imd record whose c_hash is the same for
    all of them (every configuration is bit-identical), whose requested values (seq_req, crt_req) are what was passed
    and whose effective values (seq, crt, phase_seq, phase_crt) are what the library ran (a forced set is read back as
    itself; "default" is read back as the device class's set, which carries bit 8 = the limb CRT on sm_90; the CRT is
    int / f64 as forced, and for crt=default it is the limb CRT exactly when the set the plan ran has bit 8: the class default
    on sm_90, never a forced set 0..7);
  * the settings end with the method: the record's seq_after_reset / crt_after_reset (the plan read back after the reset) are
    the device class's set and its CRT (f64 on sm_90, int elsewhere).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

CASE = ["--name", "smoke", "--spec", "ik,kj->ij", "--ext", "i=24,j=16,k=40"]
FAST = ["--N", "8", "--reps", "2", "--warmup", "1", "--phase-reps", "2", "--max-seconds", "5"]


def run(bench, extra, out):
    cmd = [bench] + CASE + FAST + extra + ["--out", out]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def records(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, help="the profiling-build imd_bench")
    ap.add_argument("--no-seq-twins", action="store_true", help="built with IMMERDINK_SEQ_TWINS=OFF: only sets 0 and 1")
    ap.add_argument("--keep", action="store_true", help="keep the JSONL files (printed)")
    args = ap.parse_args()
    bad = []
    tmp = tempfile.mkdtemp(prefix="imd_seq_crt_")

    # The usage call needs no GPU and returns in well under a second on the dev box; on the Hopper nodes of R3 it hit the old 60 s timeout
    # (docs/NOTES-22.md, immerdink-eval findings r3-20261026). stdin is closed so nothing can wait on it, and a timeout is reported with the
    # elapsed time and the process's dynamic-library view instead of a bare traceback, so the next cluster run explains itself.
    t0 = time.time()
    try:
        u = subprocess.run([args.bench], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        ldd = subprocess.run(["ldd", args.bench], capture_output=True, text=True, timeout=60).stdout[-1500:]
        print(f"FAIL  the no-argument usage call did not return within {time.time() - t0:.0f} s\nstderr so far: "
              f"{(e.stderr or b'')[-400:]!r}\nldd:\n{ldd}")
        return 1
    if time.time() - t0 > 10:
        print(f"note  the usage call took {time.time() - t0:.1f} s (normally < 1 s)")
    if u.returncode == 0 or "--seq" not in u.stderr or "--crt" not in u.stderr:
        bad.append("the usage text does not name --seq and --crt (or exits 0)")

    for flag, val in (("--seq", "2"), ("--seq", "6"), ("--seq", "8"), ("--seq", "24"), ("--seq", "128"), ("--seq", "130"), ("--seq", "256"), ("--seq", "257"), ("--seq", "272"), ("--seq", "512"), ("--seq", "x"), ("--seq", "-1"), ("--crt", "foo"), ("--crt", "2"), ("--crt", "-1")):
        r = run(args.bench, [flag, val], os.path.join(tmp, "bad.jsonl"))
        if r.returncode == 0:
            bad.append(f"{flag} {val} was accepted")

    sets = ("default", "0", "1") if args.no_seq_twins else ("default", "0", "1", "7", "17", "113", "133", "245", "421", "437")   # 17 = E1 + klwarp, 113 = E1 + the three KL cuts, 133 = E1 + F1 + k5fin, 245 = E1 + F1 + all KL + k5fin (issue 23), 421 = 165 + klpersist (issue 25), 437 = 181 + klpersist (= the sm_90 default 189 + klpersist with crt=f64, the R6 duel B side)
    hashes, default_set, default_crt = {}, None, None
    for crt in ("default", "int", "f64"):
        for seq in sets:
            out = os.path.join(tmp, f"{seq}_{crt}.jsonl")
            r = run(args.bench, ["--seq", seq, "--crt", crt], out)
            tag = f"seq={seq} crt={crt}"
            if r.returncode != 0:
                bad.append(f"{tag}: exit {r.returncode}: {r.stderr[-400:]}")
                continue
            recs = [x for x in records(out) if x.get("method") == "imd"]
            if len(recs) != 1:
                bad.append(f"{tag}: {len(recs)} imd records")
                continue
            x = recs[0]
            want_req = -1 if seq == "default" else int(seq)
            if x.get("seq_req") != want_req or x.get("crt_req") != crt:
                bad.append(f"{tag}: requested {x.get('seq_req')}/{x.get('crt_req')}")
            want_crt = crt if crt != "default" else ("f64" if (x.get("seq") or 0) & 8 else "int")
            if x.get("crt_req") != crt or x.get("crt") != want_crt or x.get("phase_crt") != want_crt:
                bad.append(f"{tag}: effective crt {x.get('crt')}/{x.get('phase_crt')}, expected {want_crt}")
            if crt == "default" and seq != "default" and x.get("crt") != "int":
                bad.append(f"{tag}: a forced set with the CRT following the set must run the integer CRT, ran {x.get('crt')}")
            if x.get("phase_seq") != x.get("seq"):
                bad.append(f"{tag}: phase_seq {x.get('phase_seq')} != seq {x.get('seq')}")
            if seq == "default":
                if default_set is None:
                    default_set = x.get("seq")
                    default_crt = "f64" if (default_set & 8) else "int"
                elif x.get("seq") != default_set:
                    bad.append(f"{tag}: class default {x.get('seq')} != {default_set}")
            elif x.get("seq") != int(seq):
                bad.append(f"{tag}: forced {seq} but the plan ran {x.get('seq')}")
            if not x.get("phases"):
                bad.append(f"{tag}: no phases")
            hashes[tag] = x.get("c_hash")
    if not hashes or len(set(hashes.values())) != 1 or None in hashes.values():
        bad.append(f"c_hash differs across configurations: {hashes}")

    # The settings end with the method: the record carries what the plan reads back once they are reset (the class's set,
    # crt int), and a forced run's record must agree with the default run's.
    out = os.path.join(tmp, "forced.jsonl")
    r = run(args.bench, ["--seq", "0", "--crt", "f64"], out)
    if r.returncode != 0:
        bad.append(f"forced run: exit {r.returncode}: {r.stderr[-400:]}")
    else:
        x = [y for y in records(out) if y.get("method") == "imd"][0]
        if x.get("seq") != 0 or x.get("seq_after_reset") != default_set or x.get("crt_after_reset") != default_crt:
            bad.append(f"after a forced run the settings read back {x.get('seq_after_reset')}/{x.get('crt_after_reset')}, "
                       f"expected {default_set}/{default_crt}")

    if args.keep:
        print("records in", tmp)
    if bad:
        for b in bad:
            print("FAIL ", b)
        return 1
    print(f"OK  {len(hashes)} configurations, one c_hash {next(iter(hashes.values()))}; class default set {default_set}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
