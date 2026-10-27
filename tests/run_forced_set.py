#!/usr/bin/env python3
"""R4 integration: run the oracle suites (e2e, eps) and the committed hashes (small, mid, large: 19 cases) with ONE forced instruction-sequence set and K5 CRT.

    python3 tests/run_forced_set.py --lib build/libimmerdink.so --shim build/tests/libimd_shim.so --seq 245 --crt f64

The same two environment variables the single-set ctests use (IMD_TEST_SEQ, IMD_TEST_K5_CRT; tests/harness.py reads the set and the CRT back and fails the run if they did
not take). One ctest per R4 duel side (tests/CMakeLists.txt "r4_set_<seq>_<crt>"), so a failure names the set. The sets are bit-identical to the baseline by construction;
the oracle and the committed hashes are the gate.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--crt", choices=("int", "f64", "default"), default="default")
    args = ap.parse_args()
    env = dict(os.environ, IMD_TEST_SEQ=str(args.seq))
    if args.crt != "default":
        env["IMD_TEST_K5_CRT"] = "1" if args.crt == "f64" else "0"
    pre = dict(env, LD_PRELOAD=args.shim)
    steps = [(pre, "test_contract.py", ["--suite", "e2e"]), (pre, "test_contract.py", ["--suite", "eps"]),
             (env, "test_output_hashes.py", ["--tags", "small", "mid"]), (env, "test_output_hashes.py", ["--tags", "large"])]
    rc = 0
    for e, script, extra in steps:
        cmd = [sys.executable, os.path.join(HERE, script), "--lib", args.lib] + extra
        if script == "test_contract.py":
            cmd += ["--shim", args.shim, "-v"]
        print("RUN", " ".join(cmd[1:]), f"(IMD_TEST_SEQ={args.seq} crt={args.crt})", flush=True)
        r = subprocess.run(cmd, env=e)
        if r.returncode:
            print(f"FAIL: {script} {extra} exited {r.returncode}", flush=True)
            rc = 1
    print(("PASS" if rc == 0 else "FAIL") + f"  forced set {args.seq} crt {args.crt}: oracle e2e / eps and the 19 committed hashes")
    return rc


if __name__ == "__main__":
    sys.exit(main())
