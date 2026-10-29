#!/usr/bin/env python3
"""Committed output hashes: the library's C must match them bit for bit on every GPU.

    python3 tests/test_output_hashes.py [--lib build/libimmerdink.so] [--tags small mid]
    python3 tests/test_output_hashes.py --oracle        # also recompute the CPU-source hashes
    python3 tests/test_output_hashes.py --regen NAME... # print fresh hashes (see below)

Why: the INT8 sums are exact integers, so the result C of a contraction is a function of the inputs
alone. It must be the same on an RTX 4090, an H100, an H200 and an RTX PRO 6000. A hash mismatch is a
bug (in the library, or in how a new GPU is used), never noise. tests/data/output_hashes.json holds
one hash per case; each case says where it came from:

  "cpu"      the pure-Python oracle (spec/oracle.py, exact rational arithmetic) on the same inputs.
             Independent of any GPU. Only for shapes the oracle can do in seconds.
  "rtx4090"  the library on the RTX 4090. For shapes too big for the oracle. These hashes are only
             as trustworthy as the 4090 run; the small shapes tie that run to the oracle.

Inputs are made on the host with integer arithmetic and exact floating-point steps only
(no exp, log or libm), so every machine builds the same bits:
  h1 = splitmix64(seed ^ 2i), h2 = splitmix64(seed ^ (2i+1)), v = (h1 >> 11) * 2^-53 - 1/2,
  family "u":    A[i] = v
  family "pow2": A[i] = v * 2^((h2 >> 59) - 16)        (a wide dynamic range, still exact)
i is the row-major index of the logical tensor; operand ids give the seed (A: 3, B: 6, C: 9).
The hash is imd_bench's c_hash: the sum over C's elements (row-major in C's own mode order) of
splitmix64(bits(C[i]) ^ (0x9E3779B97F4A7C15 * (i + 1))), modulo 2^64.

Environment: IMD_HASH_JSON=FILE appends one JSON line per case run (name, got, committed, match).

--regen NAME runs the case and prints its JSON entry: with source "cpu" the hash comes from the
oracle (and the GPU result is compared with it), with "rtx4090" from the GPU.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

DATA = os.path.join(HERE, "data", "output_hashes.json")
M64 = (1 << 64) - 1
GOLD = np.uint64(0x9E3779B97F4A7C15)
SEED = {"A": 3, "B": 6, "C": 9}
CHUNK = 1 << 22


class Fail(Exception):
    pass


def splitmix64(x):
    """splitmix64 on a uint64 numpy array (wraps modulo 2^64)."""
    with np.errstate(over="ignore"):
        x = x + GOLD
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def fill(n, seed, family):
    """n doubles of the input family (see the module docstring), in chunks to bound memory."""
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        i = np.arange(lo, hi, dtype=np.uint64)
        h1 = splitmix64(np.uint64(seed) ^ (np.uint64(2) * i))
        v = (h1 >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 - 0.5
        if family == "pow2":
            h2 = splitmix64(np.uint64(seed) ^ (np.uint64(2) * i + np.uint64(1)))
            e = (h2 >> np.uint64(59)).astype(np.int64) - 16
            v = np.ldexp(v, e.astype(np.int32))
        elif family != "u":
            raise Fail(f"unknown family {family}")
        out[lo:hi] = v
    return out


def hash_c(c):
    """imd_bench's c_hash of the double array c, as 16 hex digits."""
    bits = np.ascontiguousarray(c, dtype=np.float64).view(np.uint64)
    total = 0
    for lo in range(0, bits.size, CHUNK):
        hi = min(bits.size, lo + CHUNK)
        i = np.arange(lo, hi, dtype=np.uint64)
        with np.errstate(over="ignore"):
            x = splitmix64(bits[lo:hi] ^ (GOLD * (i + np.uint64(1))))
        total = (total + int(x.sum(dtype=np.uint64))) & M64
    return f"{total:016x}"



def run_gpu(lib, cuda, case):
    """Run the case on the GPU; returns the C buffer (numpy, dense in C's mode order)."""
    import ctypes as C
    T = {}
    for X in "ABC":
        ext = case[X]["extent"]
        T[X] = hz.make_tensor(ext, hz.dense(ext), case[X]["label"])
    N, alpha, beta = case["N"], case.get("alpha", 1.0), case.get("beta", 0.0)
    rounding = hz.TRUNC if case.get("rounding", "RN") == "TRUNC" else hz.RN
    st, ws_bytes = lib.workspace_size(T["A"], T["B"], T["C"], N, 0, rounding)
    if st != "SUCCESS":
        raise Fail(f"{case['name']}: imd_workspace_size {st}")
    st, plan = lib.plan_create(T["A"], T["B"], T["C"], N, 0, rounding)
    if st != "SUCCESS":
        raise Fail(f"{case['name']}: imd_plan_create {st}")
    bufs = {}
    try:
        for X in "ABC":
            n = int(np.prod(case[X]["extent"]))
            host = fill(n, SEED[X], case["family"] if X != "C" else "u")
            bufs[X] = (cuda.malloc(host.nbytes), host)
            cuda.upload(bufs[X][0], host)
        ws = cuda.malloc(max(ws_bytes, 1))
        st = lib.contract(plan, alpha, bufs["A"][0], bufs["B"][0], beta, bufs["C"][0], ws, ws_bytes)
        cuda.sync()
        cuda.free(ws)
        if st != "SUCCESS":
            raise Fail(f"{case['name']}: imd_contract {st}")
        out = np.empty_like(bufs["C"][1])
        cuda.download(out, bufs["C"][0])
        return out
    finally:
        for p, _ in bufs.values():
            cuda.free(p)
        lib.plan_destroy(plan)


def run_oracle(case):
    """The CPU reference: spec/oracle.py's contract_tensor on the same bits; returns C (numpy)."""
    import oracle as o
    D = {X: o.Desc(case[X]["extent"], hz.dense(case[X]["extent"]), case[X]["label"]) for X in "ABC"}
    Tn = {}
    for X in "ABC":
        n = int(np.prod(case[X]["extent"]))
        Tn[X] = o.Tensor(fill(n, SEED[X], case["family"] if X != "C" else "u").tolist(), 0, D[X])
    out = o.contract_tensor(Tn["A"], Tn["B"], Tn["C"], case.get("alpha", 1.0), case.get("beta", 0.0),
                            case["N"], case.get("rounding", "RN"), 0, o.EMULATE)
    return np.array(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--tags", nargs="*", default=["small", "mid"], help="case tags to run")
    ap.add_argument("--oracle", action="store_true", help="also recompute the CPU-source hashes")
    ap.add_argument("--regen", nargs="*", help="print fresh JSON entries for these case names")
    ap.add_argument("--write", action="store_true", help="with --regen: store the hashes in the file")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    with open(DATA) as f:
        doc = json.load(f)
    cases = doc["cases"]
    if args.list:
        for c in cases:
            print(f"{c['name']:24s} {c['source']:8s} {','.join(c['tags'])}")
        return 0
    expected = [c for c in cases if set(c["tags"]) & set(args.tags)]
    if args.regen is None and not expected:      # a selection that matches nothing must not read as a pass
        print(f"FAIL  no case carries the tags {args.tags}")
        return 1
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    t0 = time.time()
    if args.regen is not None:
        for c in cases:
            if c["name"] not in args.regen:
                continue
            got = hash_c(run_gpu(lib, cuda, c))
            if c["source"] == "cpu":
                want = hash_c(run_oracle(c))
                if want != got:
                    print(f"FAIL  {c['name']}: GPU {got} != oracle {want} (a mismatch against the oracle is a library bug)")
                    return 1
            print(json.dumps({"name": c["name"], "c_hash": got}), flush=True)
            if args.write:
                c["c_hash"] = got
                with open(DATA, "w") as f:
                    json.dump(doc, f, indent=1)
                    f.write("\n")
        return 0
    bad, ran = [], 0
    for c in cases:
        if not set(c["tags"]) & set(args.tags):
            continue
        t1 = time.time()
        try:
            got = hash_c(run_gpu(lib, cuda, c))
        except Exception as e:      # a case that did not run is a failure with a record, not a missing line
            bad.append(f"{c['name']}: did not run: {type(e).__name__}: {e}")
            print(f"{c['name']:24s} {c['source']:8s} DID NOT RUN: {type(e).__name__}: {e}", flush=True)
            if os.environ.get("IMD_HASH_JSON"):
                with open(os.environ["IMD_HASH_JSON"], "a") as jf:
                    jf.write(json.dumps({"name": c["name"], "source": c["source"], "c_hash": None,
                                         "committed": c["c_hash"], "match": False}) + "\n")
            continue
        line = f"{c['name']:24s} {c['source']:8s} gpu {got}"
        if got != c["c_hash"]:
            bad.append(f"{c['name']}: GPU {got} != committed {c['c_hash']} ({c['source']})")
            line += f"  MISMATCH (committed {c['c_hash']})"
        if args.oracle and c["source"] == "cpu":
            want = hash_c(run_oracle(c))
            if want != c["c_hash"]:
                bad.append(f"{c['name']}: oracle {want} != committed {c['c_hash']}")
                line += f"  ORACLE-MISMATCH ({want})"
            else:
                line += "  oracle ok"
        ran += 1
        if os.environ.get("IMD_HASH_JSON"):   # one line per case for the campaign analyzer
            with open(os.environ["IMD_HASH_JSON"], "a") as jf:
                jf.write(json.dumps({"name": c["name"], "source": c["source"], "c_hash": got,
                                     "committed": c["c_hash"], "match": got == c["c_hash"]}) + "\n")
        print(f"{line}  {time.time() - t1:5.1f}s", flush=True)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        if any("committed" in b and "GPU" in b for b in bad):
            print("  A source rtx4090 hash is the 4090 library's own output: to tell a library bug from a bad baseline, compare with the 4090 library run and with the\n"
                  "  oracle on a small shape (--tags small --oracle).")
        return 1
    if ran != len(expected):
        print(f"FAIL  ran {ran} of {len(expected)} cases")
        return 1
    print(f"PASS  {ran} output hashes match ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
