#!/usr/bin/env python3
"""Issue 23: the dynamic cycle split of the KL pack kernel (sections) on a square contraction, from the clock64() section timers.

    python3 tools/kl_timers.py --lib <build>/libimmerdink.so [--shapes 4096 8192] [--seq 1 17 33 ...] [--N 14] [--b-kn]

Needs a library built with -DIMMERDINK_KL_TIMERS=1 (sections) or =2 (also the modulus loop's emit and store apart; perturbs that loop) and
IMMERDINK_BUILD_TESTS=ON (the hook imd_test_kl_timers lives in src/test_hooks.cpp); build only the `immerdink` target. Any GPU: on the cluster
no Hopper counters exist, this is the dynamic split there. For every shape and instruction-sequence set it runs one warm-up call, resets the timers,
runs one call (two KL launches: A and B) and prints, per section, the warp-cycles per warp-tile (lane 0 of every warp stamps clock64()), the share of
the sum, and the CTA lifetime. Sections (contract.cu kl_scale_pack): setup (tile row info, divmods), load (pass 1: gather or direct loads + the
flag / gamma scan), red1 (gmax / flag reduction), ssum (pass 2: the u128 square sum), red2 (the S reduction), sigma (largest_u, broadcast, row
offsets, sigma / flag stores), reload (chunks re-read in pass 3), quant (the emitter's quantize / scale), emit (the modulus loop: emit and store; with
level 2 emit only), store (level 2 only: the store), tail (final barrier). Barrier waits are inside the section that ends at the barrier. The sections
are sums over warps: the shares are the split of a warp's time, not of the kernel's wall time. Rule R4 reads: sigma + red1 + red2 >= 10% -> item B1
(klwarp); ssum >= 15% -> the FP64-pipe S (not part of issue 23).
The GPU lock (IMMERDINK_GPU_LOCK, default as tools/test.sh) is held for the runs. No timing claims from a card other than the target.
"""
import argparse
import fcntl
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
import harness as hz  # noqa: E402
import test_output_hashes as T  # noqa: E402

SECTIONS = ["setup", "load", "red1", "ssum", "red2", "sigma", "reload", "quant", "emit", "store", "tail"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--shapes", type=int, nargs="*", default=[4096, 8192], help="square sizes m = n = k")
    ap.add_argument("--seq", type=int, nargs="*", default=[-1], help="forced instruction-sequence sets (-1: the device class's)")
    ap.add_argument("--N", type=int, default=14)
    ap.add_argument("--b-kn", action="store_true", help="B as [k, n] (the usual row-major NN layout: B is read along rows, klload does not apply to it)")
    args = ap.parse_args()
    lock = os.environ.get("IMMERDINK_GPU_LOCK", os.path.expanduser("~/.cache/ehrenfest/gpu.lock"))
    if lock != "none":
        if not os.path.isdir(os.path.dirname(lock)):
            lock = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "gpu.lock")
        lf = open(lock, "a")
        fcntl.flock(lf, fcntl.LOCK_EX)
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks or lib.kl_timers() is None:
        print("the library has no KL timers: build with -DIMMERDINK_KL_TIMERS=1 (or 2) and IMMERDINK_BUILD_TESTS=ON", file=sys.stderr)
        return 2
    hashes = {}
    for n in args.shapes:
        B = {"extent": [n, n], "label": [3, 2]} if args.b_kn else {"extent": [n, n], "label": [2, 3]}
        case = {"name": f"sq{n}", "A": {"extent": [n, n], "label": [1, 3]}, "B": B, "C": {"extent": [n, n], "label": [1, 2]}, "N": args.N, "family": "u"}
        for seq in args.seq:
            lib.set_seq(seq)
            lib.set_fused(-1)
            T.run_gpu(lib, cuda, case)       # warm-up (plan creation, first-call setup)
            lib.kl_timers(reset=True)
            lib.kl_launch_counts(reset=True)
            hashes.setdefault(n, set()).add(T.hash_c(T.run_gpu(lib, cuda, case)))
            t = lib.kl_timers(reset=True)
            launches = lib.kl_launch_counts(reset=True)
            wt = t[11] or 1
            tot = sum(t[:11]) or 1
            print(f"\nsquare {n}, N = {args.N}, seq {seq} ({'B as [k n]' if args.b_kn else 'B as [n k]'}), KL launches {launches}: {t[12]} CTAs, {t[11]} warp-tiles, "
                  f"CTA lifetime {t[13] / max(t[12], 1):.0f} cycles (mean), sum of sections {tot / wt:.0f} cycles per warp-tile")
            print(f"  {'section':8s} {'cyc/warp-tile':>14s} {'share':>7s}")
            for i, name in enumerate(SECTIONS):
                if t[i]:
                    print(f"  {name:8s} {t[i] / wt:14.1f} {100.0 * t[i] / tot:6.1f}%")
            fixed = t[2] + t[4] + t[5]
            print(f"  sigma + red1 + red2 = {100.0 * fixed / tot:.1f}% (R4: >= 10% -> klwarp)   ssum = {100.0 * t[3] / tot:.1f}% (R4: >= 15% -> FP64-pipe S)")
    for n, hs in hashes.items():
        print(f"\nC hash over the sets at {n}: {sorted(hs)} ({'identical' if len(hs) == 1 else 'DIFFERENT'})")
    lib.set_seq(-1)
    return 0 if all(len(h) == 1 for h in hashes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
