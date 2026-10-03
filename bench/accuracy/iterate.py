#!/usr/bin/env python3
"""M5 accuracy study, part 2: a Jacobi fixed-point iteration built on one contraction.

    tools/memcap.sh python3 bench/accuracy/iterate.py [--out FILE]

Solve (Delta + W) x = r for x[a,b,i,j], where Delta[a,b,i,j] = e_a + e_b - e_i - e_j is
a diagonal of "orbital-energy-like" denominators (in [2, 10]) and W[a,b,c,d] is a dense
symmetric operator acting through the contraction

    y[a,b,i,j] = sum_cd W[a,b,c,d] x[c,d,i,j]          (GEMM view m = k = v^2, n = o^2)

by the Jacobi iteration x <- (r - y) / Delta from x = 0. W is scaled so that the spectral
radius of Delta^-1 W is RHO (< 1), so the iteration contracts and errors are damped by
1 / (1 - RHO) at most. The same iteration runs with the contraction done in native FP64
(numpy) and by imd_contract at several N (RN and TRUNC); the per-iteration drift
||x_N^t - x_64^t|| / ||x_64^t|| and the final true residual ||r - (Delta + W) x|| / ||r||
(evaluated in native FP64) are reported.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def setup(v, o, rho, seed):
    rng = np.random.default_rng(seed)
    ev = np.sort(rng.uniform(0.5, 3.0, v))
    eo = np.sort(rng.uniform(-2.0, -0.5, o))
    Delta = ev[:, None, None, None] + ev[None, :, None, None] - eo[None, None, :, None] - \
        eo[None, None, None, :]
    G = rng.standard_normal((v * v, v * v))
    Wm = (G + G.T) / 2                                   # symmetric in (ab) <-> (cd)
    # amplitude-like decay: couplings fall off with the "energy" of both pairs
    eab = (ev[:, None] + ev[None, :]).reshape(-1)
    Wm *= np.exp(-0.5 * (eab[:, None] + eab[None, :] - 2))
    # spectral radius of Delta^-1 W (similar to the symmetric D^-1/2 W D^-1/2, per (i,j))
    dmin = Delta.min(axis=(2, 3)).reshape(-1)            # worst (largest-gain) occ pair
    S = Wm / np.sqrt(dmin[:, None] * dmin[None, :])
    lam = np.max(np.abs(np.linalg.eigvalsh(S)))
    Wm *= rho / lam
    r = rng.standard_normal((v, v, o, o))
    return Delta, Wm.reshape(v, v, v, v), r


def run(contract, Delta, W, r, iters):
    x = np.zeros_like(r)
    hist = []
    for _ in range(iters):
        y = contract(W, x)
        x = (r - y) / Delta
        hist.append(x.copy())
    return x, hist


def residual(Delta, W, r, x):
    """||r - Delta x - W x|| / ||r|| in x87 extended precision (64-bit significand), so
    the residual of an FP64 iterate is not itself limited by FP64 rounding."""
    v = W.shape[0]
    L = np.longdouble
    assert np.finfo(L).nmant >= 63, "needs 80-bit long double"
    y = (W.reshape(v * v, v * v).astype(L) @ x.reshape(v * v, -1).astype(L)).reshape(x.shape)
    res = r.astype(L) - Delta.astype(L) * x.astype(L) - y
    return float(np.sqrt(np.sum(res * res)) / np.linalg.norm(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=None)
    ap.add_argument("--v", type=int, default=40)
    ap.add_argument("--o", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    v, o = args.v, args.o
    Delta, W, r = setup(v, o, args.rho, args.seed)
    runner = cm.Runner(args.lib)

    def native(W, x):
        return (W.reshape(v * v, v * v) @ x.reshape(v * v, -1)).reshape(x.shape)

    def emulated(N, rnd):
        def f(W, x):
            st, y = runner.contract("abcd", W, "cdij", x, "abij", x.shape, N, rnd)
            assert st == "SUCCESS", st
            return y
        return f

    t0 = time.time()
    x64, h64 = run(native, Delta, W, r, args.iters)
    res64 = residual(Delta, W, r, x64)
    print(f"v={v} o={o} rho={args.rho} iters={args.iters}: GEMM m=k={v * v} n={o * o}; "
          f"native FP64 final residual {res64:.2e}, last-step change "
          f"{np.linalg.norm(h64[-1] - h64[-2]) / np.linalg.norm(h64[-1]):.2e}")
    nrm = [np.linalg.norm(h) for h in h64]
    out = open(args.out, "w") if args.out else None
    checkpoints = [1, 5, 10, 20, 40, 60, 80, args.iters]
    print(f"{'run':>10} " + " ".join(f"{'it' + str(c):>9}" for c in checkpoints) +
          f" {'max drift':>10} {'residual':>9}")
    for rnd, Ns in (("RN", [8, 10, 12, 14, 16, 20]), ("TRUNC", [8, 12, 16])):
        for N in Ns:
            x, h = run(emulated(N, rnd), Delta, W, r, args.iters)
            drift = [float(np.linalg.norm(a - b) / n) for a, b, n in zip(h, h64, nrm)]
            res = residual(Delta, W, r, x)
            print(f"{rnd + ' N=' + str(N):>10} " +
                  " ".join(f"{drift[c - 1]:9.2e}" for c in checkpoints) +
                  f" {max(drift):10.2e} {res:9.2e}", flush=True)
            if out:
                out.write(json.dumps(dict(N=N, rounding=rnd, v=v, o=o, rho=args.rho,
                                          iters=args.iters, drift=drift, residual=res,
                                          residual_fp64=res64)) + "\n")
    runner.close()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
