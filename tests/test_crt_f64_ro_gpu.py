#!/usr/bin/env python3
"""Issue 21: device-level test of the K5 exact FP64 finish 'k5ro' (seq bit 8192) against its bit-exact CPU model and the oracle.

    python3 tests/test_crt_f64_ro_gpu.py [--lib build/libimmerdink.so] [--quick]

What it checks. The k5ro finish (src/contract.cu crt_ro_try, spec in issue 21 comment 6750980047) rebuilds D = RN(z 2^e) from the N residues of one K5 output in the
FP64 pipe, and hands a few elements back to the integer finish (the "fallback": |z~| near P/2, subnormal or overflowing results). spec/crt_f64_ro.py is a
bit-exact Python model of that instruction sequence; spec/test_crt_f64_ro.py proved it against the oracle on CPU (gate 1). This test runs the REAL device code
(hook imd_test_crt_elem variant 11) on the same kinds of vectors and requires, for every vector:

  * the final double (the k5fin kernels' k5f_finish_ro: the fast path, or the integer finish on its fallback) equals the oracle RN(z 2^e) bit for bit;
  * the device took the fast path exactly when the model does (so the device computed the same z~ and the same fallback decision), and on the fast path its
    double equals the model's.

The GEMM test tests/test_crt_f64_gpu.py runs k5ro end to end, but its random data almost never reaches the centring window (2^-37 of uniform values) or the
exponent edges: the vectors here are the gate-1 adversarial set (spec/test_crt_f64_ro.py adversarial(): z within 2^-36 P of +-P/2, x = P/2, ties and near
ties of the final rounding, the subnormal and overflow edges, lemma 7's double-rounding trap), the corpus edge values, and random z, for every N = 2..17
(N = 18..20 keep the integer finish: the hook refuses them) and every byte position of the plane word.
"""
import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spec"))
sys.path.insert(0, HERE)
import fma_compat  # noqa: E402,F401
import crt_f64_ro as R  # noqa: E402
import harness as hz  # noqa: E402
import oracle as o  # noqa: E402
import test_crt_f64_ro as G  # noqa: E402

WORDS = 5          # kTestCrtWords in src/contract.cu
VARIANT_RO = 11


def vectors(c, rng, many):
    """(x, e) pairs: gate 1's adversarial set, the edge values of z, and random z with random normal-range exponents."""
    P = c.P
    nb = P.bit_length()
    out = list(G.adversarial(c, rng, many))
    for zz in (0, 1, 2, 3, 255, 256, 65535, P // 2 - 1, P // 2, P // 2 + 1, P - 1, P - 2, P // 3, P // 255):
        for e in (0, -60, 17):
            out.append((zz % P, e))
    for _ in range(4 * many):
        out.append((rng.randrange(P), rng.randint(-1000, 1022 - nb)))
    return out


def run_device(lib, cuda, N, eb, W, ex):
    """The hook on residue matrix W [N][count] (uint8) and exponents ex [count] (int32): (D bits uint64, fast flag, fast bits uint64) per vector."""
    count = W.shape[1]
    d_W, d_out = cuda.malloc(W.nbytes), cuda.malloc(count * WORDS * 4)
    try:
        cuda.upload(d_W, np.ascontiguousarray(W))
        buf = np.zeros((count, WORDS), dtype=np.uint32)
        buf[:, 0] = ex.astype(np.int32).view(np.uint32)
        cuda.upload(d_out, buf)
        lib.crt_elem(N, VARIANT_RO, eb, count, d_W, d_out)
        cuda.sync()
        cuda.download(buf, d_out)
    finally:
        cuda.free(d_W)
        cuda.free(d_out)
    d = buf[:, 0].astype(np.uint64) | (buf[:, 1].astype(np.uint64) << np.uint64(32))
    f = buf[:, 3].astype(np.uint64) | (buf[:, 4].astype(np.uint64) << np.uint64(32))
    return d, buf[:, 2], f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true", help="fewer random and adversarial vectors per N")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    rng = random.Random(20261113)
    many = 40 if args.quick else 200
    bad, total, fallbacks = [], 0, 0
    for N in range(2, 18):
        c = R.F64CrtRO(N)
        p = c.plan.p
        vec = vectors(c, rng, many)
        W = np.array([[x % pl for x, _ in vec] for pl in p], dtype=np.uint8)
        ex = np.array([e for _, e in vec], dtype=np.int64)
        want_d, want_fast, want_fd = [], [], []
        for x, e in vec:
            Wv = [x % pl for pl in p]
            ref = o.round_dyadic(o.crt(Wv, c.plan), e)           # the oracle: centred z, then one rounding of z 2^e
            d, why = c.finish(Wv, e)                              # the CPU model of the device sequence (gate 1)
            if R.f2b(d) != R.f2b(ref):
                raise SystemExit(f"model/oracle disagree at N={N} x={x} e={e}: the CPU gate (spec/test_crt_f64_ro.py) must pass first")
            want_d.append(R.f2b(ref))
            want_fast.append(1 if why is None else 0)
            want_fd.append(R.f2b(d) if why is None else 0)
        for eb in range(4):
            d, fast, fd = run_device(lib, cuda, N, eb, W, ex)
            for i in range(len(vec)):
                if int(d[i]) != want_d[i]:
                    bad.append(f"N={N} byte {eb} x={vec[i][0]} e={vec[i][1]}: D {int(d[i]):#018x}, oracle {want_d[i]:#018x}")
                elif int(fast[i]) != want_fast[i]:
                    bad.append(f"N={N} byte {eb} x={vec[i][0]} e={vec[i][1]}: fast path {int(fast[i])}, model {want_fast[i]}")
                elif want_fast[i] and int(fd[i]) != want_fd[i]:
                    bad.append(f"N={N} byte {eb} x={vec[i][0]} e={vec[i][1]}: fast double {int(fd[i]):#018x}, model {want_fd[i]:#018x}")
                if len(bad) > 40:
                    break
            total += len(vec)
        fallbacks += len(vec) - sum(want_fast)
        print(f"  N={N:2d}: {len(vec)} vectors x 4 byte positions, {len(vec) - sum(want_fast)} take the fallback")
    if bad:
        print("FAIL")
        for b in bad[:40]:
            print("  " + b)
        return 1
    print(f"PASS  {total} device finishes (N = 2..17, 4 byte positions): D equals the oracle bit for bit, and the fast path / fallback decision and the fast double "
          f"equal the CPU model's on every vector ({fallbacks} vectors per byte position take the fallback)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
