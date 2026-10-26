#!/usr/bin/env python3
"""Issue 21: device-level unit test of the binary64-limb CRT (src/contract.cu crt_f64_elem) and of the integer running CRT, on the edge vectors.

    python3 tests/test_crt_f64_unit.py [--lib build/libimmerdink.so] [--quick]

The hook imd_test_crt_elem (src/test_hooks.cpp, test builds only) runs the REAL device functions on caller-supplied residue vectors and returns the multi-word integer
x in [0, P) BEFORE crt_finish / round_scaled, so errors in the low bits of the 128-bit value are visible (the GEMM test tests/test_crt_f64_gpu.py compares final
doubles and its random data never reaches the edges below). Inputs, for every N = 2..20 and each of the 4 byte positions e of the plane word:

  * the edge vectors of spec/test_crt_f64.py: z in {0, 1, 2, 3, 255, 256, 65535, H-1, H, H+1, P-1, P-2, P-3, P-256, P//2, P//3, P//255}, the 300 smallest and 300
    largest z, W_l = p_l - 1 (z = P - 1), W = 0;
  * exponentially spaced offsets from 0 and from P - 1: z = floor(P 2^-b) + {-2..2} and P - 1 - floor(P 2^-b) + {-2..2} for b = 1..46 (the quotient estimate's
    edge: z0 within ~2^-30 P of 0 or P, where t = -1 or the estimate is one low; 2^-30 is b = 30), and random z of every bit length near both ends;
  * random z in [0, P), and random z within a few multiples of P 2^-30 of both ends;
  * the vectors that maximize / minimize each limb remainder r_j = s_j - t P_j (the carries change sign there);
  * vectors found by a differential search ON THE DEVICE: 100,000 each of three families (z within 2^-s P of P for s = 30..46, sparse residue vectors, uniform
    residues) are run through the shipped limb CRT and through every mutation, and up to 24 per family on which a mutation differs are kept (so the corners that
    only a weakened bias or a rarely taken branch reaches are in the set, found by the real code and then checked against the exact value like every other vector).
  (Any residue vector is the image of exactly one z in [0, P), so choosing z and setting W_l = z mod p_l covers the whole input space; z is computed from the
  residues by the textbook formula.)

Checks: the integer CRT and the limb CRT both equal z (= spec/oracle.py crt before centring = spec/crt_f64.py) exactly; the other three bytes of each plane word hold
unrelated data and are ignored (byte positions e = 0..3). Mutation check: the hook also instantiates the limb CRT with compile-time defects (Mut knob of
crt_f64_elem); each device mutation must equal a bit-exact Python model of the mutated arithmetic on every vector (so it does what it says), and must be CAUGHT
(differ from z on some vector) at every N where it is a defect (benign(): proved equivalent to the shipped code otherwise): bias 0 and 2^-46, a wrong low limb constant
k[N][N-1][0], a wrong Pj[0], the bottom limb's carry dropped or taken by a logical shift, R == P not reduced. Bias 2^-40 is reported only (see INFORMATIONAL).
"""
import argparse
import math
from fractions import Fraction
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spec"))
sys.path.insert(0, HERE)
import fma_compat  # noqa: E402,F401
import oracle as o  # noqa: E402
import crt_f64 as C  # noqa: E402
import harness as hz  # noqa: E402

WORDS = 5   # src/contract.cu kTestCrtWords
INT, LIMB, FINK = 0, 1, 10   # FINK: the limb CRT of the k5fin kernels (issue 23: the final subtraction of P as one carry chain), checked against the same integers
# Mutation 2, the bias 2^-40, is reported but not required to be caught: the worst-case error of the quotient estimate (spec/crt_f64.py bound 2^-37) does not
# exclude it from being a defect, but the observed overshoot of the estimate with no bias at all is at most about 2^-43 in every search (the real error is a few
# ulps of a quotient below 2^11, ulp 2^-42 and below), so 2^-40 works on every vector found; it is a defect only for a vector of quotient above 2^11 with a
# near-worst alignment of the roundings, which no search here (about 300,000 vectors per N, three families, plus the edge sets) produced. Bias 2^-46 (mutation 8) is
# a defect that is caught.
INFORMATIONAL = 2
MUTS = {  # variant -> (name, Mut)
    2: "bias 0", 3: "bias 2^-40", 9: "bias 2^-46", 4: "k[N][N-1][0] + 1", 5: "Pj[0] + 1", 6: "bottom carry dropped", 7: "bottom carry by logical shift", 8: "R == P not reduced"}


def vectors(c, rng, quick):
    """The z values of one N (see the module docstring)."""
    P, H = c.P, c.plan.H
    zs = {0, 1, 2, 3, 255, 256, 65535, H - 1, H, H + 1, P - 1, P - 2, P - 3, P - 256, P // 2, P // 3, P // 255}
    zs |= set(range(0, 300)) | set(range(P - 300, P))
    for b in range(1, 47):
        d = P >> b
        for r in range(-2, 3):
            zs.add(d + r)
            zs.add(P - 1 - d + r)
    nb = P.bit_length()
    nrand = 400 if quick else 1500
    for _ in range(nrand):
        w = rng.randint(1, nb)                          # a random z of every bit length, and random offsets below P
        zs.add(rng.randrange(1 << w))
        zs.add(P - 1 - rng.randrange(1 << w))
        zs.add(rng.randrange(P))
    # the edge of the 2^-30 bias: random z within a few multiples of P 2^-30 of both ends
    e30 = P >> 30
    for _ in range(nrand):
        zs.add(rng.randrange(0, 4 * e30 + 4))
        zs.add(P - 1 - rng.randrange(0, 4 * e30 + 4))
    return sorted({z % P for z in zs})


def extremal_cols(c):
    """Residue vectors that maximize / minimize each limb remainder r_j = s_j - t P_j (t ~ S / P = sum_l W_l K_l / P): W_l = p_l - 1 where the term W_l (K_l P_j / P - k_lj)
    is positive (negative), else 0, and the same with half the residues. Not reachable by random draws, but they are where the limb carries change sign."""
    cols = []
    for j in range(c.J):
        for sign in (1, -1):
            hot = [sign * (c.K[l] * c.Pj[j] - c.k[l][j] * c.P) > 0 for l in range(c.N)]
            cols.append([(c.plan.p[l] - 1) if hot[l] else 0 for l in range(c.N)])
            cols.append([(c.plan.p[l] - 1) // 2 if hot[l] else 0 for l in range(c.N)])
            cols.append([(c.plan.p[l] - 1) if hot[l] else (c.plan.p[l] - 1) // 3 for l in range(c.N)])
    return cols


def model_device(c, W, mut):
    """A bit-exact Python model of crt_f64_elem<N, L, Mut> (mut 0 = shipped): returns the word x the device stores (an integer below 2^(32 L)),
    which equals z only if there is no defect."""
    N, J, w, P = c.N, c.J, c.w, c.P
    L = params_L32(N)
    k = [list(map(float, c.k[l])) for l in range(N)]
    Pj = [float(x) for x in c.Pj]
    if mut == 3:
        k[N - 1][0] += 1.0
    if mut == 4:
        Pj[0] += 1.0
    bias = {1: 0.0, 2: 2.0 ** -40, 8: 2.0 ** -46}.get(mut, 2.0 ** -30)
    s = [0.0] * J
    for l in range(N):
        wl = float(W[l])
        for j in range(J):
            s[j] = math.fma(wl, k[l][j], s[j])
    Sd = s[J - 1]
    for j in range(J - 2, -1, -1):
        Sd = math.fma(Sd, float(c.B), s[j])
    t = math.floor(math.fma(Sd, c.Pinv, -bias))
    assert -(1 << 31) <= t < (1 << 31)
    carry, limbs = 0, []
    for j in range(J):
        r = round(math.fma(-float(t), Pj[j], s[j]))
        a = r + carry
        assert -(1 << 63) <= a < (1 << 63)
        limbs.append(a & ((1 << w) - 1))
        carry = a >> w
        if mut == 5 and j == 0:
            carry = 0
        if mut == 6 and j == 0:
            carry = (a & ((1 << 64) - 1)) >> w
    x = sum(limb << (w * j) for j, limb in enumerate(limbs)) & ((1 << (32 * L)) - 1)
    if x >= P and not (mut == 7 and x == P):
        x -= P
    return x


def params_L32(N):
    return C.params.limbs_for(o.Plan(N, 0).P)


def benign(c, mut):
    """True where a mutation is provably no defect (equivalent to the shipped code), so it need not be caught.
    * bias b (mutations 1, 2, 8): the quotient estimate's error is below tmax (J + 1) 2^-52 (spec/crt_f64.py); an overshoot needs the computed quotient to reach an
      integer n + 1 + b while the true one is below n + 1 by at least 1/P (z0 <= P - 1): impossible if tmax (J + 1) 2^-52 < b + 1/P.
    * one limb (J = 1), mutations 5, 6: the carry out of the bottom limb is the carry out of the top limb, which is dropped by design.
    * mutation 6 (logical shift of a negative r_0 adds 2^24 to limb 1, i.e. 2^64 to x): vanishes when the word has 64 bits or less, and when r_0 is never negative:
      r_0 = s_0 - t P_0 = sum_l W_l (k_l0 - K_l P_0 / P) + (S/P - t) P_0 >= the minimum of the linear part over W_l in [0, p_l - 1] (exact, t <= S / P)."""
    err = c.tmax * (c.J + 1) * 2.0 ** -52
    L = params_L32(c.N)
    if mut in (1, 2, 8):
        return err < {1: 0.0, 2: 2.0 ** -40, 8: 2.0 ** -46}[mut] + 1.0 / c.P
    if mut in (5, 6) and c.J == 1:
        return True
    if mut == 6:
        if L <= 2:
            return True
        lin = sum((c.plan.p[l] - 1) * min(0, Fraction(c.k[l][0]) - Fraction(c.K[l] * c.Pj[0], c.P)) for l in range(c.N))
        return lin >= 0
    return False


def z_of(c, W):
    """The CRT of the residues W by the textbook formula (independent of the code under test)."""
    pl = c.plan
    return sum(pl.M[l] * ((pl.q[l] * int(W[l])) % pl.p[l]) for l in range(c.N)) % c.P


def search_batches(c, gen, n):
    """Residue matrices uint8 [N][n] of three families that reach the corners of the arithmetic: z within 2^-s P of P (s = 30..46: the quotient estimate's overshoot
    edge), sparse residue vectors (most W_l = 0 or p_l - 1: the limb remainder r_0 = s_0 - t P_0 is negative only here), uniform residues."""
    p = np.array(c.plan.p, dtype=np.uint64)[:, None]
    nb = c.P.bit_length()
    out = {}
    s_bits = gen.integers(30, 47, size=n)
    w = np.maximum(nb - s_bits, 1)                                        # r < 2^w, z = P - 1 - r
    lo = gen.integers(0, 2 ** 64, size=n, dtype=np.uint64)
    hi = gen.integers(0, 2 ** 64, size=n, dtype=np.uint64)
    lo = lo & np.where(w >= 64, np.uint64(0xFFFFFFFFFFFFFFFF), (np.uint64(1) << np.minimum(w, 63).astype(np.uint64)) - np.uint64(1))
    hi = np.where(w > 64, hi & ((np.uint64(1) << np.clip(w - 64, 0, 63).astype(np.uint64)) - np.uint64(1)), np.uint64(0))
    two64 = np.array([(1 << 64) % pl for pl in c.plan.p], dtype=np.uint64)[:, None]
    r = ((hi[None, :] % p) * two64 + lo[None, :] % p) % p
    out["near P"] = ((p - np.uint64(1) - r) % p).astype(np.uint8)
    full = np.array([gen.integers(0, pl, size=n) for pl in c.plan.p], dtype=np.uint8)
    pick = gen.integers(0, 3, size=(c.N, n))
    mx = (p - np.uint64(1)).astype(np.uint8)
    sparse = np.where(pick == 0, 0, np.where(pick == 1, mx, full)).astype(np.uint8)
    keep = gen.random((c.N, n)) < gen.random((1, n))                      # a random density per vector
    out["sparse"] = np.where(keep, sparse, 0).astype(np.uint8)
    out["uniform"] = full
    return out


def find_breakers(lib, cuda, c, variant, nsearch, keep=24):
    """Residue vectors (columns) on which mutation `variant` differs from the shipped limb CRT on the device, found by running both on many vectors."""
    gen = np.random.Generator(np.random.PCG64(1000 * c.N + variant))
    found = []
    d_W, d_out = cuda.malloc(c.N * nsearch), cuda.malloc(nsearch * WORDS * 4)
    try:
        for name, Wm in search_batches(c, gen, nsearch).items():
            ref = run_raw(lib, cuda, c.N, LIMB, 0, Wm, d_W, d_out)
            mut = run_raw(lib, cuda, c.N, variant, 0, Wm, d_W, d_out)
            for i in np.nonzero((ref != mut).any(axis=1))[0][:keep]:
                found.append(Wm[:, i].copy())
    finally:
        cuda.free(d_W)
        cuda.free(d_out)
    return found


def run_raw(lib, cuda, N, variant, e, Wmat, d_W, d_out):
    count = Wmat.shape[1]
    cuda.upload(d_W, np.ascontiguousarray(Wmat))
    lib.crt_elem(N, variant, e, count, d_W, d_out)
    cuda.sync()
    out = np.empty((count, WORDS), dtype=np.uint32)
    cuda.download(out, d_out)
    return out


def run(lib, cuda, N, variant, e, Wmat, d_W, d_out):
    """Device word x (Python ints) for each column of Wmat (uint8 [N][count])."""
    count = Wmat.shape[1]
    cuda.upload(d_W, np.ascontiguousarray(Wmat))
    cuda.memset(d_out, 0xCD, count * WORDS * 4)
    lib.crt_elem(N, variant, e, count, d_W, d_out)
    cuda.sync()
    out = np.empty((count, WORDS), dtype=np.uint32)
    cuda.download(out, d_out)
    L = params_L32(N)
    assert not out[:, L:].any(), "words above L32(N) must be zero"
    return [sum(int(out[i, k]) << (32 * k) for k in range(L)) for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true", help="fewer random vectors")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks or not hasattr(lib.lib, "imd_test_crt_elem"):
        print("FAIL  the library has no imd_test_crt_elem (build with IMMERDINK_BUILD_TESTS=ON)")
        return 1
    rng = random.Random(2121)
    fails, total, caught = [], 0, {v: {} for v in MUTS}
    model_cache, found_stats = {}, {}
    stats = {"t_minus_1": 0, "r_eq_P_or_above": 0}
    for N in range(o.N_MIN, o.N_MAX + 1):
        c = C.F64Crt(N)
        p, P = c.plan.p, c.P
        zs = vectors(c, rng, args.quick)
        cols = [[z % pl for pl in p] for z in zs] + extremal_cols(c)
        # vectors found by a differential search on the device (see find_breakers): where a weakened bias or a rarely taken branch makes a difference
        nfound = {}
        for variant in MUTS:
            br = find_breakers(lib, cuda, c, variant, 20000 if args.quick else 100000)
            nfound[variant] = len(br)
            cols += [list(map(int, W)) for W in br]
        seen = {}
        for W in cols:
            seen.setdefault(tuple(W), z_of(c, W))
        cols = list(seen)
        zs = [seen[W] for W in cols]
        count = len(zs)
        Wmat = np.array(cols, dtype=np.uint8).T.copy()                               # [N][count]
        # the largest limb sums: W_l = p_l - 1 is z = P - 1 (in zs); a zero vector is z = 0 (in zs)
        assert Wmat.shape == (N, count)
        d_W, d_out = cuda.malloc(Wmat.nbytes), cuda.malloc(count * WORDS * 4)
        try:
            # the model and the reference agree with z (spec/test_crt_f64.py runs 63,841 vectors; here the very vectors of the device)
            for i in range(0, count, max(1, count // 400)):
                z, W = zs[i], list(cols[i])
                zc = z if z <= c.plan.H else z - P
                assert c.crt(W) == z and o.crt(W, c.plan) == zc, (N, z)
            for e in range(4):
                for variant in (INT, LIMB, FINK):
                    got = run(lib, cuda, N, variant, e, Wmat, d_W, d_out)
                    total += count
                    bad = [(zs[i], got[i]) for i in range(count) if got[i] != zs[i]]
                    if bad:
                        z, g = bad[0]
                        fails.append(f"N={N} e={e} {'integer' if variant == INT else 'limb' if variant == LIMB else 'k5fin limb'} CRT: {len(bad)} of {count} wrong, first z={z:#x} got {g:#x} (xor {z ^ g:#x})")
                    # the shipped model is bit-exact too (also checks the model_device code against the device for mut 0)
                if e == 0 or e == 3:
                    for variant, name in MUTS.items():
                        mut = variant - 1
                        got = run(lib, cuda, N, variant, e, Wmat, d_W, d_out)
                        total += count
                        if (N, variant) not in model_cache:
                            model_cache[(N, variant)] = [model_device(c, W, mut) for W in cols]
                        want = model_cache[(N, variant)]
                        nm = sum(1 for i in range(count) if got[i] != want[i])
                        if nm:
                            fails.append(f"N={N} e={e} mutation '{name}': the device differs from the Python model of the mutated arithmetic on {nm} of {count} vectors, first z={zs[next(i for i in range(count) if got[i] != want[i])]:#x}")
                        wrong = sum(1 for i in range(count) if got[i] != zs[i])
                        caught[variant][N] = max(caught[variant].get(N, 0), wrong)
            # statistics of the edge: vectors where the quotient estimate sits one low (R >= P before the conditional subtraction)
            for W in cols:
                s = c.sums(list(W))
                Sd = s[c.J - 1]
                for j in range(c.J - 2, -1, -1):
                    Sd = math.fma(Sd, float(c.B), s[j])
                t = math.floor(math.fma(Sd, c.Pinv, -C.EPS_T))
                S = sum(int(s[j]) << (c.w * j) for j in range(c.J))
                if t == -1:
                    stats["t_minus_1"] += 1
                if S - t * P >= P:
                    stats["r_eq_P_or_above"] += 1
        finally:
            cuda.free(d_W)
            cuda.free(d_out)
        found_stats[N] = nfound
    # mutation verdicts: a defect must be caught at every N where it is one
    summary = []
    for variant, name in MUTS.items():
        mut = variant - 1
        miss, benign_ns, info_ns = [], [], []
        for N in range(o.N_MIN, o.N_MAX + 1):
            c = C.F64Crt(N)
            if caught[variant][N] > 0:
                continue
            if benign(c, mut):
                benign_ns.append(N)
                continue
            if mut == INFORMATIONAL:
                info_ns.append(N)
                continue
            miss.append(N)
        ns = sorted(N for N, w in caught[variant].items() if w > 0)
        where = f"N = {ns[0]}..{ns[-1]}" if ns else "no N"
        summary.append(f"    mutation {variant - 1} '{name}': caught at {where} ({len(ns)} of {o.N_MAX - o.N_MIN + 1} N)" + (f", no defect at N = {benign_ns} (proved below)" if benign_ns else "") + (f", not caught at N = {info_ns} (informational: see INFORMATIONAL)" if info_ns else "") + (f", NOT caught at N = {miss}" if miss else ""))
        if miss:
            fails.append(f"mutation '{name}' not caught at N = {miss}")
    if fails:
        print("FAIL")
        for f in fails[:40]:
            print("  " + f)
        return 1
    print(f"PASS  {total} device CRT evaluations over N = {o.N_MIN}..{o.N_MAX}, byte positions e = 0..3: integer CRT and limb CRT equal z exactly "
          f"({stats['t_minus_1']} vectors with t = -1, {stats['r_eq_P_or_above']} with R >= P before the last subtraction); mutations:")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
