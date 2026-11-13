"""Gate 1 of the K5 exact FP64 finish (issue 21): spec/crt_f64_ro.py against the reference, bit for bit. CPU only, single-threaded.

The finish computes D = RN(z 2^e) from the residues with FP64 operations only (a centred quotient, exact limb remainders, the Boldo-Melquiond
round-to-odd sum, an exponent add) and sends a few elements to today's integer finish (the "fallback"). The kill criterion of the gate: a result
that differs from the reference (oracle.crt then oracle.round_dyadic; spec/crt_f64.py's F64Crt is checked to agree) while the fallback predicate
did not fire. The sections:

  1. the round-to-odd emulation (ro_add) equals true round-to-odd, on random and on integer-structured binary64 pairs;
  2. the whole three-term sum (TwoSum, TwoSum, emulated RO, RN) at a small precision p, exhaustively over every triple of a value set: it equals
     RN(a + b + c) (the same algorithm as on binary64, at a size where every rounding pattern is reached);
  3. the 63,841-vector corpus of spec/test_crt_f64.py (the same z0 values, the same seed), each with e = 0 and with a random e;
  4. adversarial vectors for every N the finish covers: z within 2^-36 P of +-P/2 (the centred-quotient window), x = P/2 exactly, ties and near
     ties of the final rounding, exponents at the subnormal and overflow edges (with the double-rounding trap of lemma 7);
  5. lemma 5 (the spec obligation) on every vector: the quotient misrounds exactly when the exact remainder R is outside [-P/2, P/2), every misround
     is caught by both the spec's threshold and the minimal one RN(P/2), and lies in the band |z| >= P/2 - delta_N P;
  6. mutations: each planted bug (RN instead of RO, the RO nudge without the parity test or in the wrong direction, no centring clause, a threshold
     one ulp above RN(P/2), the range test on the scaled product) gives an uncaught mismatch, so the vectors can see each rule;
  7. the fraction of elements that take the fallback, per N.

    python3 spec/test_crt_f64_ro.py            (about 35 s)
"""
import math
import os
import random
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_f64_ro as R  # noqa: E402
import oracle as o  # noqa: E402
import test_crt_f64 as T  # noqa: E402

FOCUS = (14, 15, 16)          # the N of the Hopper kernels (k5f_crt_n<14,4>) and their neighbours: more adversarial vectors


def residues(x, p):
    return [x % pl for pl in p]


# ---- 1. the RO emulation -------------------------------------------------------------------------------------------------------------------
def check_ro(rng):
    n = 0
    pairs = []
    for _ in range(40000):
        a = rng.uniform(-1, 1) * 2.0 ** rng.randint(-60, 60)
        b = rng.uniform(-1, 1) * 2.0 ** rng.randint(-60, 60)
        pairs.append((a, b))
    for _ in range(40000):                      # integer-structured, like tl and ul: sizes far apart, power-of-two edges
        k = rng.randint(54, 170)
        a = float(rng.choice([1, -1]) * ((1 << k) + rng.choice([0, 1, -1]) * rng.randrange(1 << rng.randint(0, k - 50))))
        b = float(rng.choice([1, -1]) * rng.randrange(1, 1 << rng.randint(1, k - 1)))
        pairs.append((a, b))
    for a, b in pairs:
        got = R.ro_add(a, b)
        want = R.ro_exact(Fraction(a) + Fraction(b))
        assert R.f2b(got) == R.f2b(want), (a, b, got, want)
        n += 1
    print(f"ok  ro_add (RN, TwoSum error, +-1 on the bit pattern) equals round-to-odd on {n} binary64 pairs")


# ---- 2. the three-term sum at a small precision, exhaustively ------------------------------------------------------------------------------
def make_small(p, plain=False):
    """RN, TwoSum, emulated RO and the BM sum at precision p on integers (no underflow: the values are integers, as the limbs are).
    `plain` replaces the RO by RN (the mutation the check must see)."""
    def rn(x):
        a = -x if x < 0 else x
        n = a.bit_length()
        if n <= p:
            return x
        sh = n - p
        m, rem = a >> sh, a & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and m & 1):
            m += 1
        m <<= sh
        return -m if x < 0 else m

    def two_sum(a, b):                          # the same six operations as crt_f64_ro.two_sum
        s = rn(a + b)
        bb = rn(s - a)
        err = rn(rn(a - rn(s - bb)) + rn(b - bb))
        return s, err

    def ro_add(a, b):                           # the same steps as crt_f64_ro.ro_add
        s, err = two_sum(a, b)
        if err == 0 or plain:
            return s
        m = -s if s < 0 else s
        n = m.bit_length()
        assert n > p
        if (m >> (n - p)) & 1 == 0:
            if (err > 0) == (s > 0):
                m += 1 << (n - p)
            else:
                m -= (1 << (n - 1 - p)) if m == 1 << (n - 1) else (1 << (n - p))
        return -m if s < 0 else m

    def ro_true(x):
        a = -x if x < 0 else x
        n = a.bit_length()
        if n <= p or a & ((1 << (n - p)) - 1) == 0:
            return x
        lo = (a >> (n - p)) << (n - p)
        r = lo if (lo >> (n - p)) & 1 else lo + (1 << (n - p))
        return -r if x < 0 else r

    def bm3(a, b, c):
        uh, ul = two_sum(b, c)
        th, tl = two_sum(a, uh)
        v = ro_add(tl, ul)
        return rn(th + v), tl + ul, v
    return rn, two_sum, ro_true, bm3


def check_small(p, K, plain=False):
    rn, two_sum, ro_true, bm3 = make_small(p, plain)
    vals = set(range(1, 1 << p))
    vals |= {m << q for m in range(1 << (p - 1), 1 << p) for q in range(1, K)}
    vals = sorted(vals | {-v for v in vals} | {0})
    pos = [v for v in vals if v >= 0]
    bad = n = 0
    for a in pos:                               # a >= 0: negating all three inputs negates every step
        for b in vals:
            for c in vals:
                z, y, v = bm3(a, b, c)
                n += 1
                if z != rn(a + b + c) or (v != ro_true(y) and not plain):
                    bad += 1
    return n, bad


# ---- 3-6. the finish against the reference ------------------------------------------------------------------------------------------------
class Tally:
    def __init__(self):
        self.n = self.fast_ok = self.kill = self.caught_bad = self.misround = 0
        self.why = {"centre": 0, "range": 0}
        self.min_centre = 0
        self.kills = []

    def add(self, other):
        for k in ("n", "fast_ok", "kill", "caught_bad", "misround", "min_centre"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        for k in self.why:
            self.why[k] += other.why[k]
        self.kills += other.kills


def run_case(c, x, e, tl, ro_mut=0, thr=None, range_on_product=False):
    """One residue vector (x = z mod P) and exponent e: the finish, the reference, the lemma-5 checks. Returns True on an uncaught mismatch."""
    P, plan = c.P, c.plan
    W = residues(x, plan.p)
    zref = o.crt(W, plan)                                          # the oracle's centred z
    want = o.round_dyadic(zref, e)
    zt, t, Rx, _ = c.fast(W, ro_mut)
    why = c.fallback_reason(zt, e, range_on_product=range_on_product, thr=thr)
    if why is None:
        got = R.dmul_pow2(zt, e) if range_on_product else c.scale(zt, e)
    else:
        got = c.integer_finish(W, e)
    tl.n += 1
    # lemma 4 / 5 on the exact values
    S = sum(int(sj) for sj in c.sums_folded(W))
    tstar = (S - zref) // P
    assert S - tstar * P == zref and Rx == S - int(t) * P
    mis = int(t) != tstar
    assert mis == (2 * Rx >= P or 2 * Rx < -P), ("lemma 5", c.N, x)
    if mis:
        tl.misround += 1
        assert abs(int(t) - tstar) == 1
        assert Fraction(abs(zref)) >= Fraction(P, 2) - c.delta * P, ("misround outside the delta band", c.N, x)
        assert abs(zt) >= c.half_rn and abs(zt) >= c.thr_spec, ("misround not caught", c.N, x)
    else:
        assert Rx == zref
    if abs(zt) >= c.half_rn:
        tl.min_centre += 1
    if why is not None:
        tl.why[why] += 1
        if why == "centre" and R.f2b(R.dmul_pow2(zt, e)) != R.f2b(want):
            tl.caught_bad += 1                                     # the FP64 result would have been wrong: the fallback was needed
    if R.f2b(got) != R.f2b(want):
        assert why is None
        tl.kill += 1
        if len(tl.kills) < 5:
            tl.kills.append((c.N, x, e, got, want))
        return True
    if why is None:
        tl.fast_ok += 1
    return False


def adversarial(c, rng, many):
    """(x, e) pairs: the centring window, x = P/2, ties of the final rounding, the subnormal / overflow edges."""
    P = c.P
    nb = P.bit_length()
    out = []
    rand_e = lambda: rng.randint(-1000, 1022 - nb)                 # noqa: E731  a normal result for every z
    # the centred-quotient window: z = +-(P/2 - d), d small and d log-uniform up to 2^-36 P; x = P/2 itself
    span = max(1, (P >> 36))
    ds = list(range(0, 257)) + [rng.randrange(0, 1 << rng.randint(1, max(1, span.bit_length()))) for _ in range(many)]
    for d in ds:
        for zz in (P // 2 - d, -(P // 2) + d):
            if -(P // 2) <= zz < P // 2:
                out.append((zz % P, 0))
                out.append((zz % P, rand_e()))
    out.append((P // 2, 0))
    # ties and near ties of RN(R): |z| = (2m + 1) 2^s with a 54-bit odd significand, and +-1, +-2, +-2^(s-1) around it; the carry into the next binade
    if nb > 56:
        for _ in range(many):
            m = (1 << 53) | rng.getrandbits(53)                     # 54 bits, the last one dropped by RN
            m |= 1
            s = rng.randint(0, nb - 56)
            tie = m << s
            off = rng.randrange(1, 1 << rng.randint(1, s + 1))       # a sticky offset of any size below the tie's half ulp
            for dz in (0, 1, -1, 2, -2, 3, -3, off, -off) + ((1 << (s - 1), -(1 << (s - 1))) if s >= 1 else ()):
                for sg in (1, -1):
                    zz = sg * (tie + dz)
                    out.append((zz % P, rand_e() if rng.random() < 0.5 else 0))
        for k in range(55, nb - 1):
            for zz in ((1 << k) - (1 << (k - 54)), (1 << k) - (1 << (k - 54)) - 1, (1 << k) - (1 << (k - 54)) + 1, (1 << k) - 1):
                out.append((zz % P, 0))
                out.append(((-zz) % P, 0))
    # the subnormal and overflow edges: e such that ilogb(z) + e is around -1022 and 1023
    zs = [1, 3, (1 << 52) + 1] + [rng.randrange(1, P // 2) for _ in range(16)]
    if nb > 56:
        zs += [(((1 << 53) | rng.getrandbits(53)) | 1) << rng.randint(0, nb - 56) for _ in range(16)]
    for zz in zs:
        lg = zz.bit_length() - 1
        for target in list(range(-1078, -1018)) + list(range(1019, 1026)):
            for sg in (1, -1):
                out.append(((sg * zz) % P, target - lg))
    # lemma 7's trap: z = 2^k - 2^(k-53) - d rounds up to 2^k - 2^(k-53), which is a tie of the subnormal grid when z 2^e is near 2^-1022
    for k in range(54, nb - 1):
        for d in (1, 2, 3, 1 << max(0, k - 60)):
            zz = (1 << k) - (1 << (k - 53)) - d
            for sg in (1, -1):
                out.append(((sg * zz) % P, -1022 - k))
                out.append(((sg * zz) % P, -1023 - k))
    return out


def main():
    t0 = time.time()
    rng = random.Random(2111)
    check_ro(rng)

    # 2. the small-precision exhaustive check
    t1 = time.time()
    for p, K in ((4, 7), (5, 5)):
        n, bad = check_small(p, K)
        assert bad == 0, (p, K, bad)
        print(f"ok  BM sum with the emulated RO at precision {p}, exponents 0..{K - 1} and every integer below 2^{p}: all {n} triples (a >= 0) equal RN(a + b + c)")
        if p == 4:                              # (at p = 5 the exponent span 0..4 is too narrow for a double rounding to show)
            n, bad = check_small(p, K, plain=True)
            assert bad > 0
            print(f"    with RN in place of the RO: {bad} of {n} triples are wrong (the check sees double rounding)")
    print(f"    ({time.time() - t1:.1f} s)")

    # 3. the corpus of spec/test_crt_f64.py
    t1 = time.time()
    models, refused = {}, []
    corpus_total = 0
    per_n = {}
    erng = random.Random(7)
    for N, cref, zlist in T.corpus(random.Random(21)):
        cases = list(zlist) + [(-1) % cref.P]                        # + the W_l = p_l - 1 vector of test_crt_f64.main
        corpus_total += len(cases)
        try:
            c = models[N] = R.F64CrtRO(N)
        except o.NotSupported:
            refused.append((N, len(cases)))
            continue
        assert c.Pinv == cref.Pinv and c.Pj == cref.Pj and c.k == cref.k
        assert c.delta < Fraction(1, 1 << 39) and c.thr_spec <= c.half_rn and c.thr_spec < c.P / 2
        tl = Tally()
        nb = c.P.bit_length()
        H = c.plan.H
        for x in cases:
            assert c.crt_centred(residues(x, c.plan.p)) == o.crt(residues(x, c.plan.p), c.plan)   # F64Crt agrees with the oracle
            before = tl.why["centre"]
            run_case(c, x, 0, tl)
            run_case(c, x, erng.randint(-1000, 1022 - nb), tl)
            if tl.why["centre"] != before:
                assert x in (H - 1, H, H + 1), ("a centring fallback away from the corpus's boundary points", N, x)
        per_n[N] = tl
    assert corpus_total == 63841, corpus_total
    tc = Tally()
    for tl in per_n.values():
        tc.add(tl)
    print(f"ok  corpus: {corpus_total} residue vectors (spec/test_crt_f64.py, seed 21); N = {min(per_n)}..{max(per_n)} run with e = 0 and a random e: "
          f"{tc.n} finishes, {tc.kill} uncaught mismatches, {tc.why['centre']} centring + {tc.why['range']} range fallbacks, {tc.misround} quotient misrounds"
          f" ({time.time() - t1:.1f} s)")
    print("    quotient error bound delta_N (lemma 3): " + ", ".join(f"N={N} 2^{math.log2(models[N].delta):.1f}" for N in FOCUS)
          + f"; largest 2^{max(math.log2(m.delta) for m in models.values()):.1f}; every corpus centring fallback is one of x = H - 1, H, H + 1")
    print(f"    refused at plan time (J = 4 needs a four-term sum; those N keep the integer finish): "
          + ", ".join(f"N={N} ({k} vectors)" for N, k in refused))
    assert tc.kill == 0, tc.kills

    # 4-5. adversarial vectors
    t2 = time.time()
    adv = {}
    adv_cases = {}
    for N, c in models.items():
        cases = adversarial(c, rng, 1500 if N in FOCUS else 200)
        adv_cases[N] = cases
        tl = Tally()
        for x, e in cases:
            run_case(c, x, e, tl)
        adv[N] = tl
        assert tl.kill == 0, (N, tl.kills)
    ta = Tally()
    for tl in adv.values():
        ta.add(tl)
    assert ta.misround > 0
    print(f"ok  adversarial: {ta.n} finishes over N = {min(adv)}..{max(adv)} (window, x = P/2, ties, range edges): {ta.kill} uncaught mismatches; "
          f"{ta.misround} quotient misrounds, all caught; {ta.caught_bad} fallbacks whose FP64 result would have been wrong ({time.time() - t2:.1f} s)")
    for N in FOCUS:
        tl = adv[N]
        print(f"    N={N}: {tl.n} finishes, {tl.misround} misrounds, {tl.why['centre']} centring and {tl.why['range']} range fallbacks, {tl.kill} uncaught")
    print("ok  lemma 5 on every vector: misround <=> R outside [-P/2, P/2); every misround is in the delta_N band and caught by RD(P/2 - 2^-38 P) and by RN(P/2)")

    # 4b. the minimal threshold RN(P/2) is also sound (rerun the focus N with it)
    tm = Tally()
    for N in FOCUS:
        if N in models:
            for x, e in adv_cases[N]:
                run_case(models[N], x, e, tm, thr=models[N].half_rn)
    assert tm.kill == 0, tm.kills
    print(f"ok  with the minimal threshold RN(P/2): {tm.n} finishes at N = {FOCUS}, {tm.kill} uncaught mismatches, {tm.why['centre']} centring fallbacks")

    # 6. mutations: each must give an uncaught mismatch somewhere
    t3 = time.time()
    muts = {
        "RN instead of RO for v": dict(ro_mut=1),
        "RO nudge without the parity test": dict(ro_mut=2),
        "RO nudge away from the error": dict(ro_mut=3),
        "no centring clause": dict(thr=math.inf),
        "threshold one ulp above RN(P/2)": "thr+",
        "range test on the scaled product": dict(range_on_product=True),
    }
    for name, kw in muts.items():
        hits = 0
        for N in sorted(models):
            c = models[N]
            k = dict(thr=math.nextafter(c.half_rn, math.inf)) if kw == "thr+" else kw
            tl = Tally()
            for x, e in adv_cases[N]:
                try:
                    if run_case(c, x, e, tl, **k):
                        hits += 1
                except AssertionError:
                    hits += 1                                     # a lemma check fired: also a detection
            if hits:
                break
        assert hits > 0, f"mutation not detected: {name}"
        print(f"ok  mutation detected: {name} (first at N = {N})")
    print(f"    ({time.time() - t3:.1f} s)")

    # 7. fallback fractions
    print("fallback fraction per N (corpus, both e; adversarial):")
    for N in sorted(per_n):
        tl, ta_ = per_n[N], adv[N]
        fb = tl.why["centre"] + tl.why["range"]
        fa = ta_.why["centre"] + ta_.why["range"]
        print(f"    N={N:2d}  corpus {fb:4d}/{tl.n} = {fb / tl.n:.2e} (with RN(P/2): {tl.min_centre})   adversarial {fa}/{ta_.n} = {fa / ta_.n:.2e}")
    print("    uniformly distributed z: 2 * 2^-38 = 2^-37 = 7.3e-12 with the spec's threshold (RN(P/2): about 2^-53 relative); "
          "the corpus fraction is all from its deliberate boundary points x = H - 1, H, H + 1 (H = P/2 - 1), the adversarial one is by construction")
    print(f"PASS  crt_f64_ro: verdict PASS (0 uncaught mismatches)   {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
