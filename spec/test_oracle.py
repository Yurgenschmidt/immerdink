"""Tests for spec/oracle.py (DESIGN.md §3, matrix case).  Plain asserts, stdlib only.

Run ONLY via the quiet-machine wrapper:
    tools/run_quiet.sh python3 spec/test_oracle.py

What is checked (numbers refer to the oracle task list, sections to DESIGN.md):
  (0) primitives: correct rounding (incl. subnormal / overflow / double-rounding trap),
      exact fma vs math.fma, multiply vs hardware, quantization, residues (§3.4), shift
      search, constants (§2), CRT (§3.6) and the necessity of three design rules
      (K_c saturation, L_w word width, kappa slack of the RN shift rule).
  (1) z == exact A'B' for every emulated element of every run (§3.6 "z = (A'B') exactly"),
      plus the §3.2 guarantee ||a'||_2 <= T, sum|a'||b'| <= H.
  (2) §3.8 error bound, exact rational comparison, l1 form and 2 sqrt(k)/(T - kappa/2)
      normwise form (RN; TRUNC analogue derived here), phi x k x N x rounding sweep.
  (3) bitwise identical results across K-chunk lengths and residue-group sizes g (§1).
  (4) non-finite rows/cols: native path with IEEE semantics, other entries unaffected.
  (5) zero rows/cols, k = 0, alpha = 0, beta = 0 (C poisoned), huge and subnormal inputs.
  (6) RN vs TRUNC error statistics across phi and N (validates the §3.3 deviation).
  (7) smallest N reaching native-FP64-dot-like error at phi = 0, 1, 2, 4, k = 1000.
  (A) adversarial boundary cases that random data never reaches (added after mutation
      testing; each item names the mutant it kills):
        A1 §3.7 single rounding through contract(): subnormal first-rounding-tie traps
           (G8-style double rounding of D gives the even neighbour).
        A2 §3.2 ceil in S: k=2 rows at N=14..20 (TRUNC and RN) ON the shift boundary,
           where floor(S) would accept u+1 (TRUNC: ||a'|| > T and a wrapped CRT).
        A3 §3.2 kappa slack: N=2 RN rows whose entries all round up; with slack (2T-kappa+1)
           or kappa = floor(sqrt k) the guarantee ||a'|| <= T breaks (shown as negatives).
        A4 §3.2 an independent recomputation of sigma from the DESIGN text (Fraction
           arithmetic) for every row/column of every run (check_run), plus the corrected
           second-order bound 2^-sigma < 2 sqrt(||a||^2 + k 4^gamma 2^-F) / (T - kappa/2).
        A5 §3.6 word bound at its true boundary: N=16 with L_w=126 (P < 2^126 < 2P)
           overflows; L_w=127 (2P <= 2^127) never does.
        A6 §3.7 beta epilogue is ONE fma (constructed 2^-104 case + random triples).
        A7 §3.7 native path canonical k order ([MAX, MAX, -Inf].[2, 2, 1] = NaN).
        A8 §3.5 default chunk 131008 actually splits (k = 131009, and 131056 in the v1.13
           window) and add-mod is exact.
  NB residue-group size g cannot influence the oracle's result by construction (each plane
  is computed independently from the same quantized operands); g-independence of the real
  buffer schedule is the Pipeline spec's job. The g loop in (3) only checks plumbing.
"""

import math
import os
import random
import statistics
import struct
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle as o  # noqa: E402

U = Fraction(1, 2**53)          # unit roundoff of binary64
SIGMA_SEEN = [10**9, -10**9]    # observed sigma/tau range across all runs
RESULTS = {}


class Poison:
    """A matrix that must not be read: any element access raises."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def __getitem__(self, i):
        raise AssertionError("operand was read but must not be (§3.7 / BLAS read rules)")


def rnd_matrix(rng, r, c, phi):
    """Task's generator: (rand - 0.5) * exp(phi * randn)."""
    return [[(rng.random() - 0.5) * math.exp(phi * rng.gauss(0.0, 1.0)) for _ in range(c)]
            for _ in range(r)]


def fr(x):
    return Fraction(x)


def dy(md):
    return o.dyadic_to_fraction(*md)


def same_bits(x, y):
    if math.isnan(x) and math.isnan(y):
        return True
    return o.bits(x) == o.bits(y)


def note_sigma(vals):
    for s in vals:
        SIGMA_SEEN[0] = min(SIGMA_SEEN[0], s)
        SIGMA_SEEN[1] = max(SIGMA_SEEN[1], s)


# ---------------------------------------------------------------------------------------
# (0) primitives
# ---------------------------------------------------------------------------------------

def rand_double(rng, emin=-1080, emax=1030):
    """Random finite double with a wide exponent spread, incl. subnormals and extremes."""
    e = rng.randint(emin, emax)
    m = rng.getrandbits(53) | (1 << 52) if rng.random() < 0.9 else rng.getrandbits(rng.randint(1, 53))
    v = o.round_dyadic(m if rng.random() < 0.5 else -m, e - 52)
    if math.isinf(v):
        v = math.copysign(1.7976931348623157e308, v)
    return v


def test_primitives():
    rng = random.Random(1)
    # §2 constants
    assert o.MODULI == o.DESIGN_MODULI
    assert o.K_C == 131071 and o.K_C_ALIGNED == 131008 == 2047 * 64
    assert 128**2 * o.K_C <= 2**31 - 1 < 128**2 * (o.K_C + 1)
    for N in range(2, 21):
        pl = o.Plan(N, 1)
        assert 2 * pl.P <= 2**pl.L_w
        assert pl.T**2 <= pl.H and math.gcd(pl.P, 1) == 1
        for l in range(N):
            assert (pl.M[l] * pl.q[l]) % pl.p[l] == 1
    assert o.Plan(16, 1).L_w == 128 and o.Plan(17, 1).L_w == 160
    assert 2 * o.Plan(17, 1).P > 2**128            # so 160-bit words are needed at N = 17

    # correct rounding of rationals: compare with CPython's correctly rounded int/int
    for _ in range(4000):
        e = rng.randint(-1200, 1100)
        m = rng.getrandbits(rng.randint(1, 120)) * rng.choice((1, -1))
        mine = o.round_dyadic(m, e)
        f = o.dyadic_to_fraction(m, e)
        try:
            ref = m / 2**-e if e < 0 else float(m * 2**e)
            ref2 = float(f)
        except OverflowError:
            ref = ref2 = None
        if ref is None:
            # Python refuses; check IEEE overflow threshold exactly: |f| >= 2^1024 - 2^970
            assert math.isinf(mine) and abs(f) >= 2**1024 - 2**970, (m, e)
        else:
            assert same_bits(mine, ref) and same_bits(mine, ref2), (m, e, mine, ref)
        assert same_bits(o.round_fraction(f), mine)
    # rationals with non-dyadic denominators
    for _ in range(2000):
        f = Fraction(rng.getrandbits(80) - 2**79, rng.getrandbits(70) + 1) * Fraction(2) ** rng.randint(-1150, 1000)
        try:
            ref = float(f)
        except OverflowError:
            ref = math.copysign(math.inf, f)
        assert same_bits(o.round_fraction(f), ref), f
    MAXD = 1.7976931348623157e308
    assert o.round_dyadic(2**53 - 1, 971) == MAXD
    assert o.round_dyadic(2**54 - 2, 970) == MAXD                  # exactly max
    assert o.round_dyadic(2**54 - 1, 970) == math.inf              # max + half ulp: tie->even = 2^1024
    assert o.round_dyadic(2**55 - 3, 969) == MAXD                  # max + quarter ulp
    assert o.round_dyadic(-(2**54 - 1), 970) == -math.inf
    assert o.round_dyadic(1, -1074) == 5e-324
    assert o.round_dyadic(1, -1075) == 0.0                         # half of min subnormal: tie->0
    assert o.round_dyadic(3, -1076) == 5e-324                      # 0.75 -> 1
    assert o.round_dyadic(3, -1075) == 1e-323                      # 1.5 -> 2 (even)
    assert o.signbit(o.round_dyadic(-1, -1080))                    # underflow keeps sign
    # double-rounding trap: 2^-1075 + 2^-1134 must round UP to 2^-1074 (single rounding);
    # rounding to 53 bits first gives 2^-1075, a tie, then even -> 0.
    assert o.round_dyadic(2**59 + 1, -1134) == 5e-324
    # largest subnormal / smallest normal boundary
    assert o.round_dyadic(2**53 - 1, -1075) == 2.2250738585072014e-308

    # exact fma vs math.fma (finite random, incl. subnormal products and cancellation)
    specials = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 5e-324, -5e-324, MAXD, -MAXD]
    nf = 0
    for _ in range(20000):
        a = rand_double(rng) if rng.random() < 0.9 else rng.choice(specials)
        b = rand_double(rng) if rng.random() < 0.9 else rng.choice(specials)
        if rng.random() < 0.3 and math.isfinite(a) and math.isfinite(b):
            c = -(a * b) if math.isfinite(a * b) else rand_double(rng)   # cancellation
            if rng.random() < 0.5 and math.isfinite(c):
                c = math.nextafter(c, rng.choice((math.inf, -math.inf)))
        else:
            c = rand_double(rng) if rng.random() < 0.9 else rng.choice(specials)
        mine = o.fma(a, b, c)
        try:
            ref = math.fma(a, b, c)
        except ValueError:      # CPython raises on invalid (Inf*0 etc.) instead of NaN
            ref = math.nan
        except OverflowError:
            ref = math.inf if fr(a) * fr(b) + fr(c) > 0 else -math.inf
        assert same_bits(mine, ref), (a, b, c, mine, ref)
        assert same_bits(o.mul(a, b), a * b), (a, b)
        nf += 1
    RESULTS["fma_checked"] = nf

    # quantization (§3.3) vs exact rational rounding; residues (§3.4) vs exact SMod
    for _ in range(5000):
        v = rand_double(rng, -300, 300)
        s = rng.randint(-200, 200)
        x = fr(v) * Fraction(2) ** s
        for mode in ("RN", "TRUNC"):
            mu, eps = o.quantize(v, s, mode)
            a = mu << eps
            if mode == "TRUNC":
                assert a == math.trunc(x)
            else:
                fl = math.floor(x)
                d = x - fl
                exp = fl + (1 if d > Fraction(1, 2) or (d == Fraction(1, 2) and fl % 2) else 0)
                assert a == exp, (v, s, a, exp)
            for p in o.MODULI:
                r = o.residue(mu, eps, p)
                assert r == o.smod(a, p) and (r - a) % p == 0
                assert -(p // 2) <= r <= (p + 1) // 2 - 1

    # largest_u brute force
    for _ in range(3000):
        L = rng.getrandbits(rng.randint(1, 140)) + 1
        R = rng.getrandbits(rng.randint(1, 140)) + 1
        u = o.largest_u(L, R)
        ok = lambda t: Fraction(L) * Fraction(4) ** t <= R  # noqa: E731
        assert ok(u) and not ok(u + 1)

    # CRT (§3.6): random residues reconstruct the unique centered integer
    for N in (2, 5, 9, 15, 16, 17, 20):
        pl = o.Plan(N, 1)
        for _ in range(300):
            z = rng.randint(-pl.H, pl.H) if rng.random() < 0.8 else rng.choice((-pl.H, pl.H, 0, -1, 1))
            W = [z % p for p in pl.p]
            assert o.crt(W, pl) == z

    # Negative 1 (§2 L_w): 128-bit words at N = 17 overflow for some inputs
    pl = o.Plan(17, 1, L_w=128)
    hit = 0
    for _ in range(200):
        try:
            o.crt([rng.randrange(p) for p in pl.p], pl)
        except o.WordOverflow:
            hit += 1
    assert hit > 0, "expected the 128-bit CRT word to overflow at N=17"
    # ... and never at N = 16 (2P < 2^128): peak bound
    pl16 = o.Plan(16, 1)
    assert 2 * pl16.P <= 2**128
    RESULTS["crt_L128_N17_overflows"] = f"{hit}/200 random W"

    # Negative 2 (§2 K_c, §3.5): all residues -128 (p = 256) over 131072 terms saturates int32
    ra = [-128] * 131072
    try:
        o._chunk_dot(ra, ra)
        raise AssertionError("expected saturation")
    except o.Saturation:
        pass
    assert o._chunk_dot(ra[:o.K_C], ra[:o.K_C]) == 128**2 * o.K_C      # K_c itself is fine
    try:
        o.Plan(8, 10, chunk=o.K_C + 1)
        raise AssertionError("chunk > K_c must be rejected")
    except o.InvalidValue:
        pass

    # Negative 3 (§3.2 kappa slack): RN rounding with the TRUNC shift rule breaks uniqueness.
    # N = 2 (T = 180, H = 32639), k = 49, every entry 1.59375 = 25.5/16: the TRUNC rule picks
    # u = 4 (49 * 25.5^2 = 31862 <= T^2), RN rounds 25.5 -> 26 and a'.b' = 49*676 = 33124 > H.
    A = [[1.59375] * 49]
    B = [[1.59375] for _ in range(49)]
    t = {}
    o.contract(A, B, None, 1.0, 0.0, 2, "RN", trace=t, _rule="TRUNC")
    exact = sum(qa * qb for qa, qb in zip(map(o.qvalue, t["QA"][0]), map(o.qvalue, t["QB"][0])))
    assert exact == 33124 > t["plan"].H and t["z"][0][0] == exact - t["plan"].P != exact
    t = {}
    o.contract(A, B, None, 1.0, 0.0, 2, "RN", trace=t)
    exact = sum(qa * qb for qa, qb in zip(map(o.qvalue, t["QA"][0]), map(o.qvalue, t["QB"][0])))
    assert t["z"][0][0] == exact
    RESULTS["neg_rn_without_kappa"] = "z = -32156 (wrapped) vs exact 33124 > H = 32639"

    # §3.2 plan-time rejection boundary at N = 2 (T = 180): k = 179^2 ok, 180^2 rejected
    assert o.Plan(2, 179**2, "RN").kappa == 179
    try:
        o.Plan(2, 179**2 + 1, "RN")
        raise AssertionError("kappa = T must be rejected")
    except o.NotSupported:
        pass
    o.Plan(2, 180**2, "TRUNC")              # TRUNC has no kappa slack, never rejected
    # run the boundary case for real: k = 32041, kappa = 179 = T - 1.  Row 0 / column 0 hold
    # 0.5 + 2^-10 everywhere: the RN rule gives 2^u <= 181 / (2 * 179 * 1.00195) -> u = -1,
    # so every scaled entry is 0.50098 and ROUNDS UP to 1: ||a'|| = 179 (T = 180) and
    # a'.b' = 32041 (H = 32639), i.e. the quantized norm really approaches T (not all-zero).
    k = 179**2
    v = 0.5 + 2.0**-10
    A = [[v] * k, [rng.random() - 0.5 for _ in range(k)]]
    B = [[v, rng.random() - 0.5] for _ in range(k)]
    t = {}
    C = o.contract(A, B, None, 1.0, 0.0, 2, "RN", trace=t)
    check_run(A, B, C, t, k)
    nq = sum(o.qvalue(q) ** 2 for q in t["QA"][0])
    assert nq == 179**2 and t["z"][0][0] == 179**2 and nq <= t["plan"].T ** 2
    RESULTS["rn_boundary_N2_k32041"] = (f"sigma={t['sigma']}, tau={t['tau']}, ||a'_0||^2 = {nq} "
                                        f"(T^2 = {t['plan'].T ** 2}), z[0][0] = {t['z'][0][0]} (H = {t['plan'].H})")


def test_params_crosscheck():
    """Cross-check against the independent implementation in spec/params.py.  Skipped ONLY
    when spec/params.py does not exist; if it exists but fails to import (or lacks the
    expected API) the test FAILS, so the suite cannot silently pass at reduced strength."""
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(here, "params.py")):
        RESULTS["params_crosscheck"] = "skipped (spec/params.py absent)"
        return
    import params as pm                                 # must import when present
    consts, scale_row, pquant = pm.constants, pm.scale_row, pm.quantize
    for N in range(2, 21):
        c = consts(N)
        pl = o.Plan(N, 1)
        assert (c["P"], c["H"], c["T"], list(c["M"]), list(c["q"])) == (pl.P, pl.H, pl.T, pl.M, pl.q), N
        assert c["Lw"] <= pl.L_w
    # §4 k_hat = ceil(k)_A: the oracle's rounding and params.k_hat_of (libimmerdink_host's
    # mirror) agree, and so do the chunk lengths (§3.5 K_c')
    for A in (16, 32, 64):
        for k in list(range(0, 300)) + [131007, 131008, 131009, 131056, 131057, 131072, 2**40 + 3]:
            assert o.k_hat_of(k, A) == pm.k_hat_of(k, A), (k, A)
    assert o.k_hat_of(5) == pm.k_hat_of(5) == 16 and pm.K_C_PRIME == o.K_C_ALIGNED
    rng = random.Random(11)
    n = 0
    for _ in range(400):
        k = rng.choice((1, 2, 3, 17, 100))
        phi = rng.choice((0, 1, 4, 8))
        row = [x * 2.0 ** rng.randint(-1000, 1000) if rng.random() < 0.05 else x
               for x in rnd_matrix(rng, 1, k, phi)[0]]
        if rng.random() < 0.2:
            row[0] = 5e-324 * rng.randint(1, 1000)
        for N in (2, 8, 14, 20):
            for mode in ("RN", "TRUNC"):
                try:
                    pl = o.Plan(N, k, mode)
                except o.NotSupported:
                    continue
                sig = o.scale_vector(row, pl)[0]
                psig = scale_row(row, N, mode)[0]
                assert sig == psig, (row, N, mode, sig, psig)
                for v in row:
                    mu, eps = o.quantize(v, sig, mode)
                    assert pquant(v, sig, mode)[0] == mu << eps
                n += 1
    RESULTS["params_crosscheck"] = f"constants N=2..20 and {n} shift/quantize cases agree"


# ---------------------------------------------------------------------------------------
# (1)+(2) exactness and error bound
# ---------------------------------------------------------------------------------------

BOUND_STATS = {"l1_max_ratio": 0.0, "norm_max_ratio": 0.0, "twopow_max_ratio": 0.0,
               "checked": 0, "twopow_rows": 0, "twopow_uncorrected_fail": 0}


def frac_sqrt(f):
    """sqrt of a nonnegative Fraction to ~60 significant bits (for reporting only)."""
    if f == 0:
        return Fraction(0)
    h = (f.numerator.bit_length() - f.denominator.bit_length()) // 2 - 60
    return Fraction(math.isqrt(math.floor(f / Fraction(4) ** h))) * Fraction(2) ** h


def ref_sigma(vec, N, k, mode):
    """A4: sigma recomputed straight from the DESIGN §3.2 text, independently of the oracle:
    constants from the §2 table, gamma via frexp, S via Fraction ceilings, u by a linear walk
    over the defining inequality.  Returns (sigma, gamma); gamma None for an all-zero row."""
    vals = [v for v in vec if math.isfinite(v) and v != 0.0]       # §3.1: non-finite -> 0
    if not vals:
        return 0, None                                              # step 1: sigma := 0
    P = math.prod(o.DESIGN_MODULI[:N])
    T = math.isqrt((P - 1) // 2)
    kap = math.isqrt(k - 1) + 1                                     # ceil(sqrt k), k >= 1
    g = max(math.frexp(v)[1] - 1 for v in vals)                     # ilogb(max |a_h|)
    S = sum(math.ceil(Fraction(v) ** 2 * Fraction(2) ** (60 - 2 * g)) for v in vals)
    lhs, rhs = (4 * S, (2 * T - kap) ** 2 * 2**60) if mode == "RN" else (S, T * T * 2**60)

    def ok(u):
        return lhs * Fraction(4) ** u <= rhs
    u = 0
    while not ok(u):
        u -= 1
    while ok(u + 1):
        u += 1
    return u - g, g


def check_shift(vec, s, plan, k):
    """A4 for one row/column: sigma equals the independent recomputation, and the corrected
    second-order claim (DESIGN §3.8 with the (1 + k 2^-61)-type term, see design issues):
        2^-sigma < 2 sqrt(||a||^2 + k 4^gamma 2^-F) / den,  den = T - kappa/2 (RN), T (TRUNC).
    Returns 2^-sigma / (2 ||a|| / den) for reporting (DESIGN's uncorrected form)."""
    rs, g = ref_sigma(vec, plan.N, k, plan.rounding)
    assert s == rs, ("sigma differs from independent §3.2 recomputation", s, rs, plan.N, plan.rounding)
    if g is None:
        return None
    n2 = sum(fr(x) ** 2 for x in vec if math.isfinite(x))
    den = Fraction(2 * plan.T - plan.kappa, 2) if plan.rounding == "RN" else Fraction(plan.T)
    lhs = Fraction(4) ** (-s) * den * den
    assert lhs < 4 * (n2 + k * Fraction(4) ** g / 2**o.F), "maximality of u (second-order bound)"
    BOUND_STATS["twopow_rows"] += 1
    if lhs > 4 * n2:                      # DESIGN's uncorrected form 2^-sigma <= 2||a||/den fails
        BOUND_STATS["twopow_uncorrected_fail"] += 1
    return math.sqrt(lhs / (4 * n2))


def check_run(A, B, C, t, k):
    """(1) and (2) for one run: exactness of z, §3.2 guarantee, §3.8 bound (exact)."""
    plan = t["plan"]
    m, n = len(A), len(B[0])
    Bt = [[B[h][j] for h in range(k)] for j in range(n)]
    note_sigma(t["sigma"] + t["tau"])
    for vec, s in list(zip(A, t["sigma"])) + list(zip(Bt, t["tau"])):
        r = check_shift(vec, s, plan, k)
        if r is not None:
            BOUND_STATS["twopow_max_ratio"] = max(BOUND_STATS["twopow_max_ratio"], r)
    QAv = [[o.qvalue(q) for q in row] for row in t["QA"]]
    QBv = [[o.qvalue(q) for q in col] for col in t["QB"]]
    # §3.2 guarantee
    for vec in QAv + QBv:
        assert sum(x * x for x in vec) <= plan.T**2, "||a'||_2 <= T violated"
    delta = Fraction(1, 2) if plan.rounding == "RN" else Fraction(1)
    kap = plan.kappa
    nA2 = [sum(fr(x) ** 2 for x in row) for row in A]
    nB2 = [sum(fr(x) ** 2 for x in col) for col in Bt]
    nA1 = [sum(abs(fr(x)) for x in row) for row in A]
    nB1 = [sum(abs(fr(x)) for x in col) for col in Bt]
    for i in range(m):
        for j in range(n):
            if t["fA"][i] or t["fB"][j]:
                continue
            z = t["z"][i][j]
            # (1) z == exact A'B'
            ex_q = sum(x * y for x, y in zip(QAv[i], QBv[j]))
            assert z == ex_q, (i, j, z, ex_q)
            assert sum(abs(x * y) for x, y in zip(QAv[i], QBv[j])) <= plan.H
            D = t["D"][i][j]
            assert D == C[i][j] or (D != D and C[i][j] != C[i][j])
            s, tt = t["sigma"][i], t["tau"][j]
            ex = dy(o.exact_dot(A[i], Bt[j]))
            emul = Fraction(z) * Fraction(2) ** (-(s + tt))
            # the single rounding of §3.7 (also in the subnormal / overflow range)
            assert same_bits(D, o.round_fraction(emul))
            p2s, p2t = Fraction(2) ** (-s), Fraction(2) ** (-tt)
            # (2a) l1 form: |emul - ex| <= delta (2^-t ||a||_1 + 2^-s ||b||_1) + delta^2 k 2^-s-t
            l1 = delta * (p2t * nA1[i] + p2s * nB1[j]) + delta**2 * k * p2s * p2t
            assert abs(emul - ex) <= l1, ("l1", i, j)
            if l1:
                BOUND_STATS["l1_max_ratio"] = max(BOUND_STATS["l1_max_ratio"], float(abs(emul - ex) / l1))
            if math.isinf(D) or (z != 0 and abs(D) < 2.2250738585072014e-308):
                continue                                  # §3.8 excludes over/underflow of D
            err = abs(fr(D) - ex)
            uD = U * abs(fr(D))
            assert err <= l1 + uD
            # (2b) normwise form, exactly as DESIGN §3.8 (RN) / TRUNC analogue:
            #   RN:    err <= 2 sqrt(k) ||a|| ||b|| / (T - kappa/2) + k 2^(-s-t-2) + u|D|
            #   TRUNC: err <= 4 sqrt(k) ||a|| ||b|| / T            + k 2^(-s-t)   + u|D|
            if plan.rounding == "RN":
                c2 = Fraction(16 * k, (2 * plan.T - kap) ** 2)
                rest = k * p2s * p2t / 4 + uD
            else:
                c2 = Fraction(16 * k, plan.T**2)
                rest = k * p2s * p2t + uD
            X = err - rest
            prod2 = c2 * nA2[i] * nB2[j]
            assert X <= 0 or X * X <= prod2, ("normwise", i, j, plan.N, plan.rounding)
            bnd = frac_sqrt(prod2) + rest           # ~60-bit sqrt, exact otherwise
            if bnd:
                BOUND_STATS["norm_max_ratio"] = max(BOUND_STATS["norm_max_ratio"], float(err / bnd))
            BOUND_STATS["checked"] += 1


def test_sweep():
    rng = random.Random(2)
    PHIS = (0, 0.5, 1, 2, 4, 8)
    KS = (1, 2, 3, 17, 100, 1000)
    runs = skipped = 0
    for phi in PHIS:
        for k in KS:
            m, n = (3, 3) if k <= 100 else (2, 2)
            A = rnd_matrix(rng, m, k, phi)
            B = rnd_matrix(rng, k, n, phi)
            for N in range(2, 21):
                for mode in ("RN", "TRUNC"):
                    try:
                        o.Plan(N, k, mode)
                    except o.NotSupported:
                        skipped += 1
                        continue
                    t = {}
                    C = o.contract(A, B, None, 1.0, 0.0, N, mode, trace=t)
                    check_run(A, B, C, t, k)
                    runs += 1
    RESULTS["sweep_runs"] = f"{runs} runs ({skipped} RN plans rejected), {BOUND_STATS['checked']} elements bound-checked"


# ---------------------------------------------------------------------------------------
# (3) determinism across chunk lengths and residue groups
# ---------------------------------------------------------------------------------------

def test_determinism():
    rng = random.Random(3)
    cnt = 0
    for (N, k, phi) in ((3, 100, 2), (9, 100, 4), (17, 70, 1), (20, 33, 8)):
        A = rnd_matrix(rng, 3, k, phi)
        B = rnd_matrix(rng, k, 3, phi)
        C0 = rnd_matrix(rng, 3, 3, 1)
        for mode in ("RN", "TRUNC"):
            ref = None
            for chunk in (1, 7, 16, k, o.K_C_ALIGNED):
                for g in sorted({1, 2, 3, (N + 1) // 2, N}):
                    t = {}
                    out = o.contract(A, B, C0, -1.25, 0.5, N, mode, g=g, chunk=chunk, trace=t)
                    sig = [o.bits(x) for row in out for x in row]
                    planes = t["W"]
                    if ref is None:
                        ref, refW = sig, planes
                    assert sig == ref and planes == refW, (N, mode, chunk, g)
                    cnt += 1
    RESULTS["determinism_runs"] = cnt


# ---------------------------------------------------------------------------------------
# (4) non-finite rows / columns
# ---------------------------------------------------------------------------------------

def ref_native(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        try:
            acc = math.fma(x, y, acc)
        except ValueError:
            acc = math.nan
        except OverflowError:
            acc = math.inf if fr(x) * fr(y) + fr(acc) > 0 else -math.inf
    return acc


def test_nonfinite():
    rng = random.Random(4)
    k, m, n = 6, 5, 4
    inf, nan = math.inf, math.nan
    A = rnd_matrix(rng, m, k, 1)
    B = rnd_matrix(rng, k, n, 1)
    A[1][0] = inf                    # row 1: +Inf at h=0
    A[3][2] = nan                    # row 3: NaN
    A[4][1], A[4][3] = inf, -inf     # row 4: Inf - Inf
    B[0][2] = 0.0                    # col 2 has 0 at h=0 -> Inf*0 = NaN in (1,2)
    B[5][3] = -inf                   # col 3 flagged
    for j in range(3):               # row 4 meets +Inf*|b| and -Inf*|b|: Inf - Inf = NaN
        B[1][j] = abs(B[1][j]) + 0.5
        B[3][j] = abs(B[3][j]) + 0.5
    t = {}
    C = o.contract(A, B, None, 1.0, 0.0, 9, "RN", trace=t)
    assert t["fA"] == [False, True, False, True, True] and t["fB"] == [False, False, False, True]
    Bt = [[B[h][j] for h in range(k)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            if t["fA"][i] or t["fB"][j]:
                assert same_bits(C[i][j], ref_native(A[i], Bt[j])), (i, j)
    assert math.isnan(C[1][2])                       # Inf * 0
    assert C[1][0] == math.copysign(inf, B[0][0])    # Inf * finite
    assert all(math.isnan(C[3][j]) for j in range(n))
    assert all(math.isnan(C[4][j]) for j in range(3))   # Inf + (-Inf)
    assert math.isinf(C[0][3]) or math.isnan(C[0][3])
    # flagged rows: sigma computed from the finite entries only (non-finite treated as 0)
    A1f = [0.0 if not math.isfinite(v) else v for v in A[1]]
    assert t["sigma"][1] == o.scale_vector(A1f, t["plan"])[0]
    # other entries unaffected: replace flagged rows/cols by finite junk, compare bits
    A2 = [row[:] for row in A]
    B2 = [row[:] for row in B]
    for i in (1, 3, 4):
        A2[i] = [rng.random() * 1e30 for _ in range(k)]
    for h in range(k):
        B2[h][3] = rng.random() * 1e-30
    C2 = o.contract(A2, B2, None, 1.0, 0.0, 9, "RN")
    for i in (0, 2):
        for j in (0, 1, 2):
            assert same_bits(C[i][j], C2[i][j])
    # flagged + alpha/beta
    Cin = rnd_matrix(rng, m, n, 0)
    C3 = o.contract(A, B, Cin, 2.0, -1.0, 9, "RN")
    assert math.isnan(C3[1][2]) and math.isnan(C3[3][0])
    RESULTS["nonfinite"] = "ok"


# ---------------------------------------------------------------------------------------
# (5) edge cases
# ---------------------------------------------------------------------------------------

def test_edges():
    rng = random.Random(5)
    nan = math.nan
    # zero rows/cols
    A = rnd_matrix(rng, 3, 5, 1)
    B = rnd_matrix(rng, 5, 3, 1)
    A[1] = [0.0] * 5
    A[2][0] = -0.0
    for h in range(5):
        B[h][2] = 0.0
    Cin = rnd_matrix(rng, 3, 3, 1)
    t = {}
    C = o.contract(A, B, Cin, 3.0, 0.25, 8, "RN", trace=t)
    assert t["sigma"][1] == 0 and t["tau"][2] == 0
    for i in range(3):
        for j in range(3):
            if i == 1 or j == 2:
                assert t["z"][i][j] == 0 and same_bits(t["D"][i][j], 0.0)
                assert same_bits(C[i][j], math.fma(0.25, Cin[i][j], 0.0))
    # k = 0: C <- beta C, A and B not read
    Cin = rnd_matrix(rng, 2, 3, 1)
    C = o.contract(Poison(2), Poison(0), Cin, 1.0, -0.5, 8, m=2, n=3, k=0)
    assert all(same_bits(C[i][j], -0.5 * Cin[i][j]) for i in range(2) for j in range(3))
    # k = 0 and beta = 0: C not read either -> +0
    C = o.contract(Poison(2), Poison(0), Poison(2), 1.0, 0.0, 8, m=2, n=3, k=0)
    assert all(same_bits(x, 0.0) for row in C for x in row)
    # alpha = 0: A, B not read (NaN-free result even though they would contain NaN)
    C = o.contract(Poison(2), Poison(4), Cin, 0.0, 2.0, 8, m=2, n=3, k=4)
    assert all(same_bits(C[i][j], 2.0 * Cin[i][j]) for i in range(2) for j in range(3))
    # beta = 0: C not read (poisoned) and NaN in C would not propagate
    A = rnd_matrix(rng, 2, 4, 1)
    B = rnd_matrix(rng, 4, 3, 1)
    C1 = o.contract(A, B, Poison(2), 1.0, 0.0, 10, m=2, n=3, k=4)
    C2 = o.contract(A, B, [[nan] * 3] * 2, 1.0, 0.0, 10)
    assert all(not math.isnan(x) for row in C2 for x in row) and C1 == C2
    # alpha/beta rules: E = alpha==1 ? D : RN(alpha D); C = fma(beta, C, E)
    t = {}
    Cin = rnd_matrix(rng, 2, 3, 1)
    C = o.contract(A, B, Cin, -2.7, 0.3, 10, trace=t)
    for i in range(2):
        for j in range(3):
            E = -2.7 * t["D"][i][j]
            assert same_bits(C[i][j], math.fma(0.3, Cin[i][j], E))
    # m = 0 / n = 0: no-op
    assert o.contract([], [[1.0]], [], 1.0, 1.0, 8, m=0, n=1, k=1) == []
    # empty plan errors
    for bad in ((1, "RN", None), (21, "RN", None), (8, "RZ", None), (8, "RN", 9)):
        try:
            o.Plan(bad[0], 4, bad[1], bad[2])
            raise AssertionError(bad)
        except o.InvalidValue:
            pass

    # huge and subnormal inputs: no overflow in scaling, correct results
    k = 8
    cases = {
        "huge*tiny (2^1000 x 2^-1000)": (1000, -1000),
        "huge*huge -> Inf (2^600 x 2^600)": (600, 600),
        "tiny*huge (2^-1000 x 2^1000)": (-1000, 1000),
        "subnormal*huge (2^-1060 x 2^1020)": (-1060, 1020),
        "subnormal result (2^-540 x 2^-520)": (-540, -520),
        "deep underflow (2^-1000 x 2^-1000)": (-1000, -1000),
        "near-max (2^1015 x 2^3)": (1015, 3),
    }
    for name, (ea, eb) in cases.items():
        A = [[math.ldexp(rng.random() + 0.5, ea) * rng.choice((1, -1)) for _ in range(k)] for _ in range(3)]
        B = [[math.ldexp(rng.random() + 0.5, eb) * rng.choice((1, -1)) for _ in range(3)] for _ in range(k)]
        A[0][0] = math.ldexp(1.0, ea - 400) if ea - 400 > -1074 else 5e-324   # mixed range row
        if ea <= -1060:
            A[1] = [5e-324 * rng.randint(1, 2**20) for _ in range(k)]           # subnormals
        for mode in ("RN", "TRUNC"):
            t = {}
            C = o.contract(A, B, None, 1.0, 0.0, 16, mode, trace=t)
            check_run(A, B, C, t, k)
            note_sigma(t["sigma"] + t["tau"])
            for i in range(3):
                for j in range(3):
                    s, tt = t["sigma"][i], t["tau"][j]
                    exact_e = Fraction(t["z"][i][j]) * Fraction(2) ** (-(s + tt))
                    assert same_bits(C[i][j], o.round_fraction(exact_e))
                    ex = dy(o.exact_dot(A[i], [B[h][j] for h in range(k)]))
                    if "Inf" in name:
                        assert math.isinf(C[i][j]) and (C[i][j] > 0) == (ex > 0)
                    elif "deep underflow" in name:
                        assert C[i][j] == 0.0
                    elif "subnormal result" in name:
                        D = C[i][j]
                        assert 0 < abs(D) < 2.2250738585072014e-308
                        # emulation error is tiny relative; D within half a subnormal ulp + emul err
                        assert abs(fr(D) - ex) <= Fraction(1, 2**1075) + abs(exact_e - ex)
                        assert abs(exact_e - ex) <= abs(ex) * Fraction(1, 2**40)
                    else:
                        assert abs(fr(C[i][j]) - ex) <= abs(ex) * Fraction(1, 2**40), name
    RESULTS["edges"] = "ok"


# ---------------------------------------------------------------------------------------
# (A) adversarial boundary cases
# ---------------------------------------------------------------------------------------

def run_checked(A, B, N, mode, C=None, alpha=1.0, beta=0.0, **kw):
    t = {}
    out = o.contract(A, B, C, alpha, beta, N, mode, trace=t, **kw)
    if beta == 0.0 and alpha == 1.0:
        check_run(A, B, out, t, len(B))
    return out, t


def test_adversarial():
    rng = random.Random(8)
    res = {}

    # A1 §3.7 one rounding, through contract().  a = b = 2^-512 (1 + 2^-(52-j)) * 2^(-j|0):
    # a b = 2^E (1 + 2^-(51-j) + 2^-(104-2j)), E = -1024-j, subnormal with lsb 2^-1074, i.e.
    # 50-j fraction bits.  2^-(51-j) is exactly half an ulp and 2^-(104-2j) a sticky bit, so
    # the single rounding gives 2^E (1 + 2^-(50-j)); rounding z first to 53 bits drops the
    # sticky bit and the tie then goes to the even neighbour 2^E (GEMMul8-style, §3.7).
    # k = 1, N >= 14: both a', b' are exact (T >= 2^55), so z = a'b' exactly.
    n1 = 0
    for j in range(0, 26):
        a = math.ldexp(1 + 2.0 ** -(52 - j), -512)
        b = math.ldexp(1 + 2.0 ** -(52 - j), -512 - j)
        want = math.ldexp(1 + 2.0 ** -(50 - j), -1024 - j)
        wrong = math.ldexp(1.0, -1024 - j)
        assert want != wrong and 0 < want < 2.2250738585072014e-308
        for sa in (1.0, -1.0):
            for N in (14, 16, 20):
                for mode in ("RN", "TRUNC"):
                    D = run_checked([[sa * a]], [[b]], N, mode)[0][0][0]
                    assert o.bits(D) == o.bits(sa * want), (j, sa, N, mode, D, want)
                    n1 += 1
    res["A1 subnormal tie traps via contract (single rounding)"] = n1

    # A2 §3.2 ceil(x^2 2^F): k = 2 rows ON the shift boundary at real width.
    # Budget B = T (TRUNC) or T - 1 = (2T - kappa)/2 (RN, kappa = 2).  U = ilogb(B) >= 53,
    # w1 = top 53 bits of B, w2 = isqrt(B^2 - w1^2) + 1, so ||w||^2 = B^2 + tiny > B^2, and
    # a = w 2^-U (a1 in [1, 2), gamma = 0).  The ceiling rule must reject u = U and pick
    # U - 1; floor(S) accepts u = U (asserted below), which in TRUNC quantizes to ||a'|| =
    # ||w|| > T and wraps the CRT.  B column = A row, so z = ||a'||^2 is the extreme product.
    n2 = 0
    for N in range(14, 21):
        for mode in ("TRUNC", "RN"):
            pl = o.Plan(N, 2, mode)
            Bud = pl.T if mode == "TRUNC" else (2 * pl.T - pl.kappa) // 2
            assert mode == "TRUNC" or 2 * Bud == 2 * pl.T - pl.kappa
            U = Bud.bit_length() - 1
            assert U >= 53
            w1 = (Bud >> (U - 52)) << (U - 52)
            w2 = math.isqrt(Bud * Bud - w1 * w1) + 1
            assert 0 < w2 < 2**53 and w1 * w1 + w2 * w2 > Bud * Bud
            a1, a2 = math.ldexp(w1 >> (U - 52), -52), math.ldexp(w2, -U)
            assert fr(a1) * 2**U == w1 and fr(a2) * 2**U == w2
            # boundary witness: floor(S) would accept u = U, ceil(S) does not
            mul4 = 4 if mode == "RN" else 1
            rhs = (2 * pl.T - pl.kappa) ** 2 if mode == "RN" else pl.T ** 2
            Sfloor = sum(math.floor(fr(x) ** 2 * 2**o.F) for x in (a1, a2))
            Sceil = sum(math.ceil(fr(x) ** 2 * 2**o.F) for x in (a1, a2))
            assert mul4 * Sfloor * 4**U <= rhs * 2**o.F < mul4 * Sceil * 4**U
            out, t = run_checked([[a1, a2]], [[a1], [a2]], N, mode)
            assert t["sigma"][0] == U - 1, (N, mode, t["sigma"][0], U)
            n2 += 1
    res["A2 ceil(S) boundary rows N=14..20 x {TRUNC,RN}"] = n2

    # A3 §3.2 kappa slack, N = 2 (T = 180): rows of k < 60 entries c_h + 1/2 + 2^-30 whose
    # entries ALL round up; chosen so that sum (c_h + 1)^2 > T^2 while the scaled-up norm
    # would be accepted by a slack that is one too generous ((2T - kappa + 1)^2) or by
    # kappa = floor(sqrt k).  Positive: the correct rule keeps ||a'|| <= T and z exact.
    # Negatives: the same rows under the mutated right-hand sides break ||a'|| <= T.
    cands = []
    T2 = 180
    for k in range(2, 60):
        kap = o.ceil_sqrt(k)
        for c in range(1, 200):
            for j in range(k + 1):
                nn = [c + 1] * j + [c] * (k - j)
                s4 = sum((2 * x + 1) ** 2 for x in nn)
                if sum((x + 1) ** 2 for x in nn) > T2 * T2 and \
                        (s4 <= (2 * T2 - math.isqrt(k)) ** 2 or s4 <= (2 * T2 - kap + 1) ** 2):
                    cands.append(nn)
    assert len(cands) >= 40
    neg = {"slack+1": 0, "floor-kappa": 0}
    for nn in cands:
        row = [x + 0.5 + 2.0**-30 for x in nn]
        k = len(row)
        run_checked([row], [[v] for v in row], 2, "RN")
        for name in neg:
            pl = o.Plan(2, k, "RN")
            kp = pl.kappa if name == "slack+1" else math.isqrt(k)
            pl.shift_rhs = ((2 * pl.T - kp + (1 if name == "slack+1" else 0)) ** 2) << o.F
            s = o.scale_vector(row, pl)[0]
            q = [o.qvalue(o.quantize(v, s, "RN")) for v in row]
            if sum(x * x for x in q) > pl.T ** 2:
                neg[name] += 1
    assert neg["slack+1"] > 0 and neg["floor-kappa"] > 0, neg
    res["A3 RN all-round-up rows (N=2)"] = (f"{len(cands)} positives; guarantee broken by "
                                            f"slack (2T-kappa+1)^2 on {neg['slack+1']}, "
                                            f"by kappa=floor(sqrt k) on {neg['floor-kappa']}")

    # A5 §3.6 the word must hold the PEAK x + M_l y < 2P, not merely x < P.
    pl = o.Plan(16, 1, L_w=126)
    assert pl.P < 2**126 < 2 * pl.P
    hit = 0
    for _ in range(300):
        try:
            o.crt([rng.randrange(p) for p in pl.p], pl)
        except o.WordOverflow:
            hit += 1
    assert hit > 0, "L_w = 126 at N = 16 (P < 2^126 < 2P) must overflow for some W"
    pl = o.Plan(16, 1, L_w=127)
    assert 2 * pl.P <= 2**127
    for _ in range(300):
        z = rng.randint(-pl.H, pl.H)
        assert o.crt([z % p for p in pl.p], pl) == z
    # worst residues for the running sum: y_l = p_l - 1 for every l
    Wworst = [((p - 1) * pow(pl.q[l], -1, p)) % p for l, p in enumerate(pl.p)]
    o.crt(Wworst, pl)
    res["A5 CRT word at boundary"] = f"N=16 L_w=126 overflows on {hit}/300 random W; L_w=127 never"

    # A6 §3.7 beta epilogue is ONE fma.  k = 1, D = E = -(1 + 2^-51) exactly; beta = 1 + 2^-52,
    # C = 1 + 2^-52: beta C = 1 + 2^-51 + 2^-104, so fma gives 2^-104; RN(beta C) + E gives 0.
    e = -(1 + 2.0**-51)
    out = o.contract([[e]], [[1.0]], [[1 + 2.0**-52]], 1.0, 1 + 2.0**-52, 16)
    assert out[0][0] == 2.0**-104, out
    # random triples (beta, C, D) with frequent cancellation, compared with math.fma
    nt = 0
    for _ in range(12):
        beta = rand_double(rng, -40, 40)
        ds = [rand_double(rng, -300, 300) for _ in range(30)]
        Cs = []
        for d in ds:
            c = -d / beta if rng.random() < 0.6 else rand_double(rng, -300, 300)
            if not math.isfinite(c):
                c = 1.0
            Cs.append([math.nextafter(c, rng.choice((math.inf, -math.inf))) if rng.random() < 0.5 else c])
        t = {}
        out = o.contract([[d] for d in ds], [[1.0]], Cs, 1.0, beta, 16, trace=t)
        for i, d in enumerate(ds):
            assert t["D"][i][0] == d                  # k = 1, N = 16: D = d exactly
            try:
                ref = math.fma(beta, Cs[i][0], d)
            except OverflowError:
                ref = math.copysign(math.inf, fr(beta) * fr(Cs[i][0]) + fr(d))
            assert same_bits(out[i][0], ref), (beta, Cs[i][0], d, out[i][0], ref)
            nt += 1
    res["A6 beta epilogue single fma"] = f"2^-104 case + {nt} random triples vs math.fma"

    # A7 §3.7 native path in canonical k order: MAX*2 + MAX*2 overflows to +Inf, then
    # + (-Inf * 1) = NaN.  Reversed order: -Inf, then -Inf + finite = -Inf.
    MAXD = 1.7976931348623157e308
    out = o.contract([[MAXD, MAXD, -math.inf], [-math.inf, MAXD, MAXD]],
                     [[2.0], [2.0], [1.0]], None, 1.0, 0.0, 9)
    assert math.isnan(out[0][0]) and out[1][0] == -math.inf, out
    res["A7 native canonical order"] = "[MAX,MAX,-Inf].[2,2,1] = NaN (reversed order would give -Inf); [-Inf,MAX,MAX].[2,2,1] = -Inf"

    # A8 §3.5 default chunk K_c' = 131008 (v1.13) really splits at k = 131009 and at
    # k = 131056 (one chunk under the old 131056, two now): add-mod is exact and matches an
    # unsplit run (chunk = K_c = 131071).  Last entry nonzero so chunk 2 matters.
    for k in (o.K_C_ALIGNED + 1, 131056):
        a = [rng.choice((-1.0, 1.0)) * (1 + rng.random()) for _ in range(k)]
        b = [rng.choice((-1.0, 1.0)) * (1 + rng.random()) for _ in range(k)]
        A, B = [a], [[x] for x in b]
        for N in (3, 5):
            t1, t2 = {}, {}
            c1 = o.contract(A, B, None, 1.0, 0.0, N, "TRUNC", trace=t1)
            c2 = o.contract(A, B, None, 1.0, 0.0, N, "TRUNC", trace=t2, chunk=o.K_C)
            exq = sum(o.qvalue(x) * o.qvalue(y) for x, y in zip(t1["QA"][0], t1["QB"][0]))
            last = o.qvalue(t1["QA"][0][-1]) * o.qvalue(t1["QB"][0][-1])
            assert last != 0 and any(last % p for p in t1["plan"].p)
            assert t1["z"][0][0] == exq == t2["z"][0][0] and t1["W"] == t2["W"]
            assert o.bits(c1[0][0]) == o.bits(c2[0][0])
    res["A8 k=131009, 131056 split vs unsplit"] = "identical W planes, z exact (N=3, 5)"

    RESULTS["adversarial"] = "\n".join(f"  {key}: {val}" for key, val in res.items())


# ---------------------------------------------------------------------------------------
# (6) RN vs TRUNC statistics, (7) N for DGEMM-like accuracy
# ---------------------------------------------------------------------------------------

def errs(A, B, C, k):
    """Elementwise relative errors and normwise errors |D-ex|/(||a|| ||b||), vs exact."""
    rel, nrm = [], []
    for i in range(len(A)):
        na = math.sqrt(float(sum(fr(x) ** 2 for x in A[i])))
        for j in range(len(C[0])):
            col = [B[h][j] for h in range(k)]
            nb = math.sqrt(float(sum(fr(x) ** 2 for x in col)))
            ex = dy(o.exact_dot(A[i], col))
            e = abs(fr(C[i][j]) - ex)
            if ex != 0:
                rel.append(float(e / abs(ex)))
            nrm.append(float(e) / (na * nb))
    return rel, nrm


def test_stats():
    rng = random.Random(6)
    k, m, n = 100, 6, 6
    lines = ["(6) k=100, 6x6: median / max elementwise rel. error vs exact  [RN | TRUNC]  ratio=TRUNC/RN medians"]
    data = {}
    for phi in (0, 0.5, 1, 2, 4, 8):
        A = rnd_matrix(rng, m, k, phi)
        B = rnd_matrix(rng, k, n, phi)
        for N in (4, 6, 8, 10, 12, 14, 16):
            row = {}
            for mode in ("RN", "TRUNC"):
                C = o.contract(A, B, None, 1.0, 0.0, N, mode)
                rel, nrm = errs(A, B, C, k)
                row[mode] = (statistics.median(rel), max(rel), statistics.median(nrm), max(nrm))
            data[(phi, N)] = row
            r, tr = row["RN"], row["TRUNC"]
            lines.append(f"  phi={phi:<3} N={N:2d}  RN {r[0]:9.2e} / {r[1]:9.2e} | TRUNC {tr[0]:9.2e} / {tr[1]:9.2e}"
                         f"  ratio {tr[0] / r[0] if r[0] else float('inf'):6.1f}"
                         f"   normwise max RN {r[3]:8.1e} TRUNC {tr[3]:8.1e}")
    ratios = [data[key]["TRUNC"][0] / data[key]["RN"][0] for key in data
              if data[key]["RN"][0] > 1e-14 and data[key]["TRUNC"][0] > 1e-14]
    lines.append(f"  TRUNC/RN median-error ratio over cells above FP64 noise: "
                 f"median {statistics.median(ratios):.1f}, min {min(ratios):.1f}, max {max(ratios):.1f}"
                 f"  (~{math.log2(statistics.median(ratios)) / 8:.2f} moduli at ~8 bits/modulus)")
    # same-sign data (DESIGN §3.3: trunc bias adds up in long sums of same-sign products)
    lines.append("  same-sign inputs (|rand-0.5| * exp(phi randn)), k=1000, 4x4: median rel. error RN | TRUNC  ratio")
    k2 = 1000
    for phi in (0, 2):
        A = [[abs(x) for x in row] for row in rnd_matrix(rng, 4, k2, phi)]
        B = [[abs(x) for x in row] for row in rnd_matrix(rng, k2, 4, phi)]
        for N in (6, 8, 10, 12):
            med = {}
            for mode in ("RN", "TRUNC"):
                C = o.contract(A, B, None, 1.0, 0.0, N, mode)
                med[mode] = statistics.median(errs(A, B, C, k2)[0])
            lines.append(f"    phi={phi} N={N:2d}  {med['RN']:9.2e} | {med['TRUNC']:9.2e}  {med['TRUNC'] / med['RN']:7.1f}")
    RESULTS["stats"] = "\n".join(lines)


def plain_dot(a, b):
    """Recursive summation acc = acc + RN(a*b) (no fma).  NB: Python's sum() of floats is
    Neumaier-compensated since 3.12, so it must not be used as a 'native' reference."""
    acc = 0.0
    for x, y in zip(a, b):
        acc = acc + x * y
    return acc


def test_dgemm_like():
    rng = random.Random(7)
    k, m, n = 1000, 6, 6
    lines = ["(7) k=1000, 6x6: smallest N whose emulated error <= native FP64 fma-dot error"]
    for phi in (0, 1, 2, 4):
        A = rnd_matrix(rng, m, k, phi)
        B = rnd_matrix(rng, k, n, phi)
        Bt = [[B[h][j] for h in range(k)] for j in range(n)]
        Cn = [[o.native_dot(A[i], Bt[j]) for j in range(n)] for i in range(m)]
        Cp = [[plain_dot(A[i], Bt[j]) for j in range(n)] for i in range(m)]
        nrel, nnrm = errs(A, B, Cn, k)
        prel, _ = errs(A, B, Cp, k)
        tgt_med, tgt_max = statistics.median(nrel), max(nrel)
        found = {}
        detail = []
        for mode in ("RN", "TRUNC"):
            for N in range(2, 21):
                C = o.contract(A, B, None, 1.0, 0.0, N, mode)
                rel, nrm = errs(A, B, C, k)
                med, mx = statistics.median(rel), max(rel)
                if mode == "RN" and N in (8, 10, 12, 13, 14, 15, 16):
                    detail.append(f"N={N}:{med:.1e}/{mx:.1e}")
                if (mode, "med") not in found and med <= tgt_med:
                    found[(mode, "med")] = N
                if (mode, "max") not in found and mx <= tgt_max:
                    found[(mode, "max")] = N
                if (mode, "med") in found and (mode, "max") in found:
                    break
        lines.append(f"  phi={phi}: native fma-dot rel err median {tgt_med:.2e} max {tgt_max:.2e}"
                     f" (plain acc+a*b, no fma: median {statistics.median(prel):.2e}); normwise max {max(nnrm):.1e}")
        lines.append(f"     RN    reaches native median at N={found.get(('RN', 'med'))}, native max at N={found.get(('RN', 'max'))}")
        lines.append(f"     TRUNC reaches native median at N={found.get(('TRUNC', 'med'))}, native max at N={found.get(('TRUNC', 'max'))}")
        lines.append("     RN median/max: " + "  ".join(detail))
    RESULTS["dgemm_like"] = "\n".join(lines)


# ---------------------------------------------------------------------------------------
# (8) tensor front end: §5 lowering, scatter vectors, swap, K-empty -> DIRECT
# ---------------------------------------------------------------------------------------

def rand_desc_triple(rng):
    """Random small legal-looking contraction: labels 0..5, extents 0..3, random dense
    layouts (permuted, some negative strides), optional diagonal / single-operand labels.
    The small rank (at most 6 labels) is a coverage choice for this brute-force address
    check, not a limit of the lowering or the library (any rank is legal); operand ranks
    6-12 run on the GPU in tests/test_frontend.py suite highrank (ctest contract_highrank)."""
    nl = rng.randint(1, 5)
    ext = {l: rng.choice((1, 2, 2, 3, 3)) for l in range(nl)}
    if rng.random() < 0.05:
        ext[rng.randrange(nl)] = 0
    labs = list(range(nl))
    la = [l for l in labs if rng.random() < 0.7]
    lb = [l for l in labs if rng.random() < 0.7]
    lc = [l for l in labs if (l in la or l in lb) and rng.random() < 0.5]
    if la and rng.random() < 0.15:
        la.append(rng.choice(la))                       # diagonal in A
    rng.shuffle(la), rng.shuffle(lb), rng.shuffle(lc)

    def mk(lab):
        order = list(range(len(lab)))
        rng.shuffle(order)
        st, acc = [0] * len(lab), 1
        for d in order:
            st[d] = acc * rng.choice((1, 1, 1, -1))
            acc *= max(ext[lab[d]], 1)
        return o.Desc([ext[l] for l in lab], st, lab)
    return mk(la), mk(lb), mk(lc)


def test_tensor():
    rng = random.Random(81)
    # Scatter o GEMM o Gather generates exactly the einsum's (C, A, B) address triples.
    checked = swapped = diag = fused = 0
    for _ in range(400):
        A, B, C = rand_desc_triple(rng)
        try:
            v = o.lower(A, B, C)
        except (o.InvalidValue, o.NotSupported):
            continue
        if v.empty_output():
            continue
        assert len(v.vec_region()) == 2 * v.m + 2 * v.n + 2 * v.k + 3 * v.b
        assert o.view_addresses(v) == o.einsum_addresses(A, B, C), (A.label, B.label, C.label)
        # canonical orders sorted as specified (post-swap frame)
        big, oth = ("A", "B") if v.m >= v.n else ("B", "A")
        s = {X: {lab: abs(x) for lab, x in v.stride[X].items()} for X in "ABC"}
        key = {"I": lambda q: (s["C"][q], s["A"][q], v.first[q]),
               "J": lambda q: (s["C"][q], s["B"][q], v.first[q]),
               "L": lambda q: (s["C"][q], s["A"][q], s["B"][q], v.first[q]),
               "K": lambda q: (s[big][q], s[oth][q], v.first[q])}
        for g in "IJKL":
            ks = [key[g](q) for q in v.order[g]]
            assert ks == sorted(ks) and all(v.ext[q] != 1 for q in v.order[g])
            check_fusion(v, g)
        checked += 1
        fused += sum(len(v.order[g]) - len(v.fused[g]) for g in "IJKL")
        swapped += v.swapped
        diag += len(set(A.label)) != len(A.label)
    assert checked > 100 and swapped > 10 and diag > 3 and fused > 10, \
        (checked, swapped, diag, fused)

    # Fusion (§5 step 9) on hand-made cases. 'ij k1 k2, k1 k2 j' dense: k1 k2 fuse in K.
    i, j, k1, k2, x = 1, 2, 3, 4, 5
    v = o.lower(o.Desc([3, 4, 5], [20, 5, 1], [i, k1, k2]), o.Desc([4, 5, 6], [30, 6, 1], [k1, k2, j]),
                o.Desc([3, 6], [6, 1], [i, j]))
    assert [ls for ls, _, _ in v.fused["K"]] == [[k2, k1]], v.fused
    check_fusion(v, "K")
    # The unsound v1 rule (spec IndexMap neg_fusion): an A-only label x adjacent to a K
    # label in A must NOT fuse with it, because B carries K with stride 0 for x.
    v = o.lower(o.Desc([3, 4, 5], [20, 5, 1], [i, k1, x]), o.Desc([4, 6], [6, 1], [k1, j]),
                o.Desc([3, 6], [6, 1], [i, j]))
    assert len(v.fused["K"]) == 2, v.fused
    check_fusion(v, "K")
    # I fuses only if adjacent in A AND in C: 'abk,k->ab' with C transposed does not fuse.
    a, bb = 6, 7
    # (C's fastest label is in I, so the view is swapped and a, bb end up in J.)
    for cst, want in (([4, 1, 1], 1), ([1, 3, 1], 2)):
        v = o.lower(o.Desc([3, 4, 5], [20, 5, 1], [a, bb, k1]), o.Desc([5, 1], [1, 1], [k1, j]),
                    o.Desc([3, 4, 1], cst, [a, bb, j]))
        grp = "J" if v.swapped else "I"
        assert sorted(v.order[grp]) == [a, bb] and len(v.fused[grp]) == want, v.fused
        check_fusion(v, grp)

    # 'ikl,kjl->ijl' (b = 3, permuted A, C with l slowest): equals the per-batch matrix
    # contraction in the natural order (the emulated path does not depend on K order).
    m, n, k, b = 4, 5, 7, 3
    i, j, kk, l = 1, 2, 3, 4
    A = o.Desc([m, k, b], [k * b, b, 1], [i, kk, l])
    B = o.Desc([k, n, b], [1, k * b, k], [kk, j, l])          # k fastest
    C = o.Desc([m, n, b], [n, 1, m * n], [i, j, l])
    Ab = [rng.uniform(-1, 1) for _ in range(m * k * b)]
    Bb = [rng.uniform(-1, 1) for _ in range(k * n * b)]
    Cb = [rng.uniform(-1, 1) for _ in range(m * n * b)]
    Ab[(1 * k + 2) * b + 1] = math.inf                        # one flagged row in batch 1
    for N, rnd, alpha, beta in ((9, "RN", 1.0, 0.0), (14, "TRUNC", -1.5, 0.75)):
        t = {}
        out = o.contract_tensor(o.Tensor(Ab, 0, A), o.Tensor(Bb, 0, B), o.Tensor(Cb, 0, C),
                                alpha, beta, N, rnd, trace=t)
        assert t["kind"] == "GEMM" and not t["view"].swapped
        for lam in range(b):
            Am = [[Ab[(ii * k + h) * b + lam] for h in range(k)] for ii in range(m)]
            Bm = [[Bb[h + jj * k * b + lam * k] for jj in range(n)] for h in range(k)]
            Cm = [[Cb[ii * n + jj + lam * m * n] for jj in range(n)] for ii in range(m)]
            ref = o.contract(Am, Bm, Cm, alpha, beta, N, rnd)
            for ii in range(m):
                for jj in range(n):
                    assert same_bits(out[ii * n + jj + lam * m * n], ref[ii][jj])

    # Role swap: C stored transposed (i fastest) -> swapped view, same values bit for bit.
    Ct = o.Desc([m, n, b], [1, m, m * n], [i, j, l])
    t = {}
    out = o.contract_tensor(o.Tensor(Ab, 0, A), o.Tensor(Bb, 0, B), o.Tensor(Cb, 0, Ct),
                            1.0, 0.0, 9, trace=t)
    assert t["view"].swapped and (t["view"].m, t["view"].n) == (n, m)
    ref = o.contract_tensor(o.Tensor(Ab, 0, A), o.Tensor(Bb, 0, B), o.Tensor(Cb, 0, C),
                            1.0, 0.0, 9)
    for lam in range(b):
        for ii in range(m):
            for jj in range(n):
                assert same_bits(out[ii + jj * m + lam * m * n], ref[ii * n + jj + lam * m * n])

    # K-empty (outer product) -> DIRECT: D = fma(a, b, +0), so (-1) * (+0) gives +0.
    A1, B1, C1 = o.Desc([2], [1], [i]), o.Desc([3], [1], [j]), o.Desc([2, 3], [3, 1], [i, j])
    a1, b1 = [-1.0, 3.0], [0.0, 2.5, -math.inf]
    t = {}
    out = o.contract_tensor(o.Tensor(a1, 0, A1), o.Tensor(b1, 0, B1),
                            o.Tensor([7.0] * 6, 0, C1), 1.0, 0.0, 2, trace=t)
    assert t["kind"] == "DIRECT"
    assert same_bits(out[0], 0.0) and same_bits(out[1], -2.5) and out[2] == math.inf
    assert same_bits(out[3], 0.0) and same_bits(out[4], 7.5) and out[5] == -math.inf
    # extent-1 K label is dropped -> also K-empty; mode DIRECT on a real K.
    A2 = o.Desc([2, 1], [1, 2], [i, kk])
    B2 = o.Desc([1, 3], [3, 1], [kk, j])
    out2 = o.contract_tensor(o.Tensor(a1, 0, A2), o.Tensor(b1, 0, B2),
                             o.Tensor([7.0] * 6, 0, C1), 1.0, 0.0, 2)
    assert all(same_bits(x, y) for x, y in zip(out, out2))
    t = {}
    o.contract_tensor(o.Tensor(Ab, 0, A), o.Tensor(Bb, 0, B), o.Tensor(Cb, 0, C), 1.0, 0.0,
                      9, mode=o.DIRECT, trace=t)
    assert t["kind"] == "DIRECT"
    # alpha = 0 / k = 0: A and B are not read; beta = 0: C not read (NaN does not leak).
    Ak0 = o.Desc([2, 0], [1, 2], [i, kk])
    Bk0 = o.Desc([0, 3], [3, 1], [kk, j])
    out = o.contract_tensor(o.Tensor(None, 0, Ak0), o.Tensor(None, 0, Bk0),
                            o.Tensor([math.nan] * 6, 0, C1), 1.0, 0.0, 2)
    assert all(same_bits(x, 0.0) for x in out)
    out = o.contract_tensor(o.Tensor(None, 0, A), o.Tensor(None, 0, B), o.Tensor(Cb, 0, C),
                            0.0, -2.0, 9)
    assert all(same_bits(x, -2.0 * y) for x, y in zip(out, Cb))
    # RN acceptance only when the emulated GEMM runs: kappa >= T at N = 2 (T = 180).
    Abig, Bbig = o.Desc([1, 180 * 180], [0, 1], [i, kk]), o.Desc([180 * 180, 1], [1, 0], [kk, j])
    Cs = o.Desc([1, 1], [1, 1], [i, j])
    try:
        o.contract_tensor(o.Tensor(None, 0, Abig), o.Tensor(None, 0, Bbig),
                          o.Tensor([1.0], 0, Cs), 0.0, 1.0, 2)
        raise AssertionError("RN kappa >= T accepted")
    except o.NotSupported:
        pass
    o.contract_tensor(o.Tensor(None, 0, Abig), o.Tensor(None, 0, Bbig), o.Tensor([1.0], 0, Cs),
                      0.0, 1.0, 2, mode=o.DIRECT)
    RESULTS["tensor"] = f"{checked} random lowerings ({swapped} swapped, {diag} diagonal, {fused} fusions)"


# ---------------------------------------------------------------------------------------
# (10) packed operands (DESIGN v1.7 §6b, D1)
# ---------------------------------------------------------------------------------------

def tensor_for(desc, rng, bad=0.0):
    """o.Tensor holding random values for every element of desc (negative strides: base),
    with Inf / NaN / zeros sprinkled at rate `bad`."""
    lo = sum(min(0, s * (e - 1)) for e, s in zip(desc.extent, desc.stride)) if not desc.empty() else 0
    hi = sum(max(0, s * (e - 1)) for e, s in zip(desc.extent, desc.stride)) if not desc.empty() else 0
    buf = []
    for _ in range(hi - lo + 1):
        x = rng.uniform(-1, 1) * 2.0 ** rng.randint(-20, 20)
        if rng.random() < bad:
            x = rng.choice((math.inf, -math.inf, math.nan, 0.0))
        buf.append(x)
    return o.Tensor(buf, -lo, desc)


def check_pack_bytes(data, pv, view, trace, role):
    """Every byte of a pack against the in-line run (same orders): sigma, flags and every
    plane's residues equal the trace; K padding, the tail and the alignment gaps are 0."""
    assert len(data) == pv.pack_bytes
    rows = view.m if role == "A" else view.n
    assert (pv.b, pv.r, pv.k) == (view.b, rows, view.k)
    so, fo, ro = pv.regs["sig"][0], pv.regs["flag"][0], pv.regs["res"][0]
    defined = set()
    plane = pv.b * pv.r * pv.k_hat
    for lam, t in enumerate(trace["batches"]):
        sig, flg = (t["sigma"], t["fA"]) if role == "A" else (t["tau"], t["fB"])
        R = t["RA"] if role == "A" else t["RB"]
        for rho in range(rows):
            idx = lam * pv.r + rho
            assert int.from_bytes(data[so + 2 * idx:so + 2 * idx + 2], "little", signed=True) == sig[rho]
            assert data[fo + idx] == (1 if flg[rho] else 0)
            defined |= {so + 2 * idx, so + 2 * idx + 1, fo + idx}
            for l in range(pv.N):
                base = ro + l * plane + idx * pv.k_hat
                got = [x - 256 if x > 127 else x for x in data[base:base + pv.k]]
                assert got == R[l][rho], (l, lam, rho)
                defined |= set(range(base, base + pv.k))
    assert all(data[p] == 0 for p in range(len(data)) if p not in defined), "padding / tail / gaps"


def test_packed():
    import params as pp
    rng = random.Random(1061)
    # §6b.4 layout: the oracle's transcription == params.py (cert_pack_layout)
    for N in (2, 9, 20):
        for b in (0, 1, 3):
            for r in (0, 1, 17, 32):
                for k in (0, 1, 5, 16, 131057):
                    regs, total = o.pack_layout(b, r, k, N)
                    pregs, ptotal = pp.pack_layout(b, r, k, N)
                    assert total == ptotal and {nm: (off, nb) for nm, off, nb in pregs} == regs
                    assert o.pack_ws_bytes(b, r, k) == pp.pack_ws_bytes(b, r, k)
    stats = {"cases": 0, "packs": 0, "both": 0, "auto": 0, "swapped_role_b": 0, "flagged": 0,
             "orders_differ": 0, "mismatch_classes": 0, "skipped_k_empty": 0}
    # Random contractions: for each caller operand X, the imd_pack_desc_for pack (v1.5's
    # orders) and an AUTO pack (X's stride order); bytes vs the in-line run; packed
    # contractions bit-exact vs the unpacked run in the adopted order (and vs the plain run).
    for it in range(260):
        A, B, C = rand_desc_triple(rng)
        try:
            v = o.lower(A, B, C)
        except (o.InvalidValue, o.NotSupported):
            continue
        if v.empty_output() or v.k < 2:
            continue
        N = rng.choice((2, 3, 5, 9, 14))
        rnd = rng.choice(("RN", "TRUNC"))
        g = rng.randint(1, N)
        bad = 0.15 if rng.random() < 0.3 else 0.0
        TA, TB, TC = tensor_for(A, rng, bad), tensor_for(B, rng, bad), tensor_for(C, rng)
        alpha, beta = rng.choice(((1.0, 0.0), (-0.75, 1.25)))
        plain = o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g)
        packs = [None, None]
        for x, (Xd, Tx) in enumerate(((A, TA), (B, TB))):
            role = ("A", "B")[x] if not v.swapped else ("B", "A")[x]
            partner_only = any(lab not in Xd.label for lab in v.order["K"])
            pd = o.pack_desc_for(A, B, C, o.EMULATE, x, N, rnd)
            try:
                pv = o.pack_lower(pd)
            except o.NotSupported:          # X has no K label left (all partner-only)
                stats["skipped_k_empty"] += 1
                continue
            data = o.pack(Tx, pd)
            pk = o.PackedOperand(pd, data)
            if partner_only:                # §6b.10 #10: PACK_MISMATCH (CLASSES)
                try:
                    o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g,
                                      packs=(pk, None) if x == 0 else (None, pk))
                    raise AssertionError("partner-only K label accepted")
                except o.PackMismatch as e:
                    assert e.reason == o.MISMATCH_CLASSES
                stats["mismatch_classes"] += 1
                continue
            t = {}
            got = o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g, trace=t,
                                    packs=(pk, None) if x == 0 else (None, pk))
            assert all(same_bits(p, q) for p, q in zip(got, plain)), "desc_for pack != plain"
            ti = {}
            o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g, trace=ti)
            check_pack_bytes(data, pv, ti["view"], ti, role)
            packs[x] = pk
            stats["packs"] += 1
            stats["swapped_role_b"] += role == "B"
            stats["flagged"] += any(any(bt["fA"]) or any(bt["fB"]) for bt in ti["batches"])
            # AUTO pack: X's own stride order; == the unpacked run in the adopted order
            K = [lab for lab in dict.fromkeys(Xd.label) if lab not in C.label]
            Lb = [lab for lab in dict.fromkeys(Xd.label)
                  if lab in C.label and lab in (B if x == 0 else A).label]
            pa = o.PackDesc(Xd, N, rnd, o.ORDER_AUTO, K, Lb)
            da = o.pack(Tx, pa)
            ta = {}
            ga = o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g, trace=ta,
                                   packs=(o.PackedOperand(pa, da), None) if x == 0
                                   else (None, o.PackedOperand(pa, da)))
            orders = ta["packed"]["orders"]
            tr = {}
            ref = o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g, orders=orders, trace=tr)
            assert all(same_bits(p, q) for p, q in zip(ga, ref)), "AUTO pack != adopted-order run"
            check_pack_bytes(da, o.pack_lower(pa), tr["view"], tr, role)
            if bad == 0.0:     # unflagged: the order does not matter (EmulatedOrderFree)
                assert all(same_bits(p, q) for p, q in zip(ga, plain))
            stats["orders_differ"] += any(orders[c] != v.order[c] for c in orders)
            stats["auto"] += 1
        if packs[0] is not None and packs[1] is not None:
            got = o.contract_tensor(TA, TB, TC, alpha, beta, N, rnd, g, packs=tuple(packs))
            assert all(same_bits(p, q) for p, q in zip(got, plain)), "both packed != plain"
            stats["both"] += 1
        stats["cases"] += 1
    assert stats["cases"] > 60 and stats["both"] > 20 and stats["swapped_role_b"] > 10 and \
        stats["flagged"] > 5 and stats["orders_differ"] > 5 and stats["mismatch_classes"] > 3, stats

    # One pack reused across contractions: different partners, C layouts (swap: role B),
    # batch L and N row widths; each equals the unpacked run in the adopted order.
    i, j, kk, l, jj = 1, 2, 3, 4, 5
    X = o.Desc([5, 7, 3], [21, 3, 1], [i, kk, l])                    # X[i, k, l], l fastest
    TX = tensor_for(X, rng, 0.1)
    pdX = o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk], [l])
    dX = o.pack(TX, pdX)
    pkX = o.PackedOperand(pdX, dX)
    partners = [(o.Desc([7, 4, 3], [12, 3, 1], [kk, j, l]), o.Desc([5, 4, 3], [12, 3, 1], [i, j, l])),
                (o.Desc([7, 4, 3], [1, 21, 7], [kk, j, l]), o.Desc([5, 4, 3], [1, 15, 5], [i, j, l])),
                (o.Desc([3, 7, 9], [63, 9, 1], [l, kk, jj]), o.Desc([3, 9, 5], [45, 5, 1], [l, jj, i])),
                (o.Desc([2, 7, 3], [21, 3, 1], [j, kk, l]), o.Desc([3, 5, 2], [1, 3, 15], [l, i, j]))]
    roles = set()
    for Bd, Cd in partners:
        TB, TC = tensor_for(Bd, rng), tensor_for(Cd, rng)
        t = {}
        got = o.contract_tensor(TX, TB, TC, 1.0, 0.0, 9, "RN", 3, trace=t, packs=(pkX, None))
        ref = o.contract_tensor(TX, TB, TC, 1.0, 0.0, 9, "RN", 3, orders=t["packed"]["orders"])
        assert all(same_bits(p, q) for p, q in zip(got, ref))
        roles.add(t["packed"]["roles"][0])
        # X as the caller's B of the same contraction (operands exchanged)
        got = o.contract_tensor(TB, TX, TC, 1.0, 0.0, 9, "RN", 3, trace=t, packs=(None, pkX))
        ref = o.contract_tensor(TB, TX, TC, 1.0, 0.0, 9, "RN", 3, orders=t["packed"]["orders"])
        assert all(same_bits(p, q) for p, q in zip(got, ref))
        roles.add(t["packed"]["roles"][1])
    assert roles == {"A", "B"}, roles
    # the same tensor under renamed labels has the same key (label-free) and is accepted
    Xr = o.Desc([5, 7, 3], [21, 3, 1], [11, 13, 14])
    assert o.pack_lower(o.PackDesc(Xr, 9, "RN", o.ORDER_AUTO, [13], [14])).key == o.pack_lower(pdX).key

    # A7 through a pack: the native path of a flagged output follows the ADOPTED K order.
    # Products p(k1, k2) = X Y: p(1,0) = p(1,1) = 2 MAX, p(0,1) = -Inf, p(0,0) = 0. k1
    # fastest (X's AUTO order): 0, Inf, -Inf -> NaN; k2 fastest (the plan's own order: B is
    # big, k2 is fastest in Y): 0, -Inf, ... -> -Inf.
    k1, k2 = 6, 7
    MAX = 1.7976931348623157e308
    Xa = o.Desc([2, 2], [1, 2], [k1, k2])            # X[k1, k2]: k1 fastest
    Ya = o.Desc([2, 2, 4], [8, 1, 24], [k1, k2, j])  # Y[k1, k2, j]: k2 fastest in Y
    Ca = o.Desc([4], [1], [j])
    xa = [0.0, MAX, -math.inf, MAX]
    ya = [0.0] * 96
    for jx in range(4):
        for a1 in range(2):
            for a2 in range(2):
                ya[a1 * 8 + a2 + jx * 24] = ((1.0, 1.0), (2.0, 2.0))[a1][a2]
    TXa, TYa, TCa = o.Tensor(xa, 0, Xa), o.Tensor(ya, 0, Ya), o.Tensor([0.0] * 4, 0, Ca)
    pa = o.PackDesc(Xa, 2, "RN", o.ORDER_AUTO, [k1, k2], [])
    pka = o.PackedOperand(pa, o.pack(TXa, pa))
    t = {}
    got = o.contract_tensor(TXa, TYa, TCa, 1.0, 0.0, 2, packs=(pka, None), trace=t)
    plain = o.contract_tensor(TXa, TYa, TCa, 1.0, 0.0, 2)
    ref = o.contract_tensor(TXa, TYa, TCa, 1.0, 0.0, 2, orders=t["packed"]["orders"])
    assert t["packed"]["orders"]["K"] == [k1, k2] and o.lower(Xa, Ya, Ca).order["K"] == [k2, k1]
    assert all(same_bits(p, q) for p, q in zip(got, ref))
    assert math.isnan(got[0]) and plain[0] == -math.inf, (got, plain)   # oracle A7 via the pack
    # with imd_pack_desc_for (GIVEN, v1.5's orders) it is the plain contraction, bit for bit
    pg = o.pack_desc_for(Xa, Ya, Ca, o.EMULATE, 0, 2)
    got = o.contract_tensor(TXa, TYa, TCa, 1.0, 0.0, 2,
                            packs=(o.PackedOperand(pg, o.pack(TXa, pg)), None))
    assert all(same_bits(p, q) for p, q in zip(got, plain))

    # PACK_MISMATCH reasons (§6b.5), in the plan-time order
    Bd, Cd = partners[0]
    TB, TC = tensor_for(Bd, rng), tensor_for(Cd, rng)

    def reason(packs, N=9, rnd="RN", Ad=X, Bdd=Bd, Cdd=Cd):
        try:
            o.lower_packed(Ad, Bdd, Cdd, o.EMULATE, packs, N, rnd)
            return 0
        except o.PackMismatch as e:
            return e.reason
    assert reason((pdX, None)) == 0
    assert reason((pdX, None), N=8) == o.MISMATCH_MODULI
    assert reason((pdX, None), rnd="TRUNC") == o.MISMATCH_ROUNDING
    # a label batch in the plan but a row in the pack; a partner-only K label; another rank
    assert reason((o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk], []), None)) == o.MISMATCH_CLASSES
    Bpo = o.Desc([7, 4, 3, 2], [24, 6, 2, 1], [kk, j, l, 9])        # label 9 only in B: K
    assert reason((pdX, None), Bdd=Bpo) == o.MISMATCH_CLASSES
    X1 = o.Desc([5, 7, 3, 1], [21, 3, 1, 1], [i, kk, l, 8])
    assert reason((pdX, None), Ad=X1) == o.MISMATCH_CLASSES
    # extents / strides of the plan's X differ from the pack's tensor
    X6 = o.Desc([5, 7, 3], [21, 3, 1], [i, kk, l])
    Xs = o.Desc([5, 7, 3], [1, 5, 35], [i, kk, l])
    assert reason((o.PackDesc(Xs, 9, "RN", o.ORDER_GIVEN, [kk], [l], [i]), None), Ad=X6) == \
        o.MISMATCH_STRIDES
    Xe = o.Desc([6, 7, 3], [21, 3, 1], [i, kk, l])
    Ce = o.Desc([6, 4, 3], [12, 3, 1], [i, j, l])
    assert reason((pdX, None), Ad=Xe, Cdd=Ce) == o.MISMATCH_EXTENTS
    # both packed, K orders differ (two K labels)
    Xk = o.Desc([5, 7, 2], [14, 2, 1], [i, kk, jj])
    Yk = o.Desc([7, 2, 4], [8, 4, 1], [kk, jj, j])
    Ck = o.Desc([5, 4], [4, 1], [i, j])
    p1 = o.PackDesc(Xk, 9, "RN", o.ORDER_GIVEN, [kk, jj], [], [i])
    p2 = o.PackDesc(Yk, 9, "RN", o.ORDER_GIVEN, [jj, kk], [], [j])
    p3 = o.PackDesc(Yk, 9, "RN", o.ORDER_GIVEN, [kk, jj], [], [j])
    assert reason((p1, p2), Ad=Xk, Bdd=Yk, Cdd=Ck) == o.MISMATCH_ORDER
    assert reason((p1, p3), Ad=Xk, Bdd=Yk, Cdd=Ck) == 0
    # key digest: FNV-1a, equal keys -> equal digests
    assert o.pack_lower(p1).digest == o.fnv1a64(o.pack_lower(p1).key)

    # pack_lower statuses (§6b.3, §6b.5)
    def st(pd):
        try:
            o.pack_lower(pd)
            return "SUCCESS"
        except o.InvalidValue:
            return "INVALID_VALUE"
        except o.NotSupported:
            return "NOT_SUPPORTED"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk, 99], [l])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk, kk], [l])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk], [kk])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_GIVEN, [kk], [l], [])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [kk], [l], [i])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 21, "RN", o.ORDER_AUTO, [kk], [l])) == "INVALID_VALUE"
    assert st(o.PackDesc(o.Desc([2, 3], [1, 2], [i, i]), 9, "RN", o.ORDER_AUTO, [i])) == "INVALID_VALUE"
    assert st(o.PackDesc(X, 9, "RN", o.ORDER_AUTO, [], [l])) == "NOT_SUPPORTED"      # K empty
    Xone = o.Desc([5, 1], [1, 5], [i, kk])
    assert st(o.PackDesc(Xone, 9, "RN", o.ORDER_AUTO, [kk])) == "NOT_SUPPORTED"     # k = 1
    big = o.Desc([32042], [1], [kk])                 # N = 2: T = 180, kappa(32042) = 180
    assert st(o.PackDesc(big, 2, "RN", o.ORDER_AUTO, [kk])) == "NOT_SUPPORTED"
    assert st(o.PackDesc(big, 2, "TRUNC", o.ORDER_AUTO, [kk])) == "SUCCESS"
    assert st(o.PackDesc(o.Desc([32041], [1], [kk]), 2, "RN", o.ORDER_AUTO, [kk])) == "SUCCESS"
    assert o.pack_lower(o.PackDesc(o.Desc([0, 4], [4, 1], [i, kk]), 9, "RN", o.ORDER_AUTO,
                                   [kk])).pack_bytes == 0                          # b r = 0
    # DIRECT mode, alpha = 0 and k = 0 calls with a matching pack ignore it (same C bits)
    for mode, alpha in ((o.DIRECT, 1.0), (o.EMULATE, 0.0)):
        got = o.contract_tensor(TX, TB, TC, alpha, 0.5, 9, mode=mode, packs=(pkX, None))
        t = {}
        o.contract_tensor(TX, TB, TC, alpha, 0.5, 9, mode=mode, packs=(pkX, None), trace=t)
        ref = o.contract_tensor(TX, TB, TC, alpha, 0.5, 9, mode=mode, orders=t["packed"]["orders"])
        assert all(same_bits(p, q) for p, q in zip(got, ref))
    # g invariance of the packed path
    outs = [o.contract_tensor(TX, TB, TC, 1.0, 0.0, 9, "RN", g, packs=(pkX, None)) for g in (1, 4, 9)]
    assert all(all(same_bits(p, q) for p, q in zip(outs[0], x)) for x in outs[1:])
    RESULTS["packed"] = ", ".join(f"{k} {v}" for k, v in stats.items())


def test_k_align():
    """DESIGN v1.13 §4b stage 2 (designed; the library still runs A = 16): the oracle's
    k-alignment rule equals params.py's; format-2 layouts equal params.pack_layout; a
    format-2 pack holds format 1's sigma, flags and residues at row stride ceil(k)_64 with
    zero K padding and a zero tail; contractions through format-1 or format-2 packs (A, B,
    both; across the AUTO threshold and forced) are bit-identical to the in-line run (A
    changes no value); the key's FORMAT word is the format; a pack whose A differs from the
    plan's is MISMATCH_FORMAT, which under AUTO cannot happen (pack and plan resolve A from
    the same k) and is reached only by forcing A (the library's test hook)."""
    import params as pp
    for k in range(0, 1500):
        for pol in ("A16", "AUTO", 16, 64):
            assert o.k_align_of(k, pol) == pp.k_align_of(k, pol), (k, pol)
    assert [o.k_align_of(k, "AUTO") for k in (0, 1, 96, 97, 10**9)] == [16, 16, 16, 64, 64]
    for fmt in (o.PACK_FORMAT_A16, o.PACK_FORMAT_A64):
        for N in (2, 9, 20):
            for b in (0, 1, 3):
                for r in (0, 1, 17, 32):
                    for k in (0, 1, 5, 16, 96, 97, 144, 131057):
                        regs, total = o.pack_layout(b, r, k, N, fmt)
                        pregs, ptotal = pp.pack_layout(b, r, k, N, fmt)
                        assert total == ptotal and {nm: (off, nb) for nm, off, nb in pregs} == regs
    rng = random.Random(1131)
    i, j, kk, l = 1, 2, 3, 4
    n_runs = n_mis = 0
    for k in (2, 5, 33, 96, 97, 112, 113, 144, 145, 200):
        N = rng.choice((2, 3, 9))
        rnd = rng.choice(("RN", "TRUNC"))
        X = o.Desc([3, k, 2], [2 * k, 1, k], [i, kk, l])     # X[i, k, l]
        Y = o.Desc([k, 5, 2], [1, k, 5 * k], [kk, j, l])     # Y[k, j, l]
        Cd = o.Desc([3, 5, 2], [5, 1, 15], [i, j, l])
        TX, TY, TC = tensor_for(X, rng, 0.05), tensor_for(Y, rng), tensor_for(Cd, rng)
        try:
            plain = o.contract_tensor(TX, TY, TC, -0.5, 1.25, N, rnd, 2)
        except o.NotSupported:                               # RN kappa >= T at N = 2
            continue
        pdX = o.PackDesc(X, N, rnd, o.ORDER_AUTO, [kk], [l])
        pdY = o.PackDesc(Y, N, rnd, o.ORDER_AUTO, [kk], [l])
        v1 = o.pack_lower(pdX, o.PACK_FORMAT_A16)
        d1 = o.pack(TX, pdX, fmt=o.PACK_FORMAT_A16)
        for pol in ("A16", "AUTO", 16, 64):
            fmt = o.pack_format_for(k, pol)
            A = o.PACK_FORMAT_ALIGN[fmt]
            vX, dX = o.pack_lower(pdX, fmt), o.pack(TX, pdX, fmt=fmt)
            dY = o.pack(TY, pdY, fmt=fmt)
            assert vX.format == fmt and vX.key[0] == fmt and vX.k_hat == -(-k // A) * A
            assert vX.key[1:] == v1.key[1:]            # only the FORMAT word depends on A
            # same sigma / flags / residues as format 1; zero K padding, tail and gaps
            for lam in range(vX.b):
                assert o.pack_rows(vX, dX, lam, vX.r) == o.pack_rows(v1, d1, lam, v1.r)
            ro, plane = vX.regs["res"][0], vX.b * vX.r * vX.k_hat
            for lidx in range(N):
                for row in range(vX.b * vX.r):
                    base = ro + lidx * plane + row * vX.k_hat
                    assert all(x == 0 for x in dX[base + k:base + vX.k_hat]), "K padding"
            tail0 = ro + N * plane
            assert all(x == 0 for x in dX[tail0:]), "tail and gap"
            for packs in (((pdX, dX), None), (None, (pdY, dY)), ((pdX, dX), (pdY, dY))):
                pk = tuple(o.PackedOperand(p[0], p[1], fmt) if p else None for p in packs)
                got = o.contract_tensor(TX, TY, TC, -0.5, 1.25, N, rnd, 2, packs=pk, k_align=pol)
                assert all(same_bits(a, c) for a, c in zip(got, plain)), (k, pol, packs is None)
                n_runs += 1
        # pack and plan alignments differ: only by forcing one of them (the test hook)
        for pack_pol, plan_pol in (("AUTO", 16), ("AUTO", 64), (16, "AUTO"), (64, "AUTO"), (16, 64)):
            pf = o.pack_format_for(k, pack_pol)
            want = o.MISMATCH_FORMAT if pf != o.pack_format_for(k, plan_pol) else 0
            try:
                o.lower_packed(X, Y, Cd, o.EMULATE, (pdX, None), N, rnd, fmts=(pf, None),
                               k_align=plan_pol)
                got = 0
            except o.PackMismatch as e:
                got = e.reason
            assert got == want, (k, pack_pol, plan_pol, got)
            n_mis += want != 0
        # under AUTO the pack and the plan always agree (both resolve A from the same k)
        o.lower_packed(X, Y, Cd, o.EMULATE, (pdX, pdY), N, rnd,
                       fmts=(o.pack_format_for(k, "AUTO"),) * 2, k_align="AUTO")
    assert n_runs > 90 and n_mis > 10, (n_runs, n_mis)
    # the logical k feeds kappa and the error estimates; nothing there sees A (DESIGN §4b)
    for k in (97, 1000):
        assert o.ceil_sqrt(k) == pp.kappa(k)
    RESULTS["k_align"] = f"{n_runs} packed contractions over formats 1/2 bit-exact, {n_mis} FORMAT mismatches"


def check_fusion(v, g):
    """§5 step 9 on group g of view v: the fused modes split the canonical order into
    consecutive runs; each run is a chain (s_X(q) = s_X(p) e(p) for every tensor X), runs
    are maximal, and the fused enumeration equals the label-by-label one."""
    assert [q for ls, _, _ in v.fused[g] for q in ls] == v.order[g]

    def chain(p, q):
        return all(v.stride[X][q] == v.stride[X][p] * v.ext[p] for X in "ABC")
    for t, (ls, e, st) in enumerate(v.fused[g]):
        assert e == math.prod(v.ext[q] for q in ls)
        assert st == {X: v.stride[X][ls[0]] for X in "ABC"}
        assert all(chain(p, q) for p, q in zip(ls, ls[1:]))
        if t + 1 < len(v.fused[g]):
            nxt = v.fused[g][t + 1][0][0]
            assert not all(v.stride[X][nxt] == st[X] * e for X in "ABC"), "fusion not maximal"
    single = [(q, v.ext[q], {X: v.stride[X][q] for X in "ABC"}) for q in v.order[g]]
    fused = [(tuple(ls), e, st) for ls, e, st in v.fused[g]]
    for X in "ABC":
        assert o._radix_vec(single, X) == o._radix_vec(fused, X)



def sigma_extremes():
    """Theoretical sigma range (§3.2 step 5): k=1 row holding the least subnormal at N=20
    (largest sigma); k = 2^40 rows of values just below 2^1024 at N=2, TRUNC (smallest)."""
    hi = o.scale_vector([5e-324], o.Plan(20, 1, "TRUNC"))[0]
    hi_rn = o.scale_vector([5e-324], o.Plan(20, 1, "RN"))[0]
    # emulate a huge-k row via S directly: S = k * ceil(x^2 2^60), x just below 2
    x = 1.7976931348623157e308
    pl = o.Plan(2, 1, "TRUNC")
    lo = {}
    for lk in (17, 30, 40, 62):
        kk = 2**lk
        m_, e_ = o.dyadic(x)
        gamma = o.ilogb(x)
        term = -((-(m_ * m_)) >> -(2 * (e_ - gamma) + o.F))
        u = o.largest_u(kk * term, (pl.T**2) << o.F)
        lo[lk] = u - gamma
    return hi, hi_rn, lo


def test_lazy_lower():
    """lower(..., build_vec=False): same results as the eager lowering, vectors built only
    on access and identical; a legal 2^31-row descriptor lowers in O(rank) time and memory
    (NOTES-M2M3 issue 8: the eager lists needed tens of GB)."""
    import tracemalloc
    rng = random.Random(83)
    fields = ("m", "n", "k", "b", "swapped", "order", "fused", "stride", "first", "ext")
    compared = 0
    for _ in range(300):
        A, B, C = rand_desc_triple(rng)
        for mode in (o.EMULATE, o.DIRECT):
            try:
                ve = o.lower(A, B, C, mode)
            except (o.InvalidValue, o.NotSupported) as ex:
                try:
                    o.lower(A, B, C, mode, build_vec=False)
                except type(ex) as ex2:
                    assert str(ex2) == str(ex)
                else:
                    raise AssertionError("lazy lowering accepted a rejected descriptor")
                continue
            vl = o.lower(A, B, C, mode, build_vec=False)
            for f in fields:
                assert getattr(vl, f) == getattr(ve, f), f
            if not ve.empty_output():
                assert isinstance(vl.vec, o.LazyVecs) and not vl.vec.built()
                assert list(vl.vec) == list(ve.vec) == list(o.GemmView.VEC_ORDER)
                assert vl.vec["C_col"] == ve.vec["C_col"] and vl.vec.built() == {"C_col"}
                assert vl.vec_region() == ve.vec_region() and dict(vl.vec) == ve.vec
                compared += 1
            else:
                assert vl.vec == ve.vec == {}
    assert compared > 100, compared

    # Legal huge descriptors: 'ik,kj->ij' with m = 2^31 - 1 (EMULATE, ON the GemmCoord
    # limit) and m = 2^31 (DIRECT; EMULATE rejects it at row 5), n = 3, k = 4.
    i, j, kk = 1, 2, 3
    budget_bytes, budget_s = 4 * 2**20, 2.0
    for m, mode in ((2**31 - 1, o.EMULATE), (2**31, o.DIRECT), (2**31, o.EMULATE)):
        A = o.Desc([m, 4], [4, 1], [i, kk])
        B = o.Desc([4, 3], [1, 4], [kk, j])
        C = o.Desc([m, 3], [3, 1], [i, j])
        tracemalloc.start()
        t1 = time.perf_counter()
        try:
            v = o.lower(A, B, C, mode, build_vec=False)
            st = "SUCCESS"
        except o.InvalidValue:
            v, st = None, "INVALID_VALUE"
        dt = time.perf_counter() - t1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        want = "INVALID_VALUE" if (mode == o.EMULATE and m > o.GEMM_DIM_MAX) else "SUCCESS"
        assert st == want, (m, mode, st)
        assert peak <= budget_bytes and dt <= budget_s, (m, mode, peak, dt)
        if v is not None:
            assert (v.m, v.n, v.k, v.b, v.swapped) == (m, 3, 4, 1, False)
            assert not v.vec.built() and len(v.vec) == 9
            # small vectors are still available, built on access only
            assert v.vec["C_col"] == [0, 1, 2] and v.vec["A_k"] == [0, 1, 2, 3]
            assert v.vec.built() == {"C_col", "A_k"}
        RESULTS.setdefault("lazy_lower", []).append(
            f"m={m} {mode}: {st}, peak {peak / 1024:.1f} KiB, {dt * 1e3:.2f} ms")


def main():
    t0 = time.time()
    for name, fn in (("primitives+negatives", test_primitives),
                     ("params.py cross-check", test_params_crosscheck), ("sweep (1)+(2)", test_sweep),
                     ("determinism (3)", test_determinism), ("non-finite (4)", test_nonfinite),
                     ("edges (5)", test_edges), ("adversarial (A)", test_adversarial),
                     ("stats (6)", test_stats),
                     ("dgemm-like (7)", test_dgemm_like), ("tensor front end (8)", test_tensor),
                     ("lazy lowering (9)", test_lazy_lower),
                     ("packed operands (10)", test_packed),
                     ("k alignment (11, v1.13 §4b)", test_k_align)):
        t1 = time.time()
        fn()
        print(f"PASS  {name:24s} {time.time() - t1:7.1f}s", flush=True)
    hi, hi_rn, lo = sigma_extremes()
    print()
    print("moduli source:", o.MODULI_SOURCE)
    for key in ("tensor", "packed", "k_align", "params_crosscheck", "fma_checked", "crt_L128_N17_overflows", "neg_rn_without_kappa",
                "rn_boundary_N2_k32041", "sweep_runs", "determinism_runs"):
        print(f"{key}: {RESULTS[key]}")
    print(f"bound tightness: max |emul-exact|/l1-bound = {BOUND_STATS['l1_max_ratio']:.3f}; "
          f"max err/normwise-bound = {BOUND_STATS['norm_max_ratio']:.3f}; "
          f"max 2^-sigma / (2||a||/den) = {BOUND_STATS['twopow_max_ratio']:.4f} "
          f"(DESIGN's uncorrected second-order claim; the asserted form adds k 4^gamma 2^-F)")
    print(f"uncorrected 2^-sigma <= 2||a||/den fails exactly on {BOUND_STATS['twopow_uncorrected_fail']}"
          f" of {BOUND_STATS['twopow_rows']} checked rows/columns; corrected form holds on all")
    print(f"sigma/tau observed range in tests: [{SIGMA_SEEN[0]}, {SIGMA_SEEN[1]}]")
    print(f"sigma theoretical: max {hi} (TRUNC) / {hi_rn} (RN) at N=20 k=1 least subnormal; "
          f"min at N=2 TRUNC, max-double row: " + ", ".join(f"k=2^{a}: {b}" for a, b in lo.items()))
    print("adversarial boundary cases (A):")
    print(RESULTS["adversarial"])
    print(RESULTS["stats"])
    print(RESULTS["dgemm_like"])
    print("lazy lowering (build_vec=False):")
    for line in RESULTS["lazy_lower"]:
        print("  " + line)
    print(f"\nALL PASS in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
