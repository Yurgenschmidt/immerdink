"""Exact, bit-level executable reference of DESIGN.md §3 (matrix case, one batch).

    C[i,j] <- alpha * sum_h A[i,h] * B[h,j] + beta * C[i,j]

A is m x k, B is k x n, C is m x n; all entries are Python floats (IEEE binary64).
contract() is one batch of the GEMM view with canonical k-order h = 0, 1, ..., k-1
(DESIGN §1.1). contract_tensor() (end of file) adds the tensor front end: the §5 lowering
(canonical orders, int64 scatter vectors, role swap), batching (b > 1) and the §4 DIRECT
dispatch for K-empty contractions, i.e. exactly what the library computes.

Every step of DESIGN §3 is implemented literally and in exact integer arithmetic:

  §2    constants        Plan: greedy moduli, P, H, M_l, q_l, T = isqrt(H), K_c, L_w
  §3.1  non-finite scan  scale_vector(): flag = row/col has Inf/NaN; those entries -> 0
  §3.2  exact shift      scale_vector(): gamma = ilogb(max), S = sum ceil(x^2 2^60),
                         largest u with 4 S 2^(2u) <= (2T-kappa)^2 2^F (RN) or
                         S 2^(2u) <= T^2 2^F (TRUNC); sigma = u - gamma (int16)
                         plan-time NOT_SUPPORTED when kappa >= T in RN mode
  §3.3  quantization     quantize(): a' = RN_even / trunc (a 2^sigma) held as mu * 2^eps
  §3.4  residues         residue(): SMod((mu mod p) * (2^eps mod p), p)
  §3.5  INT8 GEMM        modular_gemm(): K-chunks of length `chunk` (default 131008),
                         int32 accumulation with an asserted no-saturation bound,
                         store-mod (first chunk) / add-mod (later chunks) into u8 planes
  §3.6  CRT              crt(): division-free running reduction, asserted word bound
                         x < 2^L_w (L_w = 32 * L32), then centering
  §3.7  output           round_dyadic(): ONE correctly rounded conversion of z 2^-(s+t)
                         (subnormals and overflow to +-Inf handled exactly);
                         native path for flagged elements: fma dot in canonical order
                         (fma emulated exactly + correct rounding); alpha / beta rules.

No floating-point arithmetic is used anywhere in the numerics: floats are decomposed into
(integer significand, exponent) pairs, and the only float ever produced is built with
math.ldexp from an already-rounded <= 54-bit integer, which is exact.

Design choices that DESIGN.md leaves open are marked ORACLE-CHOICE below (and reported):
  * native-path accumulator starts at +0.0 (NATIVE_ACC_INIT);
  * the emulated path returns +0.0 for z = 0 (an exact integer zero has no sign);
  * any K-chunk length 1..K_c is accepted (the 64-alignment of 131008 is a device concern),
    so determinism across chunkings can be tested with small chunks.

The residue-group size g only changes the ORDER in which planes are produced here (each
plane is computed independently from the same quantized operands), so g-independence holds
by construction in this oracle; that the real buffer schedule is g-independent is the
Pipeline spec's claim, not something this file can test.

Pure stdlib. Constants come from spec/params.py when present and are always re-derived
locally by the identical greedy rule and cross-checked against the DESIGN §2 table.
"""

from __future__ import annotations

import itertools
import math
import os
import struct
import sys
from collections.abc import Mapping
from fractions import Fraction

# ---------------------------------------------------------------------------------------
# §2 Constants
# ---------------------------------------------------------------------------------------

F = 60                                   # §3.2 fixed-point fraction bits of S
INT32_MAX = 2**31 - 1
RESIDUE_ABS_MAX = 128                    # |r| <= 128 for every modulus (p = 256: -128)
K_C = INT32_MAX // RESIDUE_ABS_MAX**2    # §2: 131071, max K-chunk with no saturation
K_C_ALIGNED = (K_C // 64) * 64           # §3.5 (v1.13): 131008, default chunk length
INT16_MIN, INT16_MAX = -2**15, 2**15 - 1
N_MIN, N_MAX = 2, 20
K_ALIGN = 16                             # §4: the plan's K alignment A: 16 ...
K_ALIGN_PAD = 64                         # §4b (v1.13 stage 2): ... or 64 for k > K_PAD64_MIN
K_PAD64_MIN = 96                         # §4b: AUTO(k) = 64 iff k > 96
# The library's K-alignment policy (§4b): "A16" = every plan and pack at A = 16 (implemented
# through v1.13 stage 1); "AUTO" = stage 2 (designed). The stage-2 implementation flips this
# switch (with params.K_ALIGN_POLICY); every function also takes an explicit A / format.
K_ALIGN_POLICY = "A16"


def k_hat_of(k, A=K_ALIGN):
    """§4: k_hat = ceil(k)_A, the residue row stride and K extent (typed from DESIGN;
    test_oracle cross-checks it with params.k_hat_of)."""
    return -(-k // A) * A


def k_align_of(k, policy=None):
    """§4b: the resolved K alignment of a plan or pack of (logical, lowered) contraction length
    k: AUTO = 64 iff k > 96 else 16; "A16" = 16; 16 / 64 = forced (the library's test hook)."""
    policy = K_ALIGN_POLICY if policy is None else policy
    if policy == "A16":
        return K_ALIGN
    if policy == "AUTO":
        return K_ALIGN_PAD if k > K_PAD64_MIN else K_ALIGN
    if policy in (K_ALIGN, K_ALIGN_PAD):
        return policy
    raise ValueError(f"k-alignment policy {policy!r}")


DESIGN_MODULI = (256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
                 223, 217, 211, 199, 197, 193, 191, 181, 179, 173)

NATIVE_ACC_INIT = 0.0                    # ORACLE-CHOICE: initial native fma accumulator


def greedy_moduli(count: int = N_MAX, start: int = 256) -> tuple:
    """§2: greedy pairwise-coprime moduli from `start` downwards."""
    out = []
    p = start
    while len(out) < count:
        if all(math.gcd(p, q) == 1 for q in out):
            out.append(p)
        p -= 1
    return tuple(out)


def _load_moduli():
    local = greedy_moduli()
    assert local == DESIGN_MODULI, "greedy rule disagrees with DESIGN §2 table"
    source = "local greedy rule (spec/params.py absent)"
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "params.py")):
        sys.path.insert(0, here)
        try:
            import params  # noqa: F401  (owned by the params agent)
            for name in ("REAL_MODULI", "MODULI", "moduli"):
                cand = getattr(params, name, None)
                if callable(cand):
                    try:
                        cand = cand()
                    except TypeError:
                        cand = None
                if cand is not None:
                    cand = tuple(int(x) for x in cand)
                    assert cand[:len(local)] == local[:len(cand)], \
                        "spec/params.py moduli disagree with the greedy rule"
                    source = f"spec/params.py:{name} (cross-checked with greedy rule)"
                    break
            else:
                source = "local greedy rule (spec/params.py has no moduli table)"
        except Exception as exc:  # params.py is being written concurrently
            source = f"local greedy rule (spec/params.py import failed: {exc!r})"
        finally:
            sys.path.pop(0)
    return local, source


MODULI, MODULI_SOURCE = _load_moduli()


def word_bits(N: int) -> int:
    """§2 L_w: 128 bits (4 x 32) for N <= 16, 160 bits (5 x 32) for N <= 20."""
    return 128 if N <= 16 else 160


class NotSupported(Exception):
    """Plan-time NOT_SUPPORTED (e.g. §3.2: kappa >= T in RN mode)."""


class InvalidValue(Exception):
    """Plan-time INVALID_VALUE (option out of range)."""


def ceil_sqrt(k: int) -> int:
    r = math.isqrt(k)
    return r if r * r == k else r + 1


class Plan:
    """Plan-time constants for (N, k, rounding, g, chunk).  §2, §3.2 step 4, §3.5."""

    def __init__(self, N, k, rounding="RN", g=None, chunk=K_C_ALIGNED, L_w=None):
        if not (N_MIN <= N <= N_MAX):
            raise InvalidValue(f"N={N} outside {N_MIN}..{N_MAX}")
        if rounding not in ("RN", "TRUNC"):
            raise InvalidValue(f"rounding={rounding!r}")
        g = N if g in (None, 0) else g
        if not (1 <= g <= N):
            raise InvalidValue(f"g={g} outside 1..N")
        if not (1 <= chunk <= K_C):
            raise InvalidValue(f"K-chunk {chunk} outside 1..K_c={K_C}")
        self.N, self.k, self.rounding, self.g, self.chunk = N, k, rounding, g, chunk
        self.p = MODULI[:N]
        self.P = math.prod(self.p)
        assert self.P % 2 == 0                         # 256 is always in the prefix
        self.H = (self.P - 1) // 2
        self.M = [self.P // pl for pl in self.p]
        self.q = [pow(Ml % pl, -1, pl) for Ml, pl in zip(self.M, self.p)]
        assert all(1 <= ql < pl for ql, pl in zip(self.q, self.p))
        self.T = math.isqrt(self.H)
        assert self.T * self.T <= self.H < (self.T + 1) ** 2
        self.kappa = ceil_sqrt(k) if k > 0 else 0
        self.L_w = word_bits(N) if L_w is None else L_w   # L_w override: tests only
        self.L32 = self.L_w // 32
        if L_w is None:
            assert 2 * self.P <= 2 ** self.L_w, "§2: 2P <= 2^L_w violated"
        # §3.2 step 4: plan-time rejection
        if rounding == "RN" and k > 0 and self.kappa >= self.T:
            raise NotSupported(f"N too small for this k (kappa={self.kappa} >= T={self.T})")
        # §3.2 step 4 right-hand sides, exact integers
        if rounding == "RN":
            self.shift_lhs_mul = 4
            self.shift_rhs = ((2 * self.T - self.kappa) ** 2) << F
        else:
            self.shift_lhs_mul = 1
            self.shift_rhs = (self.T ** 2) << F

    def groups(self):
        """§4: residue groups of g consecutive moduli (the last may be shorter)."""
        return [list(range(s, min(s + self.g, self.N))) for s in range(0, self.N, self.g)]


# ---------------------------------------------------------------------------------------
# Exact binary64 helpers
# ---------------------------------------------------------------------------------------

def is_finite(x: float) -> bool:
    return math.isfinite(x)


def signbit(x: float) -> bool:
    return struct.unpack("<Q", struct.pack("<d", x))[0] >> 63 == 1


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def dyadic(x: float):
    """Finite float -> (m, e) with x == m * 2^e exactly; m odd or m == 0."""
    assert math.isfinite(x)
    if x == 0.0:
        return 0, 0
    num, den = x.as_integer_ratio()          # den is a power of two
    e = -(den.bit_length() - 1)
    tz = (num & -num).bit_length() - 1       # strip trailing zeros of num
    return num >> tz, e + tz


def ilogb(x: float) -> int:
    """floor(log2 |x|) for finite nonzero x (true exponent, also for subnormals)."""
    m, e = dyadic(x)
    return abs(m).bit_length() - 1 + e


def dyadic_to_fraction(m: int, e: int) -> Fraction:
    return Fraction(m * 2**e) if e >= 0 else Fraction(m, 2**-e)


EMIN_SUB = -1074      # exponent of the least subnormal
PREC = 53


def round_dyadic(m: int, e: int, negzero: bool = False) -> float:
    """Correctly rounded (RN-even) binary64 value of m * 2^e, exact for every input:
    gradual underflow (the rounding position never goes below 2^-1074) and overflow
    (the unbounded-exponent rounded value is >= 2^1024 -> +-Inf, IEEE 754 §7.4).
    A zero result carries the sign of m (or `negzero` when m == 0)."""
    if m == 0:
        return -0.0 if negzero else 0.0
    neg = m < 0
    M = -m if neg else m
    top = M.bit_length() - 1 + e                     # exponent of the leading bit
    q = max(top - (PREC - 1), EMIN_SUB)              # exponent of the result's lsb
    if q <= e:
        r = M << (e - q)                             # exact, no rounding needed
    else:
        sh = q - e
        r = M >> sh
        rem = M & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (r & 1)):
            r += 1
    assert r <= 2**PREC
    if r == 0:
        return -0.0 if neg else 0.0
    if r.bit_length() - 1 + q >= 1024:               # overflow after rounding
        return -math.inf if neg else math.inf
    v = math.ldexp(float(r), q)                      # exact: r <= 2^53, q >= -1074
    return -v if neg else v


def round_fraction(x: Fraction) -> float:
    """Correctly rounded binary64 value of an arbitrary rational (RN-even)."""
    if x == 0:
        return 0.0
    neg = x < 0
    a = -x if neg else x
    n, d = a.numerator, a.denominator
    # choose e so that the integer quotient has >= PREC + 2 bits and keep a sticky bit
    e = n.bit_length() - d.bit_length() - (PREC + 3)
    e = max(e, EMIN_SUB - 3)
    if e >= 0:
        qt, rm = divmod(n, d << e)
    else:
        qt, rm = divmod(n << -e, d)
    qt = (qt << 1) | (1 if rm else 0)               # sticky bit, exponent e - 1
    r = round_dyadic(qt, e - 1)
    return -r if neg else r


def fma(a: float, b: float, c: float) -> float:
    """IEEE 754 fusedMultiplyAdd(a, b, c), round-to-nearest-even, emulated exactly."""
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return math.nan
    pneg = signbit(a) != signbit(b)
    if math.isinf(a) or math.isinf(b):
        if a == 0.0 or b == 0.0:                     # Inf * 0: invalid
            return math.nan
        if math.isinf(c) and signbit(c) != pneg:     # Inf - Inf: invalid
            return math.nan
        return -math.inf if pneg else math.inf
    if math.isinf(c):
        return c
    ma, ea = dyadic(a)
    mb, eb = dyadic(b)
    mc, ec = dyadic(c)
    mp, ep = ma * mb, ea + eb
    if mp == 0:
        if mc == 0:
            # exact zero sum of two zeros: -0 only if both are -0 (RN)
            return -0.0 if (pneg and signbit(c)) else 0.0
        return c
    if mc == 0:
        s, es = mp, ep
    else:
        e0 = min(ep, ec)
        s, es = (mp << (ep - e0)) + (mc << (ec - e0)), e0
    if s == 0:
        return 0.0                                   # exact cancellation: +0 in RN
    return round_dyadic(s, es)


def mul(a: float, b: float) -> float:
    """IEEE multiply RN(a*b), via fma(a, b, -0): -0 is the exact additive identity."""
    return fma(a, b, -0.0)


# ---------------------------------------------------------------------------------------
# §3.1 + §3.2 Non-finite scan and exact-integer shift
# ---------------------------------------------------------------------------------------

def largest_u(L: int, R: int) -> int:
    """Largest integer u (any sign) with L * 2^(2u) <= R, for L > 0, R > 0."""
    assert L > 0 and R > 0

    def ok(u):
        return (L << (2 * u)) <= R if u >= 0 else L <= (R << (-2 * u))

    u = (R.bit_length() - L.bit_length()) // 2
    while not ok(u):
        u -= 1
    while ok(u + 1):
        u += 1
    return u


def scale_vector(vec, plan: Plan, rule: str | None = None):
    """§3.1 + §3.2 for one row of A (or column of B).

    Returns (sigma, flag, S, gamma, u).  Non-finite entries set the flag and are treated
    as 0.  `rule` overrides the shift rule ("RN"/"TRUNC"); used ONLY by negative tests."""
    flag = False
    finite = []
    for v in vec:
        if math.isfinite(v):
            finite.append(v)
        else:
            flag = True
            finite.append(0.0)
    nz = [dyadic(v) for v in finite if v != 0.0]
    if not nz:
        return 0, flag, 0, None, None                 # zero row: sigma := 0
    gamma = max(abs(m).bit_length() - 1 + e for m, e in nz)   # ilogb(max |a_h|)
    S = 0
    for m, e in nz:
        # x_h = m 2^(e-gamma), |x_h| < 2;  x_h^2 2^F = m^2 2^(2(e-gamma)+F), ceiling
        t = 2 * (e - gamma) + F
        sq = m * m
        assert sq < 2**106                            # 53 x 53-bit product
        S += (sq << t) if t >= 0 else -((-sq) >> (-t))
    assert 2**F <= S < 2**128 or plan.k >= 2**66, "§3.2: S fits uint128 for k < 2^66"
    # DESIGN §3.2 step 4 "If S = 0, sigma := 0" is unreachable here: the max entry has
    # |x| >= 1, so S >= 2^F for every nonzero row; the all-zero row returned above.
    assert S > 0
    rule = rule or plan.rounding
    if rule == plan.rounding:
        L, R = plan.shift_lhs_mul * S, plan.shift_rhs
    elif rule == "TRUNC":
        L, R = S, (plan.T ** 2) << F
    else:
        L, R = 4 * S, ((2 * plan.T - plan.kappa) ** 2) << F
    u = largest_u(L, R)
    sigma = u - gamma
    assert INT16_MIN <= sigma <= INT16_MAX, "§3.2 step 5: sigma must fit int16"
    return sigma, flag, S, gamma, u


# ---------------------------------------------------------------------------------------
# §3.3 Quantization, §3.4 residues
# ---------------------------------------------------------------------------------------

def quantize(v: float, sigma: int, rounding: str):
    """§3.3: a' = RN_even(v 2^sigma) or trunc(v 2^sigma), held as (mu, eps), a' = mu 2^eps."""
    if not math.isfinite(v) or v == 0.0:
        return 0, 0
    m, e = dyadic(v)
    s = e + sigma
    if s >= 0:
        mu, eps = m, s                                # already an integer
    else:
        sh = -s
        neg = m < 0
        M = -m if neg else m
        r = M >> sh
        if rounding == "RN":
            rem = M & ((1 << sh) - 1)
            half = 1 << (sh - 1)
            if rem > half or (rem == half and (r & 1)):
                r += 1
        mu, eps = (-r if neg else r), 0
    assert abs(mu) <= 2**53 and eps >= 0, "§3.3: |mu| <= 2^53, eps >= 0"
    return mu, eps


def smod(x: int, p: int) -> int:
    """SMod(x, p) in [-floor(p/2), ceil(p/2) - 1]."""
    h = p // 2
    return (x + h) % p - h


def residue(mu: int, eps: int, p: int) -> int:
    """§3.4: SMod((mu mod p) * (2^eps mod p), p)."""
    r = smod((mu % p) * pow(2, eps, p), p)
    assert -RESIDUE_ABS_MAX <= r <= 127
    return r


# ---------------------------------------------------------------------------------------
# §3.5 Per-modulus INT8 GEMM with K-chunked int32 accumulation and modular epilogue
# ---------------------------------------------------------------------------------------

class Saturation(AssertionError):
    """An int32 .satfinite accumulation could saturate (§2 K_c, §3.5)."""


def _chunk_dot(ra, rb):
    """Exact int32 dot of one K-chunk; asserts that no accumulation order can leave int32.
    Every partial sum, in any order, is bounded by sum |ra_h rb_h|."""
    if len(ra) * max(map(abs, ra), default=0) * max(map(abs, rb), default=0) > INT32_MAX:
        worst = sum(abs(x * y) for x, y in zip(ra, rb))
        if worst > INT32_MAX:
            raise Saturation(f"chunk of {len(ra)}: sum|r_a r_b| = {worst} > 2^31-1")
    return sum(map(int.__mul__, ra, rb))


def modular_gemm(RA, RBt, p: int, chunk: int):
    """§3.5: W[i][j] = (sum_h RA[i][h] RBt[j][h]) mod p as u8 planes, chunk by chunk.
    RA: m rows of k residues; RBt: n columns of B (each k residues)."""
    m, n = len(RA), len(RBt)
    k = len(RA[0]) if m else 0
    W = [[0] * n for _ in range(m)]
    first = True
    for c0 in range(0, k, chunk):
        c1 = min(c0 + chunk, k)
        for i in range(m):
            ra = RA[i][c0:c1]
            Wi = W[i]
            for j in range(n):
                acc = _chunk_dot(ra, RBt[j][c0:c1])
                assert -2**31 <= acc <= INT32_MAX
                Wi[j] = acc % p if first else (Wi[j] + acc) % p   # store-mod / add-mod
                assert 0 <= Wi[j] < p <= 256                      # fits u8
        first = False
    return W


# ---------------------------------------------------------------------------------------
# §3.6 CRT reconstruction
# ---------------------------------------------------------------------------------------

class WordOverflow(AssertionError):
    """The running CRT sum exceeded the L_w-bit word (§2, §3.6)."""


def crt(Wl, plan: Plan) -> int:
    """§3.6: division-free running CRT of the N residues W_l in [0, p_l); returns centered z."""
    x = 0
    limit = 1 << plan.L_w
    for l in range(plan.N):
        pl = plan.p[l]
        assert 0 <= Wl[l] < pl
        y = (plan.q[l] * Wl[l]) % pl                 # 8-bit x 8-bit
        x = x + plan.M[l] * y                        # L_w-bit; M_l y < P
        if x >= limit:
            raise WordOverflow(f"CRT peak {x.bit_length()} bits > L_w = {plan.L_w}")
        if x >= plan.P:
            x -= plan.P
        assert 0 <= x < plan.P                       # invariant x < P
    return x if x <= plan.H else x - plan.P          # centering


# ---------------------------------------------------------------------------------------
# §3.7 Native path and the full contraction
# ---------------------------------------------------------------------------------------

def native_dot(a_row, b_col, init: float = NATIVE_ACC_INIT) -> float:
    """§3.7 native path: fma dot product in canonical k order (round to nearest)."""
    acc = init
    for x, y in zip(a_row, b_col):
        acc = fma(x, y, acc)
    return acc


def contract(A, B, C, alpha: float, beta: float, N: int, rounding: str = "RN",
             g: int | None = None, chunk: int = K_C_ALIGNED, m=None, n=None, k=None,
             trace: dict | None = None, _rule: str | None = None, _L_w: int | None = None,
             ext_a=None, ext_b=None):
    """C_out = alpha A B + beta C per DESIGN §3 (returns a new m x n list of lists).

    A: m x k, B: k x n, C: m x n (lists of lists of floats).  A and B are not touched
    when alpha == 0 or k == 0; C is not touched when beta == 0 (pass a poisoned object to
    check).  Shapes may be given explicitly (m, n, k) so that unread operands can be None.
    `trace`, if a dict, receives sigma, tau, flags, quantized operands, residue planes
    RA[l][i][h] / RB[l][j][h] (every modulus l), W planes, z and D.
    `_rule` / `_L_w` override the shift rule / CRT word width: negative tests only.
    `ext_a` / `ext_b` (DESIGN v1.7 §6b, packed operands): (sigma, flags, planes) of A's rows
    (B's columns) read from a packed buffer instead of being computed here: planes[l][i][h],
    h < k, the int8 residues of every modulus l. A / B are then read only by the native path
    of flagged elements."""
    if m is None:
        m = len(A)
    if k is None:
        k = len(A[0]) if m else len(B)
    if n is None:
        n = len(B[0]) if k else len(C[0])
    if m == 0 or n == 0:
        return [[] for _ in range(m)]                # empty I/J: no-op
    plan = Plan(N, k, rounding, g, chunk, L_w=_L_w)  # plan time (may raise NotSupported)

    def finish(i, j, E):
        if beta == 0.0:
            return E                                  # C not read
        return fma(beta, C[i][j], E)

    if alpha == 0.0 or k == 0:                        # BLAS: C <- beta C, A/B not read
        return [[(0.0 if beta == 0.0 else mul(beta, C[i][j])) for j in range(n)]
                for i in range(m)]

    Bt = [[B[h][j] for h in range(k)] for j in range(n)]   # columns of B

    # K1: §3.1 + §3.2 (a packed operand brings its sigma and flags, §6b)
    srow = [scale_vector(A[i], plan, _rule) for i in range(m)] if ext_a is None else None
    scol = [scale_vector(Bt[j], plan, _rule) for j in range(n)] if ext_b is None else None
    sigma = [s[0] for s in srow] if ext_a is None else list(ext_a[0])
    fA = [s[1] for s in srow] if ext_a is None else list(ext_a[1])
    tau = [s[0] for s in scol] if ext_b is None else list(ext_b[0])
    fB = [s[1] for s in scol] if ext_b is None else list(ext_b[1])

    # §3.3 quantization, once per element (K2/K3 recompute this per group on device)
    QA = [[quantize(v, sigma[i], rounding) for v in A[i]] for i in range(m)] if ext_a is None else None
    QB = [[quantize(v, tau[j], rounding) for v in Bt[j]] for j in range(n)] if ext_b is None else None

    # K2-K4 per residue group: residues (§3.4) then per-modulus chunked GEMM (§3.5)
    Wplanes = [None] * N
    RAall, RBall = [None] * N, [None] * N             # trace only: residue planes
    for G in plan.groups():
        RA = {l: ([[residue(mu, ep, plan.p[l]) for mu, ep in QA[i]] for i in range(m)]
                  if ext_a is None else ext_a[2][l]) for l in G}
        RB = {l: ([[residue(mu, ep, plan.p[l]) for mu, ep in QB[j]] for j in range(n)]
                  if ext_b is None else ext_b[2][l]) for l in G}
        for l in G:
            Wplanes[l] = modular_gemm(RA[l], RB[l], plan.p[l], plan.chunk)
            if trace is not None:
                RAall[l], RBall[l] = RA[l], RB[l]
        del RA, RB                                    # buffers reused by the next group

    # K5: CRT, unscale, native path, alpha/beta
    out = [[0.0] * n for _ in range(m)]
    Z = [[None] * n for _ in range(m)]
    Dm = [[None] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if fA[i] or fB[j]:
                D = native_dot(A[i], Bt[j])
            else:
                z = crt([Wplanes[l][i][j] for l in range(N)], plan)
                Z[i][j] = z
                D = round_dyadic(z, -(sigma[i] + tau[j]))   # one rounding (§3.7)
            Dm[i][j] = D
            E = D if alpha == 1.0 else mul(alpha, D)
            out[i][j] = finish(i, j, E)
    if trace is not None:
        trace.update(plan=plan, sigma=sigma, tau=tau, fA=fA, fB=fB, QA=QA, QB=QB,
                     RA=RAall, RB=RBall, W=Wplanes, z=Z, D=Dm, srow=srow, scol=scol)
    return out


# ---------------------------------------------------------------------------------------
# Exact references (used by tests)
# ---------------------------------------------------------------------------------------

def exact_dot(a_row, b_col):
    """Exact sum_h a_h b_h of finite floats as a dyadic (m, e)."""
    terms = []
    for x, y in zip(a_row, b_col):
        if x != 0.0 and y != 0.0:
            mx, ex = dyadic(x)
            my, ey = dyadic(y)
            terms.append((mx * my, ex + ey))
    if not terms:
        return 0, 0
    e0 = min(e for _, e in terms)
    return sum(t << (e - e0) for t, e in terms), e0


def qvalue(q) -> int:
    mu, eps = q
    return mu << eps


# ---------------------------------------------------------------------------------------
# §5 IndexMap lowering and the tensor front end (§1.1, §3.7, §4 DIRECT)
# ---------------------------------------------------------------------------------------
#
# What the library computes for a tensor contraction is a function of the GEMM view: b
# independent m x n x k matrix contractions over the canonical index spaces of §1.1, read
# and written through int64 additive scatter vectors (§5 step 10). Numerically only the
# canonical K order matters (it is the native path's fma order, §3.7); everything else in
# the emulated path is exact or per element. The lowering below is a literal transcription
# of DESIGN §5 steps 1-10. Step 9 (fusion) merges consecutive labels of a group without
# changing the enumeration, so the scatter vectors built from the fused modes equal the
# ones built label by label (spec/test_oracle.py checks this); the fused modes are part of
# the view so the library's lowering can be compared with them.
#
# DESIGN v1.1 §5 step 8 (shared with the library):
#   * the coalescing swap of step 8 is applied FIRST, and the canonical orders are then
#     computed in the post-swap frame (A, B, m, n and I, J are the post-swap ones, so
#     "big" in the K key is the post-swap A when m >= n);
#   * "position" is the first occurrence in the caller's A.labels ++ B.labels ++ C.labels;
#   * K-empty (k = 1 after lowering) and mode DIRECT dispatch to the native path for every
#     element: acc = +0, fma in canonical k order (a single fma for K-empty), then alpha/beta
#     exactly as §3.7.

INT64_MAX = 2**63 - 1
GEMM_DIM_MAX = 2**31 - 1        # §5 row 5: CUTLASS GemmCoord is int (EMULATE only)
EMULATE, DIRECT = "EMULATE", "DIRECT"


class Desc:
    """A strided tensor descriptor (§6 imd_tensor_t): extents, strides (elements), labels."""

    def __init__(self, extent, stride, label):
        assert len(extent) == len(stride) == len(label)
        self.extent, self.stride, self.label = list(extent), list(stride), list(label)

    def empty(self):
        return 0 in self.extent


def stride_rule(d: Desc) -> bool:
    """§5 row 4 on a non-empty tensor: over modes with extent > 1 sorted by |stride|,
    each |s_(t)| > sum_{u<t} |s_(u)| (e_(u) - 1)."""
    below = 0
    for a, _, e in sorted((abs(s), p, e) for p, (e, s) in enumerate(zip(d.extent, d.stride))
                          if e > 1):
        if a <= below:
            return False
        below += a * (e - 1)
    return True


def _radix_vec(modes, X):
    """§5 step 10: v[q] = sum_j s_X(mode_j) * digit_j(q), mixed radix, first mode fastest."""
    size = math.prod(e for _, e, _ in modes)
    v = [0] * size
    digits = [0] * len(modes)
    cur = 0
    for q in range(size):
        v[q] = cur
        for j, (_, e, st) in enumerate(modes):        # odometer increment
            digits[j] += 1
            cur += st[X]
            if digits[j] < e:
                break
            cur -= st[X] * e
            digits[j] = 0
    return v


class LazyVecs(Mapping):
    """The nine scatter vectors of a GemmView, each built by _radix_vec on first access
    and cached (lower(..., build_vec=False)). Holds only the fused modes, O(rank), so a
    status / lowering-only use of a legal 2^31-row descriptor costs no vector memory.
    The vectors, once built, are the same lists the eager lowering builds."""

    def __init__(self, modes, where):
        self._modes, self._where, self._cache = modes, where, {}

    def __getitem__(self, name):
        if name not in self._cache:
            X, g = self._where[name]                   # KeyError for an unknown name
            self._cache[name] = _radix_vec(self._modes[g], X)
        return self._cache[name]

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)

    def built(self):
        """Names of the vectors materialized so far."""
        return set(self._cache)


class GemmView:
    """Result of lower(): the b x (m x n x k) GEMM view and its scatter vectors.

    order[g] is the canonical label order of group g in {I, J, K, L} (first fastest);
    stride[X][label] the merged strides in the POST-SWAP frame (X in A, B, C; 0 where the
    label is absent: the stride-0 lowering of step 6); vec[name] the int64 scatter vectors
    A_row, A_k, A_bat, B_k, B_col, B_bat, C_row, C_col, C_bat (post-swap A and B)."""

    VEC_ORDER = ("A_row", "A_k", "A_bat", "B_k", "B_col", "B_bat", "C_row", "C_col", "C_bat")

    def vec_region(self):
        """The §4a vec region contents in order (A: row, k, batch; B: k, col, batch; C: row,
        col, batch)."""
        out = []
        for name in self.VEC_ORDER:
            out += self.vec[name]
        return out

    def empty_output(self):
        return self.b == 0 or self.m == 0 or self.n == 0


def lower(A: Desc, B: Desc, C: Desc, mode=EMULATE, build_vec=True, orders=None) -> GemmView:
    """DESIGN §5 lowering with the legality rows 1-5 in precedence order (rows 6-7 are
    execute-time). Raises InvalidValue / NotSupported.

    build_vec=True (default) builds the nine scatter vectors eagerly as lists (2m + 2n +
    2k + 3b entries; tens of GB for a legal 2^31-row descriptor). build_vec=False
    makes v.vec a LazyVecs: every status, dimension, order and fusion result is the same,
    lowering costs O(rank) time and memory, and a vector is built (identically) only when
    it is read.

    orders (DESIGN v1.7 §6b.2, packed operands): {class: [labels]} replaces the canonical
    order of the given POST-SWAP classes (a permutation of the class's kept labels): the
    orders a plan adopts from its packs. Fusion (step 9) is then applied in that order."""
    T = {"A": A, "B": B, "C": C}
    for d in T.values():
        if any(e < 0 for e in d.extent):
            raise InvalidValue("negative extent")
    ext, members, ms, first = {}, {}, {}, {}
    pos = 0
    for X in ("A", "B", "C"):                          # steps 1-2
        d = T[X]
        for e, s, l in zip(d.extent, d.stride, d.label):
            if ext.setdefault(l, e) != e:
                raise InvalidValue(f"row 1: extent mismatch for label {l}")
            members.setdefault(l, []).append(X)
            ms[(X, l)] = ms.get((X, l), 0) + s
            first.setdefault(l, pos)
            pos += 1
    if len(set(C.label)) != len(C.label):
        raise NotSupported("row 2: repeated label in C")
    for l, mem in members.items():
        if set(mem) == {"C"} and ext[l] > 1:
            raise NotSupported("row 3: label only in C with extent > 1")
    if not C.empty() and not stride_rule(C):
        raise InvalidValue("row 4: C fails the sorted-|stride| rule")
    if any(abs(v) > INT64_MAX for v in ms.values()):
        raise InvalidValue("row 5: merged diagonal stride")
    for d in T.values():
        if not d.empty() and sum(abs(s) * (e - 1) for e, s in zip(d.extent, d.stride)) > INT64_MAX:
            raise InvalidValue("row 5: span")
    cls = {}
    for l, mem in members.items():                     # steps 4-7
        s = set(mem)
        cls[l] = ("L" if s == {"A", "B", "C"} else "I" if s == {"A", "C"} else
                  "J" if s == {"B", "C"} else "I" if s == {"C"} else "K")
    size = {g: math.prod(ext[l] for l in cls if cls[l] == g and ext[l] != 1) for g in "IJKL"}
    if any(v > INT64_MAX for v in size.values()):
        raise InvalidValue("row 5: extent product")
    stride = {X: {l: ms.get((X, l), 0) for l in ext} for X in "ABC"}
    # step 8, coalescing: C's smallest-|stride| mode with extent > 1 in I -> swap roles
    swapped = False
    if not C.empty():
        kept = [(abs(s), p) for p, (e, s) in enumerate(zip(C.extent, C.stride)) if e > 1]
        if kept:
            swapped = cls[C.label[min(kept)[1]]] == "I"
    if swapped:
        stride["A"], stride["B"] = stride["B"], stride["A"]
        cls = {l: {"I": "J", "J": "I"}.get(c, c) for l, c in cls.items()}
        size["I"], size["J"] = size["J"], size["I"]
    v = GemmView()
    v.m, v.n, v.k, v.b = size["I"], size["J"], size["K"], size["L"]
    v.swapped = swapped
    for x, y in ((v.m, v.k), (v.k, v.n), (v.m, v.n)):
        if v.b * x * y > INT64_MAX:
            raise InvalidValue("row 5: operand size")
    if mode == EMULATE and not v.empty_output() and v.k >= 2 and \
            (v.m > GEMM_DIM_MAX or -(-v.n // 16) * 16 > GEMM_DIM_MAX):
        raise InvalidValue("row 5: m or n_hat > 2^31-1 (EMULATE, k >= 2)")
    # step 8: canonical orders (post-swap frame, see ORACLE-CHOICE above)
    big, oth = ("A", "B") if v.m >= v.n else ("B", "A")

    def a(X, l):
        return abs(stride[X][l])

    keys = {"I": lambda l: (a("C", l), a("A", l), first[l]),
            "J": lambda l: (a("C", l), a("B", l), first[l]),
            "L": lambda l: (a("C", l), a("A", l), a("B", l), first[l]),
            "K": lambda l: (a(big, l), a(oth, l), first[l])}
    v.order = {g: sorted((l for l in cls if cls[l] == g and ext[l] != 1), key=keys[g])
               for g in "IJKL"}
    for g, lst in (orders or {}).items():              # §6b.2: adopted orders
        assert sorted(lst) == sorted(v.order[g]), f"adopted {g} order {lst} vs {v.order[g]}"
        v.order[g] = list(lst)
    v.ext, v.stride, v.first = ext, stride, first
    # step 9: within each group, in canonical order, merge consecutive modes g, g' iff
    # s_X(g') = s_X(g) e(g) for EVERY tensor X carrying the group (stride-0 carriers too).
    # A tensor that does not carry the group has stride 0 on both, where the test holds
    # trivially, so testing all three tensors is the rule.
    v.fused = {}
    for g in "IJKL":
        fused = []                                     # [labels, extent, {X: stride}]
        for l in v.order[g]:
            if fused and all(stride[X][l] == fused[-1][2][X] * fused[-1][1] for X in "ABC"):
                fused[-1][0].append(l)
                fused[-1][1] *= ext[l]
            else:
                fused.append([[l], ext[l], {X: stride[X][l] for X in "ABC"}])
        v.fused[g] = fused
    v.vec = {}
    if not v.empty_output():
        modes = {g: [(tuple(ls), e, st) for ls, e, st in v.fused[g]] for g in "IJKL"}
        where = {"A_row": ("A", "I"), "A_k": ("A", "K"), "A_bat": ("A", "L"),
                 "B_k": ("B", "K"), "B_col": ("B", "J"), "B_bat": ("B", "L"),
                 "C_row": ("C", "I"), "C_col": ("C", "J"), "C_bat": ("C", "L")}
        if build_vec:
            for name, (X, g) in where.items():
                v.vec[name] = _radix_vec(modes[g], X)
        else:
            v.vec = LazyVecs(modes, where)
    return v


class Tensor:
    """Tensor data for contract_tensor: a flat list of floats `buf`, the index `base` of
    the element with all mode indices 0 (strides may be negative), and a Desc."""

    def __init__(self, buf, base, desc: Desc):
        self.buf, self.base, self.desc = buf, base, desc


def contract_tensor(A: Tensor, B: Tensor, C: Tensor, alpha: float, beta: float, N: int,
                    rounding: str = "RN", g: int | None = None, mode=EMULATE,
                    chunk: int = K_C_ALIGNED, trace: dict | None = None, orders=None,
                    packs=None, k_align=None):
    """C[I,J,L] <- alpha sum_K A[I,K,L] B[K,J,L] + beta C[I,J,L] (DESIGN §1) through the
    §5 lowering. Returns a new flat list for C's buffer (elements C does not address are
    copied unchanged). Dispatch (§3.7, §4):
      SCALE  alpha = 0 or k = 0: C <- beta == 0 ? +0 : RN(beta C); A, B not read;
      DIRECT K-empty (k = 1 after lowering) or mode DIRECT: native fma dot, acc = +0;
      GEMM   otherwise: contract() per batch lambda on the gathered canonical matrices.
    `trace` receives view, kind and, per batch of the GEMM path, contract()'s trace.
    DESIGN v1.7 §6b: `orders` = adopted canonical orders (lower(orders=...)); `packs` =
    (packed A or None, packed B or None), each a PackedOperand of the CALLER's operand:
    the plan's lowering adopts the packs' orders (lower_packed: PackMismatch on a key
    mismatch) and the emulated path reads sigma, the flags and the residues from the packed
    bytes (A / B themselves only on the native path of flagged elements).
    §4b: k_align = the plan's K-alignment policy or forced A (None: K_ALIGN_POLICY); it changes
    no value (the in-line path is values-only), only which packs are compatible."""
    if not (N_MIN <= N <= N_MAX):
        raise InvalidValue(f"N={N}")
    if rounding not in ("RN", "TRUNC") or mode not in (EMULATE, DIRECT):
        raise InvalidValue("option")
    if not (g in (None, 0) or 1 <= g <= N):
        raise InvalidValue(f"g={g}")
    info = None
    if packs is not None and any(pk is not None for pk in packs):
        assert orders is None, "a packed plan adopts the packs' orders"
        view, info = lower_packed(A.desc, B.desc, C.desc, mode,
                                  tuple(pk.desc if pk is not None else None for pk in packs),
                                  N, rounding,
                                  fmts=tuple(pk.fmt if pk is not None else None for pk in packs),
                                  k_align=k_align)
    else:
        view = lower(A.desc, B.desc, C.desc, mode, orders=orders)
    out = list(C.buf)
    if trace is not None:
        trace.update(view=view, batches=[], kind="NOOP", packed=info)
    if view.empty_output():
        return out
    m, n, k, b, vec = view.m, view.n, view.k, view.b, view.vec
    # plan time: §3.2 RN acceptance applies only when the emulated GEMM would run
    if mode == EMULATE and k > 1:
        Plan(N, k, rounding, g, chunk)
    opA, opB = (B, A) if view.swapped else (A, B)     # post-swap A' is the caller's B
    reads_ab = alpha != 0.0 and k != 0
    kind = "SCALE" if not reads_ab else "DIRECT" if (mode == DIRECT or k == 1) else "GEMM"
    if trace is not None:
        trace["kind"] = kind

    def gather_a(lam):
        return [[opA.buf[opA.base + vec["A_bat"][lam] + vec["A_row"][i] + vec["A_k"][h]]
                 for h in range(k)] for i in range(m)]

    def gather_b(lam):
        return [[opB.buf[opB.base + vec["B_bat"][lam] + vec["B_k"][h] + vec["B_col"][j]]
                 for j in range(n)] for h in range(k)]

    def c_off(lam, i, j):
        return C.base + vec["C_bat"][lam] + vec["C_row"][i] + vec["C_col"][j]

    for lam in range(b):
        Cl = ([[C.buf[c_off(lam, i, j)] for j in range(n)] for i in range(m)]
              if beta != 0.0 else None)                 # C is not read when beta = 0
        if kind == "SCALE":
            res = [[(0.0 if beta == 0.0 else mul(beta, Cl[i][j])) for j in range(n)]
                   for i in range(m)]
        elif kind == "DIRECT":
            Al, Bl = gather_a(lam), gather_b(lam)
            res = [[None] * n for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    D = native_dot(Al[i], [Bl[h][j] for h in range(k)])
                    E = D if alpha == 1.0 else mul(alpha, D)
                    res[i][j] = E if beta == 0.0 else fma(beta, Cl[i][j], E)
        else:
            t = {} if trace is not None else None
            ext = {"A": None, "B": None}
            if info is not None:      # §6b: the packed post-swap operands' sigma, flags, planes
                for x, role in enumerate(info["roles"]):
                    if packs[x] is not None:
                        ext[role] = pack_rows(info["views"][x], packs[x].data, lam,
                                              m if role == "A" else n)
            res = contract(gather_a(lam), gather_b(lam), Cl, alpha, beta, N, rounding, g,
                           chunk, m=m, n=n, k=k, trace=t, ext_a=ext["A"], ext_b=ext["B"])
            if trace is not None:
                trace["batches"].append(t)
        for i in range(m):
            for j in range(n):
                out[c_off(lam, i, j)] = res[i][j]
    return out


# ---------------------------------------------------------------------------------------
# §6b Packed operands (DESIGN v1.7, ROADMAP D1)
# ---------------------------------------------------------------------------------------
#
# A packed operand X: per row (lambda, rho) its sigma (§3.2) and flag (§3.1) and the residues
# of ALL N moduli over the whole canonical K (§3.3, §3.4), in the §6b.4 byte layout (format
# 1). pack() produces exactly those bytes; a plan built against a pack adopts the pack's
# canonical orders of K, L and X's row class (lower_packed); the packed contraction reads
# the packed sigma, flags and residues (contract_tensor(packs=...)). Literal transcription
# of DESIGN §6b.1-§6b.5; the choices §6b leaves to the implementation are marked
# ORACLE-CHOICE and are the library's (src/host/pack.cpp).

PACK_FORMAT_A16, PACK_FORMAT_A64 = 1, 2   # §6b.4 FORMAT word: k_hat = ceil(k)_16 / ceil(k)_64
PACK_FORMAT_ALIGN = {PACK_FORMAT_A16: 16, PACK_FORMAT_A64: 64}
PACK_FORMAT = PACK_FORMAT_A16              # the newest format written (stage 2: 2)
ORDER_AUTO, ORDER_GIVEN = "AUTO", "GIVEN"


def pack_format_for(k, policy=None):
    """§4b / §6b.4: the format of a pack (or of a plan's slot key) of contraction length k:
    1 for A = 16, 2 for A = 64, A = k_align_of(k, policy)."""
    return PACK_FORMAT_A64 if k_align_of(k, policy) == 64 else PACK_FORMAT_A16
# PACK_MISMATCH reasons (imd_lowering_t v5 pack_mismatch, immerdink_host.h IMD_PACK_MISMATCH_*)
(MISMATCH_FORMAT, MISMATCH_MODULI, MISMATCH_ROUNDING, MISMATCH_CLASSES, MISMATCH_EXTENTS,
 MISMATCH_STRIDES, MISMATCH_ORDER) = range(1, 8)
UINT64_MAX = 2**64 - 1


class PackMismatch(Exception):
    """Plan-time PACK_MISMATCH (§6b.5) with its reason code (MISMATCH_*)."""

    def __init__(self, reason, msg):
        super().__init__(msg)
        self.reason = reason


class PackDesc:
    """imd_pack_desc_t (§6b.5): the operand X (a Desc), N, rounding, the order mode and the
    label lists K (contracted), L (batch) and, for GIVEN, R (rows; AUTO: None)."""

    def __init__(self, tensor: Desc, N, rounding="RN", order=ORDER_AUTO, contracted=(),
                 batch=(), rows=None):
        self.tensor, self.N, self.rounding, self.order = tensor, N, rounding, order
        self.contracted, self.batch = list(contracted), list(batch)
        self.rows = None if rows is None else list(rows)


class PackedOperand:
    """A pack (desc + the bytes imd_pack wrote + its format, §4b) for contract_tensor(packs=...).
    fmt None: the format the policy gives the pack's k (what imd_pack_create resolves)."""

    def __init__(self, desc: PackDesc, data, fmt=None):
        self.desc, self.data, self.fmt = desc, bytes(data), fmt


def _fuse_x(modes):
    """§6b.2 key part 4: fuse consecutive (extent, stride) modes over X's strides alone
    (merge iff s' = s e)."""
    out = []
    for e, st in modes:
        if out and st == out[-1][1] * out[-1][0]:
            out[-1] = (out[-1][0] * e, out[-1][1])
        else:
            out.append((e, st))
    return out


def pack_key(N, rounding, b, r, k, fused, fmt=PACK_FORMAT_A16):
    """§6b.2 compatibility key: FORMAT (§4b: 1 = A16, 2 = A64), N, rounding (RN 0, TRUNC 1),
    b, r, k, then per class L, R, K the number of fused modes and their (extent, stride in X)."""
    words = [fmt, N, 0 if rounding == "RN" else 1, b, r, k]
    for c in ("L", "R", "K"):
        words.append(len(fused[c]))
        for e, st in fused[c]:
            words += [e, st]
    return tuple(words)


def _key_kinds(key):
    """The reason code of every word of a key (EXTENTS for b, r, k, counts and extents)."""
    kinds = [MISMATCH_FORMAT, MISMATCH_MODULI, MISMATCH_ROUNDING] + [MISMATCH_EXTENTS] * 3
    p = 6
    for _ in range(3):
        cnt = key[p]
        kinds.append(MISMATCH_EXTENTS)
        kinds += [MISMATCH_EXTENTS, MISMATCH_STRIDES] * cnt
        p += 1 + 2 * cnt
    return kinds


def key_mismatch(slot_key, pack_key_):
    """0 if equal, else the reason code of the first differing word (by the slot key's
    structure; if one key is a prefix of the other, a count differed first: EXTENTS)."""
    if slot_key == pack_key_:
        return 0
    kinds = _key_kinds(slot_key)
    for i, (x, y) in enumerate(zip(slot_key, pack_key_)):
        if x != y:
            return kinds[i]
    return MISMATCH_EXTENTS


def fnv1a64(words):
    """§6b.5 key_digest: FNV-1a (64-bit) over the key words as little-endian int64 bytes."""
    h = 0xcbf29ce484222325
    for w in words:
        for byte in (w & UINT64_MAX).to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001b3) & UINT64_MAX
    return h


def pack_layout(b, r, k, N, fmt=PACK_FORMAT_A16):
    """§6b.4 format fmt (1: k_hat = ceil(k)_16; 2, §4b: ceil(k)_64), typed from DESIGN
    (independently of params.pack_layout, which test_oracle cross-checks): {region: (offset,
    bytes)} for sig, flag, res and pack_bytes; ({}, 0) for b r k = 0. InvalidValue past
    uint64."""
    if b * r * k == 0:
        return {}, 0
    kh = k_hat_of(k, PACK_FORMAT_ALIGN[fmt])
    sizes = (("sig", 2 * b * r), ("flag", b * r),
             ("res", N * b * r * kh + (-(-r // 16) * 16 - r) * kh))
    regs, off = {}, 0
    for name, nbytes in sizes:
        regs[name] = (off, nbytes)
        off += -(-nbytes // 256) * 256
        if nbytes > UINT64_MAX or off > UINT64_MAX:
            raise InvalidValue("§6b.4: pack size exceeds uint64")
    return regs, off


def pack_ws_bytes(b, r, k):
    """ORACLE-CHOICE (the library's choice, §6b.5 pack_ws_bytes "may be 0"): imd_pack keeps
    X's scatter vectors (rows, k-bar, batches; int64) in its workspace: 8 (r + k + b)
    aligned to 256, 0 for an empty pack."""
    return 0 if b * r * k == 0 else -(-8 * (r + k + b) // 256) * 256


class PackView:
    """pack_lower(): the pack's classes, sizes, canonical orders, X-only fused modes, key and
    layout (what imd_pack_layout reports, plus the orders)."""


def _T_of(N):
    P = math.prod(MODULI[:N])
    return math.isqrt((P - 1) // 2)


def pack_lower(pd: PackDesc, fmt=None) -> PackView:
    """imd_pack_layout / imd_pack_create (§6b.3, §6b.5): validate the pack descriptor and
    lower X alone. InvalidValue: N, rounding, order out of range; a listed label not in X,
    listed twice or in two lists; GIVEN with a missing row label; AUTO with rows; §5 rows 1
    and 5 on X alone (extent mismatch of a repeated label; negative extent; merged stride,
    span, class extent product or pack size overflow). NotSupported: K empty after lowering
    (k = 1), or RN with k >= 2 and kappa >= T. AUTO orders each class by (|s_X| after the
    diagonal merge, first position in X's labels). §4b: fmt = the pack's format (None: the
    policy's format for the pack's k, as imd_pack_create resolves it; 1 or 2 forced: the
    library's test hook)."""
    if not (N_MIN <= pd.N <= N_MAX):
        raise InvalidValue(f"N={pd.N}")
    if pd.rounding not in ("RN", "TRUNC") or pd.order not in (ORDER_AUTO, ORDER_GIVEN):
        raise InvalidValue("rounding / order")
    X = pd.tensor
    if any(e < 0 for e in X.extent):
        raise InvalidValue("negative extent")
    ext, ms, first = {}, {}, {}
    for p, (e, st, l) in enumerate(zip(X.extent, X.stride, X.label)):
        if ext.setdefault(l, e) != e:
            raise InvalidValue(f"row 1: extent mismatch for label {l}")
        ms[l] = ms.get(l, 0) + st
        first.setdefault(l, p)
    K, L = pd.contracted, pd.batch
    if pd.order == ORDER_AUTO:
        if pd.rows:
            raise InvalidValue("AUTO with rows")
        R = None
    else:
        R = list(pd.rows or [])
    listed = K + L + (R or [])
    if any(l not in ext for l in listed) or len(set(listed)) != len(listed):
        raise InvalidValue("a listed label not in X, listed twice or in two lists")
    if R is None:
        R = [l for l in ext if l not in K and l not in L]
    elif set(listed) != set(ext):
        raise InvalidValue("GIVEN without every row label")
    if any(abs(v) > INT64_MAX for v in ms.values()):
        raise InvalidValue("row 5: merged stride")
    if not X.empty() and sum(abs(st) * (e - 1) for e, st in zip(X.extent, X.stride)) > INT64_MAX:
        raise InvalidValue("row 5: span")
    lists = {"K": K, "L": L, "R": R}
    size = {c: math.prod(ext[l] for l in lists[c] if ext[l] != 1) for c in "KLR"}
    if any(v > INT64_MAX for v in size.values()):
        raise InvalidValue("row 5: extent product")
    v = PackView()
    v.N, v.rounding = pd.N, pd.rounding
    v.b, v.r, v.k = size["L"], size["R"], size["K"]
    v.format = pack_format_for(v.k) if fmt is None else fmt
    v.k_hat = k_hat_of(v.k, PACK_FORMAT_ALIGN[v.format])
    v.kappa = ceil_sqrt(v.k) if v.k > 0 else 0
    v.regs, v.pack_bytes = pack_layout(v.b, v.r, v.k, pd.N, v.format)   # InvalidValue on overflow
    v.pack_ws = pack_ws_bytes(v.b, v.r, v.k)
    if not any(ext[l] != 1 for l in K):
        raise NotSupported("§6b.3: K empty after lowering (every contraction is DIRECT)")
    if pd.rounding == "RN" and v.k >= 2 and v.kappa >= _T_of(pd.N):
        raise NotSupported("§6b.3: kappa >= T (RN)")
    key = (lambda l: (abs(ms[l]), first[l])) if pd.order == ORDER_AUTO else None
    v.order = {c: [l for l in (sorted(lists[c], key=key) if key else lists[c]) if ext[l] != 1]
               for c in "KLR"}
    v.ext, v.stride = ext, ms
    v.fused = {c: _fuse_x([(ext[l], ms[l]) for l in v.order[c]]) for c in "LRK"}
    v.key = pack_key(pd.N, pd.rounding, v.b, v.r, v.k, v.fused, v.format)
    v.digest = fnv1a64(v.key)
    return v


def _x_vec(fused):
    return _radix_vec([(None, e, {"X": st}) for e, st in fused], "X")


def pack(X: Tensor, pd: PackDesc, trace: dict | None = None, fmt=None) -> bytearray:
    """imd_pack (§6b.1, §6b.4): the packed bytes of X (X.desc is the operand; pd.tensor must
    describe the same element map) in format fmt (§4b: 1 = A16, 2 = A64; None = the policy's
    format for the pack's k). Every byte of [0, pack_bytes) is defined: sigma, flags,
    residues of plane l at l b r k_hat + (lambda r + rho) k_hat + k-bar, zero K padding
    k-bar in [k, k_hat), a zero tail and zero alignment gaps."""
    v = pack_lower(pd, fmt)
    out = bytearray(v.pack_bytes)
    if v.pack_bytes == 0:
        return out
    plan = Plan(pd.N, v.k, pd.rounding)
    vL, vR, vK = _x_vec(v.fused["L"]), _x_vec(v.fused["R"]), _x_vec(v.fused["K"])
    so, fo, ro = v.regs["sig"][0], v.regs["flag"][0], v.regs["res"][0]
    plane = v.b * v.r * v.k_hat
    sig, flg = [], []
    for lam in range(v.b):
        for rho in range(v.r):
            row = [X.buf[X.base + vL[lam] + vR[rho] + vK[h]] for h in range(v.k)]
            sigma, flag, *_ = scale_vector(row, plan)
            idx = lam * v.r + rho
            out[so + 2 * idx:so + 2 * idx + 2] = (sigma & 0xFFFF).to_bytes(2, "little")
            out[fo + idx] = 1 if flag else 0
            sig.append(sigma)
            flg.append(flag)
            q = [quantize(x, sigma, pd.rounding) for x in row]
            for l in range(pd.N):
                base = ro + l * plane + idx * v.k_hat
                for h, (mu, ep) in enumerate(q):
                    out[base + h] = residue(mu, ep, plan.p[l]) & 0xFF
    if trace is not None:
        trace.update(view=v, sigma=sig, flags=flg)
    return out


def pack_rows(v: PackView, data, lam, rows):
    """(sigma, flags, planes[l][rho][h]) of batch lambda's rows rho < rows of a pack, as
    contract(ext_a / ext_b) takes them (h < k; the K padding is not read)."""
    so, fo, ro = v.regs["sig"][0], v.regs["flag"][0], v.regs["res"][0]
    plane = v.b * v.r * v.k_hat
    sig, flg, planes = [], [], []
    for rho in range(rows):
        idx = lam * v.r + rho
        sig.append(int.from_bytes(data[so + 2 * idx:so + 2 * idx + 2], "little", signed=True))
        flg.append(data[fo + idx] != 0)
    for l in range(v.N):
        pl = []
        for rho in range(rows):
            base = ro + l * plane + (lam * v.r + rho) * v.k_hat
            pl.append([x - 256 if x > 127 else x for x in data[base:base + v.k]])
        planes.append(pl)
    return sig, flg, planes


def _roles(view, x):
    """Post-swap role ("A" / "B") of the caller's operand x (0 = A, 1 = B), and its row class."""
    role = ("A", "B")[x] if not view.swapped else ("B", "A")[x]
    return role, ("I" if role == "A" else "J")


def pack_desc_for(A: Desc, B: Desc, C: Desc, mode, operand, N, rounding="RN") -> PackDesc:
    """imd_pack_desc_for (§6b.5): GIVEN with the classes and orders v1.5's unpacked plan of
    (A, B, C) uses for the caller's operand (0 = A, 1 = B). ORACLE-CHOICE (the library's):
    X's extent-1 labels (absent from the plan's orders) are appended to their class's list by
    membership (not in C: K; in C and the partner: L; else rows); a partner-only K label of
    the plan is not in X and is not listed (such a plan never matches: CLASSES)."""
    v = lower(A, B, C, mode, build_vec=False)
    X, P = (A, B) if operand == 0 else (B, A)
    role, rowcls = _roles(v, operand)
    xl = list(dict.fromkeys(X.label))
    K = [l for l in v.order["K"] if l in xl]
    L = [l for l in v.order["L"] if l in xl]
    R = [l for l in v.order[rowcls] if l in xl]    # (an extent-0 C-only label is in I, §5 7)
    for l in xl:
        if l in K or l in L or l in R:
            continue
        (K if l not in C.label else L if l in P.label else R).append(l)
    return PackDesc(X, N, rounding, ORDER_GIVEN, K, L, R)


def lower_packed(A: Desc, B: Desc, C: Desc, mode, packs, N, rounding="RN", build_vec=True,
                 fmts=(None, None), k_align=None):
    """imd_plan_create_packed's plan-time checks after §5 rows 1-5 (lower) and before the
    §3.2 RN acceptance (§6b.5): per packed caller operand (A, then B) the pack descriptor's
    own status (pack_lower), then PACK_MISMATCH reasons MODULI, ROUNDING, CLASSES (the
    pack's labels map onto X's by mode position, a consistent bijection; the plan's K, L and
    X's row class equal the pack's classes; a partner-only K label is CLASSES); both packed:
    ORDER if their K or L orders differ; then the plan ADOPTS the packs' orders (K, L, X's
    row class) and each slot key, computed from the adopted lowering restricted to X, must
    equal the pack's key (EXTENTS / STRIDES: the first differing word). Returns (view, info).
    §4b: fmts = each pack's format (None: the policy's for its k); the slot key's FORMAT word
    is the PLAN's format, of its own resolved A (k_align: a policy or a forced 16 / 64, None =
    K_ALIGN_POLICY), so a pack of another format is FORMAT (impossible under AUTO: both
    resolve A from the same k; reachable through the test hook)."""
    v0 = lower(A, B, C, mode, build_vec=False)
    pvs = [pack_lower(pd, f) if pd is not None else None for pd, f in zip(packs, fmts)]
    adopted = [None, None]
    roles = [_roles(v0, x)[0] for x in (0, 1)]
    for x, (Xd, pd, pv) in enumerate(zip((A, B), packs, pvs)):
        if pd is None:
            continue
        if pv.N != N:
            raise PackMismatch(MISMATCH_MODULI, f"operand {x}: pack N {pv.N} != {N}")
        if pv.rounding != rounding:
            raise PackMismatch(MISMATCH_ROUNDING, f"operand {x}: pack rounding")
        if len(pd.tensor.label) != len(Xd.label):
            raise PackMismatch(MISMATCH_CLASSES, f"operand {x}: rank differs")
        mp = {}
        for lp, lx in zip(pd.tensor.label, Xd.label):
            if mp.setdefault(lp, lx) != lx:
                raise PackMismatch(MISMATCH_CLASSES, f"operand {x}: labels do not map")
        if len(set(mp.values())) != len(mp):
            raise PackMismatch(MISMATCH_CLASSES, f"operand {x}: labels do not map")
        role, rowcls = _roles(v0, x)
        if any(l not in Xd.label for l in v0.order["K"]):
            raise PackMismatch(MISMATCH_CLASSES, f"operand {x}: partner-only K label")
        o = {"K": [mp[l] for l in pv.order["K"]], "L": [mp[l] for l in pv.order["L"]],
             rowcls: [mp[l] for l in pv.order["R"]]}
        for c in ("K", "L", rowcls):
            if set(o[c]) != set(v0.order[c]):
                raise PackMismatch(MISMATCH_CLASSES, f"operand {x}: class {c} differs")
        adopted[x] = o
    if adopted[0] is not None and adopted[1] is not None and \
            (adopted[0]["K"] != adopted[1]["K"] or adopted[0]["L"] != adopted[1]["L"]):
        raise PackMismatch(MISMATCH_ORDER, "the two packs' K or L orders differ")
    orders = {}
    for o in adopted:
        if o is not None:
            for c, lst in o.items():
                orders.setdefault(c, lst)
    v = lower(A, B, C, mode, build_vec=build_vec, orders=orders)
    keys = [None, None]
    for x in (0, 1):
        if pvs[x] is None:
            continue
        role, rowcls = _roles(v, x)
        st = v.stride[role]
        fused = {c: _fuse_x([(v.ext[l], st[l]) for l in v.order[g]])
                 for c, g in (("L", "L"), ("R", rowcls), ("K", "K"))}
        keys[x] = pack_key(N, rounding, v.b, v.m if role == "A" else v.n, v.k, fused,
                           pack_format_for(v.k, k_align))
        reason = key_mismatch(keys[x], pvs[x].key)
        if reason:
            raise PackMismatch(reason, f"operand {x}: slot key differs (reason {reason})")
    packed_roles = {roles[x] for x in (0, 1) if pvs[x] is not None}
    return v, {"views": pvs, "keys": keys, "roles": roles, "packed_roles": packed_roles,
               "orders": orders}


def einsum_addresses(A: Desc, B: Desc, C: Desc):
    """Brute-force sorted list of the einsum's (C, A, B) element-offset triples: one per
    assignment of every label (a repeated label takes one index: diagonal). Tests only."""
    labels = []
    for d in (A, B, C):
        for l, e in zip(d.label, d.extent):
            if l not in [x for x, _ in labels]:
                labels.append((l, e))
    out = []
    for idx in itertools.product(*[range(e) for _, e in labels]):
        asg = {l: i for (l, _), i in zip(labels, idx)}
        out.append(tuple(sum(s * asg[l] for s, l in zip(d.stride, d.label)) for d in (C, A, B)))
    return sorted(out)


def view_addresses(view: GemmView):
    """Sorted (C, A, B) offset triples generated by the GEMM view (caller's A/B frame)."""
    vec = view.vec
    out = []
    for lam in range(view.b):
        for i in range(view.m):
            for j in range(view.n):
                for h in range(view.k):
                    a = vec["A_bat"][lam] + vec["A_row"][i] + vec["A_k"][h]
                    bb = vec["B_bat"][lam] + vec["B_k"][h] + vec["B_col"][j]
                    c = vec["C_bat"][lam] + vec["C_row"][i] + vec["C_col"][j]
                    out.append((c, bb, a) if view.swapped else (c, a, bb))
    return sorted(out)


if __name__ == "__main__":
    print("moduli:", MODULI_SOURCE)
    print("K_c =", K_C, " K_c' =", K_C_ALIGNED)
    for N in (2, 8, 14, 16, 17, 20):
        pl = Plan(N, 1)
        print(f"N={N:2d} log2P={math.log2(pl.P):7.2f} T~2^{math.log2(pl.T):6.2f} L_w={pl.L_w}")
