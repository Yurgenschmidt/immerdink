"""The K5 exact FP64 finish (issue 21, "K5 lever 3"): a bit-exact CPU model of the instruction sequence, for gate 1 of the spec in
https://github.com/Yurgenschmidt/immerdink/issues/21#issuecomment-6750980047. CPU only; `python3 spec/test_crt_f64_ro.py` runs the gate.

What K5 does, and what changes
------------------------------
K5 is the last kernel of a contraction. For one output element it gets the N residues W_l = z mod p_l (one byte per modulus, from K4) and must return
D = RN(z 2^e): the exact integer z (the "CRT value", centred in [-P/2, P/2), P = p_1 ... p_N) rebuilt by the Chinese remainder theorem, times the
unscaling power 2^e, rounded once to binary64 (RN = round to nearest, ties to even). The reference is spec/oracle.py (`crt`, then `round_dyadic`).
spec/crt_f64.py (the shipped "limb CRT") already forms the CRT sum in the FP64 pipe:

    s_j = sum_l W_l k_lj        exact binary64 integers below 2^53 (K_l = q_l M_l cut into J limbs of w = 40 bits, k_lj < 2^40)
    S   = sum_j s_j 2^(40 j)     congruent to z modulo P, 0 <= S < tmax P with tmax = 255 N

and then leaves the FP64 pipe: an F2I for the quotient floor(S / P), F2I conversions of the limb remainders, a 40-to-32-bit repack into integer words,
one conditional subtraction of P, the centring and an integer round-to-binary64 (finish_f1_fast, or crt_finish + round_scaled). On Hopper the
conversions run on the XU pipe at 1/8 rate and co-limit K5. This module replaces that integer tail by FP64 arithmetic, keeping every result bit-identical:

 1. Quotient. t = RN(Sd Pinv) as an integer kept in a double, by one fma and one add with the magic constant C = 1.5 * 2^52:
        t = fma(Sd, Pinv, C) - C            (no F2I; t is the integer nearest to Sd Pinv, ties to even, see lemma 2)
    Sd = fl(S) and Pinv = fl(1 / P) are the shipped quantities. Unlike the shipped floor(Sd Pinv - 2^-30), t is the *centred* quotient, so the remainder
    R = S - t P is already the centred z (lemma 4), with no conditional subtraction and no centring step.
 2. Limbs. With the scalings folded into the constants (k'_lj = k_lj 2^(40 j), P'_j = P_j 2^(40 j), both exact doubles):
        s'_j = sum_l W_l k'_lj = s_j 2^(40 j),   r'_j = fma(-t, P'_j, s'_j) = (s_j - t P_j) 2^(40 j)     exact (|s_j - t P_j| < 2^53)
    so R = sum_j r'_j exactly, as a sum of J doubles.
 3. Sum. z~ = RN(r'_{J-1} + ... + r'_0), correctly rounded, without any integer step:
        J = 1: z~ = r'_0 (exact)        J = 2: z~ = r'_1 + r'_0 (one rounding)
        J = 3: the Boldo-Melquiond correctly rounded sum of three doubles (below)
        J = 4 (N = 16, 17): r'_3 + r'_2 is exact (lemma 6), then the J = 3 sum of (r'_3 + r'_2, r'_1, r'_0).
    The three-term sum (S. Boldo, G. Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms using rounding to odd",
    IEEE Trans. Computers 57(4):462-471, 2008, doi:10.1109/TC.2007.70819, https://hal.inria.fr/inria-00080427):
        (uh, ul) = TwoSum(r'_1, r'_0);  (th, tl) = TwoSum(r'_2, uh);  v = RO(tl + ul);  z~ = RN(th + v)
    TwoSum is Knuth's six-operation error-free sum (a + b = s + err exactly). RO is rounding to odd: the representable neighbour with an odd last
    significand bit, unless the sum is exact. The GPU has no RO mode, so it is emulated exactly as the device would do it (`ro_add`):
        s = RN(a + b);  err = the TwoSum error of (a, b);  if err != 0 and the last significand bit of s is 0: move s one ulp toward err
    (one integer +-1 on the 64-bit pattern: +1 when err has the sign of s, -1 otherwise). Why RO: rounding v to nearest would round twice (a sticky
    bit far below could be lost and turn a non-tie into a tie); RO keeps a sticky bit in v's last place, and BM prove RN(th + RO(tl + ul)) = RN(a + b + c).
 4. Scale. D = z~ 2^e by adding e to the exponent field of z~ (one integer add on the high word, as F1 does; equal to a DMUL by 2^e when 2^e is a
    normal double). Exact whenever the result is a normal double; z~ = 0 gives D = +0.
 5. Fallback. These elements take today's integer finish (spec/crt_f64.py `crt_centred` + `round_dyadic`) instead:
        |z~| >= thr                                     the centring ambiguity; the spec's thr = RD(P/2 - 2^-38 P) (any double thr <= RN(P/2) is sound)
        z~ != 0 and the biased exponent of z~ plus e is outside [1, 2046]    the result would be subnormal (or zero) or overflow
    Both are tested on z~ before the scaling (lemma 7 shows why testing the scaled result is wrong).

Why it is exact (the proof obligation of the spec)
--------------------------------------------------
Notation: z is the oracle's value, the integer in [-P/2, P/2) congruent to S (oracle.crt centres x in [0, P) at H = P/2 - 1: x = P/2 maps to -P/2);
t* = (S - z) / P is the integer with S - t* P = z; u = 2^-53.

  Lemma 1 (sums). s_j, hence s'_j, are exact: shipped certificate (params.cert_crt_f64); the folding multiplies by powers of two.
  Lemma 2 (magic quotient). 0 <= Sd Pinv < 2^13, so the exact value Sd Pinv + C lies in [2^52, 2^53), where the doubles are the integers: the fma
    returns C + RNint(Sd Pinv) (ties to even: C is even), and subtracting C is exact (Sterbenz). So t = RNint(Sd Pinv), from the unrounded product.
  Lemma 3 (quotient error). |Sd Pinv - S / P| <= delta_N := tmax ((1 + u)^(J - 1) (1 + eta) - 1), eta = |Pinv P - 1| computed exactly from the table
    constant (Sd is J - 1 roundings of a sum of nonnegative terms). delta_N < 2^-39 for every N the finish covers (`delta` below; 2^-39.2 at N = 17). Hence t is one of t* - 1, t*, t* + 1,
    0 <= t <= tmax, and r'_j is exact (the shipped bound (tmax + 1) P_j < 2^53 covers it).
  Lemma 4 (centring). R := sum_j r'_j = S - t P exactly. If t = t*, R = z. If t = t* + k with k = -1 or +1, R = z - k P.
  Lemma 5 (the misround window, the spec obligation). The quotient misrounds (t != t*) exactly when R lies outside [-P/2, P/2):
        t != t*   <=>   R >= P/2  or  R < -P/2.
    (=>) k = -1: R = z + P >= -P/2 + P = P/2. k = +1: R = z - P <= P/2 - 1 - P < -P/2. (<=) If t = t*, R = z lies in [-P/2, P/2).
    Moreover a misround needs S / P within delta_N of a half-integer, so it only happens for |z| >= P/2 - delta_N P, and then
    P/2 <= |R| <= P/2 + delta_N P. So the window is a band of relative width delta_N < 2^-39 at each end of the range.
  Lemma 6 (the sum). z~ = RN(R). J <= 2: trivial. J = 3: Boldo-Melquiond (proved for binary formats without underflow or overflow; here all values
    are integers below 2^170: none occurs). J = 4: R - r'_1 - r'_0 = r'_3 + r'_2 has magnitude at most P/2 + delta_N P + (2^53 - 1)(2^40 + 1),
    which is below 2^133 for N <= 17 (P < 2^133), and it is a multiple of 2^80, so it is a double and the add is exact (`top_merge_exact`); N >= 18 would need a
    four-term sum and is refused here (those N keep the integer finish).
  Lemma 7 (scale). If 2^-1022 <= |z~ 2^e| < 2^1024 then RN(z 2^e) (the oracle, with gradual underflow) = z~ 2^e. Above 2^-1022 both round on the
    53-bit grid and RN commutes with the power-of-two scaling. The only other case is z 2^e just below 2^-1022 while z~ 2^e = 2^-1022 (rounded up to the
    boundary): then |z 2^e| >= 2^-1022 - 2^-1076, within a quarter of the subnormal spacing 2^-1074 of 2^-1022, so the oracle gives 2^-1022 too.
    The test must be on z~'s exponent: a z~ 2^e below 2^-1022 rounds a second time in a DMUL, and can land on 2^-1022 (a normal double) with the wrong
    value: a predicate on the product's exponent would miss it (test_crt_f64_ro.py builds that case).
  Theorem. If the fallback does not fire, D equals the oracle bit for bit. Proof: the centring clause fires on every R outside [-P/2, P/2): RN is
    monotone, so |R| >= P/2 gives |z~| = RN(|R|) >= RN(P/2) >= thr. So t = t* (lemma 5), R = z (lemma 4), z~ = RN(z) (lemma 6), D = RN(z 2^e) (lemma 7).
    Lemma 5 is an equality on the exact R; on the computed z~ the minimal sound clause, |z~| >= RN(P/2), catches the misrounds and, besides them,
    only the correct elements with z within half an ulp below P/2 or z = -P/2. The spec's thr = RD(P/2 - 2^-38 P) adds the band of relative width
    2^-38 below P/2: 2^-37 of uniformly distributed z (both ends), harmless for the speed.

Everything here is Python floats (IEEE binary64 with round to nearest even, as the GPU's DADD / DMUL / DFMA with _rn) and exact integers for the
checks. The model functions name the device instruction each step stands for.
"""
import math
import os
import struct
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_compat  # noqa: E402,F401  (math.fma before Python 3.13)
import crt_f64 as C  # noqa: E402
import oracle as o  # noqa: E402

MAGIC = 6755399441055744.0      # 1.5 * 2^52: x + MAGIC rounds x to an integer for |x| < 2^51
assert MAGIC == 1.5 * 2.0 ** 52
U = Fraction(1, 1 << 53)        # the unit roundoff of binary64
THR_SPEC = "spec"               # RD(P/2 - 2^-38 P), the issue's predicate
THR_MIN = "min"                 # RN(P/2), the smallest sound threshold (lemma 5)


def f2b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def b2f(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def two_sum(a, b):
    """Knuth's TwoSum (6 DADD): s = RN(a + b) and err with a + b = s + err exactly (no overflow)."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def ro_add(a, b, mut=0):
    """RO(a + b) emulated as the device would: RN add, its TwoSum error, then if the error is nonzero and the significand's last bit is 0, one +-1
    on the bit pattern toward the error (+1 when err has the sign of s: the magnitude grows). `mut` plants a bug for the tests:
    1 plain RN (no RO), 2 nudge whatever the parity, 3 nudge away from the error."""
    s, err = two_sum(a, b)
    if mut == 1 or err == 0.0:
        return s
    bs = f2b(s)
    if (bs & 1) == 0 or mut == 2:
        assert s != 0.0                          # no underflow: a nonzero error needs |a + b| >= 2^53 here
        up = (err > 0.0) == (s > 0.0)
        if mut == 3:
            up = not up
        bs += 1 if up else -1
        s = b2f(bs)
    return s


def sum3_bm(a, b, c, ro_mut=0):
    """Boldo-Melquiond: RN(a + b + c) for the limbs a = r'_2 (top), b = r'_1, c = r'_0."""
    uh, ul = two_sum(b, c)
    th, tl = two_sum(a, uh)
    v = ro_add(tl, ul, ro_mut)
    return th + v


def ro_exact(x: Fraction):
    """Reference RO (round to odd, binary64, no under/overflow) of an exact rational: for checking ro_add."""
    if x == 0:
        return 0.0
    f = float(x)                                   # RN
    if Fraction(f) == x:
        return f
    lo, hi = (f, math.nextafter(f, math.inf)) if Fraction(f) < x else (math.nextafter(f, -math.inf), f)
    return lo if f2b(abs(lo)) & 1 else hi


def dmul_pow2(z, e):
    """RN(z 2^e) with gradual underflow and overflow to infinity: what a DMUL by an exact 2^e does (tests only: the wrong variant of lemma 7)."""
    try:
        return math.ldexp(z, e)
    except OverflowError:
        return math.copysign(math.inf, z)


class F64CrtRO(C.F64Crt):
    """Plan-time constants and the per-element sequence of the RO finish for N moduli. Raises oracle.NotSupported for N whose limb count the
    finish does not cover (J = 4 without an exact top merge: N >= 18)."""

    def __init__(self, N, thr=THR_SPEC):
        super().__init__(N)
        J, w, P = self.J, self.w, self.P
        if J == 4 and not self.top_merge_exact():
            raise o.NotSupported(f"N={N}: J = 4 and r'_3 + r'_2 is not exact (a four-term sum would be needed)")
        if J > 4:
            raise o.NotSupported(f"N={N}: J = {J}")
        # folded constants (exact: a 40-bit integer times a power of two)
        self.kf = [[float(self.k[l][j] << (w * j)) for j in range(J)] for l in range(N)]
        self.Pf = [float(self.Pj[j] << (w * j)) for j in range(J)]
        assert all(int(self.kf[l][j]) == self.k[l][j] << (w * j) for l in range(N) for j in range(J))
        assert all(int(self.Pf[j]) == self.Pj[j] << (w * j) for j in range(J))
        self.delta = self.quotient_error()
        assert self.delta < Fraction(1, 2), "the quotient must stay within one of t*"
        self.half_rn = float(P // 2)                                       # RN(P/2): P is even, int -> float rounds to nearest even
        self.thr_spec = self._round_down(Fraction(P, 2) - Fraction(P, 1 << 38))
        if thr == THR_SPEC:
            self.thr = self.thr_spec
        elif thr == THR_MIN:
            self.thr = self.half_rn
        else:
            self.thr = float(thr)

    @staticmethod
    def _round_down(x: Fraction):
        f = float(x)
        return f if Fraction(f) <= x else math.nextafter(f, 0.0)

    def quotient_error(self):
        """delta_N of lemma 3 (exact rational): |Sd Pinv - S/P| <= tmax ((1 + u)^(J-1) (1 + eta) - 1)."""
        eta = abs(Fraction(self.Pinv) * self.P - 1)
        return self.tmax * ((1 + U) ** (self.J - 1) * (1 + eta) - 1)

    def top_merge_exact(self):
        """Lemma 6, J = 4: |r'_3 + r'_2| <= P/2 + delta P + (2^53 - 1)(2^40 + 1) must be below 2^133 (a multiple of 2^80 below 2^133 is a double)."""
        w = self.w
        delta = self.quotient_error()
        bound = Fraction(self.P, 2) + delta * self.P + ((1 << 53) - 1) * ((1 << w) + 1)
        return bound < (1 << (53 + 2 * w))

    # ---- the per-element sequence --------------------------------------------------------------------------------------------------------
    def sums_folded(self, W):
        """s'_j = sum_l W_l k'_lj: one DFMA per limb per modulus (exact, lemma 1)."""
        s = [0.0] * self.J
        for l in range(self.N):
            wl = float(W[l])
            for j in range(self.J):
                s[j] = math.fma(wl, self.kf[l][j], s[j])
        return s

    def quotient(self, s):
        """Sd = s'_{J-1} + ... + s'_0 (J - 1 DADD, the shipped Horner chain with the scalings folded), t = fma(Sd, Pinv, C) - C (DFMA + DADD)."""
        Sd = s[self.J - 1]
        for j in range(self.J - 2, -1, -1):
            Sd = Sd + s[j]
        t = math.fma(Sd, self.Pinv, MAGIC) - MAGIC
        return Sd, t

    def limbs(self, t, s):
        """r'_j = fma(-t, P'_j, s'_j): J DFMA, exact."""
        return [math.fma(-t, self.Pf[j], s[j]) for j in range(self.J)]

    def sum_rn(self, r, ro_mut=0):
        J = self.J
        if J == 1:
            return r[0]
        if J == 2:
            return r[1] + r[0]
        if J == 3:
            return sum3_bm(r[2], r[1], r[0], ro_mut)
        top = r[3] + r[2]                                                 # exact (lemma 6)
        assert int(top) == int(r[3]) + int(r[2])
        return sum3_bm(top, r[1], r[0], ro_mut)

    def fast(self, W, ro_mut=0):
        """The FP64 sequence up to z~ (no predicate): returns (z~, t, R exact as an int, the limbs)."""
        s = self.sums_folded(W)
        _, t = self.quotient(s)
        r = self.limbs(t, s)
        R = sum(int(x) for x in r)                                        # the exact remainder (for the tests; the device never forms it)
        return self.sum_rn(r, ro_mut), t, R, r

    @staticmethod
    def scale(z, e):
        """D = z 2^e by an add of e to the exponent field (normal results only; the caller checks the range)."""
        if z == 0.0:
            return 0.0
        return b2f(f2b(z) + (e << 52))

    def fallback_reason(self, z, e, range_on_product=False, thr=None):
        """None (fast path) or the clause that fires: 'centre' or 'range'. `thr` overrides the plan's threshold and `range_on_product` is the wrong
        variant of lemma 7 (tests only)."""
        if abs(z) >= (self.thr if thr is None else thr):
            return "centre"
        if z == 0.0:
            return None
        if range_on_product:
            # wrong: scale by DMUL(s) whatever the range and look at the product
            d = dmul_pow2(z, e)
            return "range" if (d == 0.0 or abs(d) < 2.0 ** -1022 or math.isinf(d)) else None
        be = (f2b(z) >> 52) & 0x7FF
        return None if 1 <= be + e <= 2046 else "range"

    def integer_finish(self, W, e):
        """Today's finish: the shipped limb CRT, centred, then the one exact rounding (what crt_finish + round_scaled / F1 compute)."""
        return o.round_dyadic(self.crt_centred(W), e)

    def finish(self, W, e, ro_mut=0, range_on_product=False):
        """D for residues W and unscaling exponent e: (D, reason) with reason None for the FP64 path, else the fallback clause."""
        z, _, _, _ = self.fast(W, ro_mut)
        why = self.fallback_reason(z, e, range_on_product)
        if why is not None:
            return self.integer_finish(W, e), why
        if range_on_product:
            return dmul_pow2(z, e), None                                 # the DMUL the wrong variant would do
        return self.scale(z, e), None
