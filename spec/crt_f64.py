"""The CRT of DESIGN §3.6 with the sums in binary64 limbs (issue 21, docs/NOTES-21.md "CRT in FP64 limbs").

The reference (oracle.crt) adds M_l y_l to a running L_w-bit word with a conditional subtraction of P at every one of the N steps, y_l = (q_l W_l) mod p_l:
about 46 integer instructions per term on the GPU. Here the sum is formed without any per-term reduction and in the FP64 pipe, which Hopper has at the
rate of its integer pipe (the RTX 4090 and the RTX PRO 6000 do not: this is a per-device-class choice):

    K_l = q_l M_l                (an integer below P; congruent to M_l y_l / W_l ... see below)
    S   = sum_l K_l W_l          W_l in [0, p_l)          S is congruent to the §3.6 value z0 modulo P

  because K_l W_l - M_l ((q_l W_l) mod p_l) = M_l p_l t = P t: the difference is a multiple of P (t = floor(q_l W_l / p_l)).

K_l is cut into J limbs of w bits, K_l = sum_j k_lj B^j with B = 2^w, and the limb sums s_j = sum_l W_l k_lj are exact binary64 integers when
N * 255 * (B - 1) < 2^53 (one fma per limb per term; every partial sum is an integer below 2^53). Then S = sum_j s_j B^j exactly, and

    Sd = fl(s_0 + B s_1 + B^2 s_2 ...)   (J - 1 fused multiply-adds: error at most (J - 1) 2^-53 relative)
    t  = floor(Sd * Pinv - 2^-30)        Pinv = fl(1 / P); S / P < N * 255 < 2^13, so the quotient estimate is off by less than 2^-37:
                                          t never exceeds floor(S / P), and S - t P lies in [0, P (1 + 2^-30))
    r_j = s_j - t P_j                    P = sum_j P_j B^j, t P_j < 2^53: exact; R = sum_j r_j B^j = S - t P exactly
    R in [0, P (1 + 2^-30)): one conditional subtraction of P gives z0 in [0, P), then the usual centring.

The integer steps (limb carries, the conditional subtraction, the centring) are exact integer arithmetic here; the GPU does them in 32-bit words (bounds
asserted below). Everything in binary64 is a multiplication, an addition or an fma of exactly representable integers or of the rounded quantities named above.

Run `python3 spec/test_crt_f64.py`.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_compat  # noqa: E402,F401  (math.fma before Python 3.13)
import oracle as o  # noqa: E402

MAX_W = 255          # the residues W_l are u8 planes, W_l < p_l <= 256
TWO53 = 1 << 53
EPS_T = 2.0 ** -30   # the downward bias of the quotient estimate (the estimate's error is below 2^-37)


class F64Crt:
    """Plan-time constants of the binary64-limb CRT for N moduli (all exact integers or correctly rounded doubles)."""

    def __init__(self, N, w=None):
        self.plan = pl = o.Plan(N, 0)
        self.N, self.P = N, pl.P
        bits = pl.P.bit_length()
        # the largest limb width for which the limb sums are exact, then the fewest limbs that cover K_l < P
        wmax = 1
        while N * MAX_W * ((1 << (wmax + 1)) - 1) < TWO53 and wmax < 52:
            wmax += 1
        self.w = wmax if w is None else w
        assert N * MAX_W * ((1 << self.w) - 1) < TWO53, "limb sums would not be exact"
        self.J = -(-bits // self.w)
        self.B = 1 << self.w
        self.K = [pl.q[l] * pl.M[l] for l in range(N)]
        assert all(0 < k < pl.P for k in self.K)
        self.k = [self._limbs(k) for k in self.K]            # k[l][j]
        self.Pj = self._limbs(pl.P)
        self.Pinv = 1.0 / pl.P                               # correctly rounded (Python int / int is correctly rounded)
        self.Bd = [float(self.B ** j) for j in range(self.J)]   # exact powers of two
        self.tmax = N * MAX_W                                # S / P < sum_l W_l K_l / P < sum_l W_l <= N * 255
        self._check_bounds()

    def _limbs(self, x):
        out = []
        for _ in range(self.J):
            out.append(x & (self.B - 1))
            x >>= self.w
        assert x == 0
        return out

    def _check_bounds(self):
        N, w, J = self.N, self.w, self.J
        # limb sums: s_j <= sum_l 255 k_lj, in particular each partial sum, below 2^53
        for j in range(J):
            assert sum(MAX_W * self.k[l][j] for l in range(N)) < TWO53
        # S / P < tmax; the quotient estimate: S is rounded J - 1 times (relative 2^-53 each, plus the cancellation-free sum of nonnegatives),
        # Pinv once, the product once: the computed quotient differs from S / P by less than tmax (J + 1) 2^-52 < 2^-37
        assert self.tmax * (J + 1) * 2.0 ** -52 < 2.0 ** -37, "quotient estimate too coarse for the bias"
        # r_j = s_j - t P_j and t P_j stay below 2^53, so every r_j is an exact integer
        for j in range(J):
            smax = sum(MAX_W * self.k[l][j] for l in range(N))
            assert smax < TWO53 and (self.tmax + 1) * self.Pj[j] < TWO53 and abs(smax - (self.tmax + 1) * self.Pj[j]) < TWO53
        # R = S - tP < P (1 + 2^-30): a single conditional subtraction reduces it
        assert 2.0 ** -30 * self.P < self.P

    def sums(self, W):
        """s_j = sum_l W_l k_lj in binary64 (each fma exact)."""
        s = [0.0] * self.J
        for l in range(self.N):
            wl = float(W[l])
            for j in range(self.J):
                s[j] = math.fma(wl, float(self.k[l][j]), s[j])
        return s

    def reduce(self, s):
        """From the limb sums to z0 in [0, P): quotient estimate, exact limb remainder, one conditional subtraction."""
        Sd = s[self.J - 1]
        for j in range(self.J - 2, -1, -1):
            Sd = math.fma(Sd, float(self.B), s[j])          # Horner: one rounding per step
        t = math.floor(math.fma(Sd, self.Pinv, -EPS_T))     # fused with the bias, as the GPU does
        assert -1 <= t <= self.tmax
        r = [s[j] - t * float(self.Pj[j]) for j in range(self.J)]   # exact: integers below 2^53
        R = 0
        for j in range(self.J):
            assert r[j] == int(r[j]) and abs(r[j]) < TWO53
            R += int(r[j]) << (self.w * j)
        assert 0 <= R and R << 29 < self.P * ((1 << 29) + 1), (R, self.P)      # R < P (1 + 2^-29)
        return R - self.P if R >= self.P else R

    def crt(self, W):
        """z0 in [0, P) congruent to the CRT of the residues W (what oracle.crt computes before centring)."""
        return self.reduce(self.sums(W))

    def crt_centred(self, W):
        z0 = self.crt(W)
        return z0 if z0 <= self.plan.H else z0 - self.P
