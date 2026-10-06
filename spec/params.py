#!/usr/bin/env python3
"""immerdink parameter certificate: real-width constants, exact integer arithmetic.

This file is the single source of truth for every real-width constant that the
device code and the TLA+ specs use. It is Python 3 stdlib only and uses no
floating point in any certificate. Floats appear only in human-facing guidance
(log2 values, error guidance) and in the end-to-end test, where Python's
correctly rounded int/int division stands in for the final FP64 rounding.

Usage
  python3 spec/params.py --check          run all certificates (exit 1 on failure)
  python3 spec/params.py --table          human table per N
  python3 spec/params.py --json           write spec/params_out.json
  python3 spec/params.py --emit-header P  write the C++ device constant header to P
                                          (include/immerdink/detail/constants.hpp)
  python3 spec/params.py --scaled R A N   scaled constants for a TLA+ model
                                          (residue bits R, accumulator bits A, N moduli)
Run tests only through tools/run_quiet.sh (quiet-machine rule), e.g.
  tools/run_quiet.sh python3 spec/params.py --check

Map to docs/DESIGN.md (v1). Each certificate in CHECKS names its section.
  §2    moduli table, P, H, M_l, q_l, T, K_c, L_w ................ constants(), greedy_moduli()
  §3.2  fast-mode integer shift rule, int16 range ................. shift_u(), shift_bounds()
        plan-time RN rejection kappa >= T ......................... rn_feasible_kmax()
  §3.3  quantization a' = mu * 2^eps, |mu| <= 2^53 ................ quantize()
  §3.4  symmetric residues fit int8, 2^e mod p table .............. smod(), pow2_table()
  §3.5  K-chunk length K_c', int32 exactness, store/add-mod ........ epilogue_range()
        epilogue mod-p in machine words (bias + umulhi64) ......... barrett_plan()
  §3.6  division-free running-sum CRT in L_w = 32*L32 bits ........ crt_reconstruct()
  §3.7  one rounding at the final exponent ......................... emulate_element()
  §3.8  error contract and worst-case guidance ..................... error_guidance()
  §4a   workspace layout ........................................... workspace_size(),
                                                                     region_offsets()
  §6b.4 packed-operand byte layout (format 1, D1) .................. pack_layout(),
                                                                     pack_ws_bytes()
TLA+ models are scaled-down copies of the same rules; scaled_moduli() and
scaled_constants() produce their parameters with the same greedy rule.
"""

import argparse
import json
import math
import os
import random
import sys
from fractions import Fraction

# ---------------------------------------------------------------------------
# Fixed parameters (DESIGN §1, §2, §3.2, §3.5, §4, §4a)

N_MIN, N_MAX = 2, 20                 # §1 options: N in 2..20
REAL_LIMIT = 256                     # §2 greedy start (INT8 residues)
# §2 table, copied from DESIGN text; greedy_moduli(256, 20) must reproduce it.
DESIGN_MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
                 223, 217, 211, 199, 197, 193, 191, 181, 179, 173]
RES_BITS = 8                         # int8 residues
ACC_BITS = 32                        # int32 MMA accumulator (.satfinite: no wrap)
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31
K_C = 131071                         # §2  floor((2^31-1)/128^2)
K_CHUNK_ALIGN = 64                   # v1.13 §4b: K4's threadblock K tile (64 int8 = 64 B)
K_C_PRIME = (K_C // K_CHUNK_ALIGN) * K_CHUNK_ALIGN   # §3.5 131008 = 2047*64 (v1.13; was 131056)
F_BITS = 60                          # §3.2 fixed-point fraction bits of S
MANT = 53                            # FP64 significand bits
GAMMA_MIN, GAMMA_MAX = -1074, 1023   # ilogb range of nonzero finite FP64
TERM_MAX = 2**62 - 1023              # max ceil(x^2 * 2^F) for |x| <= 2 - 2^-52
INT16_MIN, INT16_MAX = -2**15, 2**15 - 1
LIMB = 32                            # CRT word = L32 limbs of 32 bits
REGION_ALIGN = 256                   # §4a align(x) = ceil(x)_256
K_ALIGN = 16                         # §4  k_hat = ceil(k)_16
N_ALIGN = 16                         # §4  n_hat = ceil(n)_16
UNIT_ROUNDOFF = Fraction(1, 2**53)   # u in §3.8
KMAX_CERT = 2**40                    # k range for the int16 shift certificate
SIZE_T_MAX = 2**64 - 1
RN, TRUNC = "RN", "TRUNC"
EMULATE, DIRECT = "EMULATE", "DIRECT"
VEC_GENERATE, VEC_RESIDENT = "GENERATE", "RESIDENT"   # DESIGN v1.2 vec_mode

# ---------------------------------------------------------------------------
# Integer helpers


def ceil_div(a, b):
    return -(-a // b)


def align_up(x, a):
    return ceil_div(x, a) * a


def k_hat_of(k, A=K_ALIGN):
    """DESIGN §4: k_hat = ceil(k)_A, the residue row stride and K4's GEMM K extent; A a power
    of two >= 16 (today every plan uses A = K_ALIGN = 16). Mirrors libimmerdink_host's
    k_hat_of (src/host/k_hat.hpp), the one rounding site there."""
    if A < 16 or A & (A - 1):
        raise ValueError(f"A={A}: a power of two >= 16")
    return align_up(k, A)


def smod(x, p):
    """Symmetric residue in [-floor(p/2), ceil(p/2)-1] (DESIGN §3.4; OzakiCRT SMod)."""
    r = x % p
    return r - p if r >= (p + 1) // 2 else r


def kappa(k):
    """kappa = ceil(sqrt(k)) exactly (DESIGN §3.2); kappa(0) = 0."""
    return 0 if k <= 0 else math.isqrt(k - 1) + 1


def prod(xs):
    r = 1
    for x in xs:
        r *= x
    return r


# ---------------------------------------------------------------------------
# §2 constants


def greedy_moduli(limit, count):
    """Greedy pairwise-coprime moduli descending from `limit` (DESIGN §2)."""
    out, c = [], limit
    while len(out) < count:
        if c < 2:
            raise ValueError(f"only {len(out)} greedy moduli <= {limit}, wanted {count}")
        if all(math.gcd(c, p) == 1 for p in out):
            out.append(c)
        c -= 1
    return out


REAL_MODULI = greedy_moduli(REAL_LIMIT, N_MAX)


def limbs_for(P, limb_bits=LIMB):
    """Minimal L with 2P <= 2^(limb_bits*L): the running CRT sum peaks below 2P."""
    L = 1
    while 2 * P > 2**(limb_bits * L):
        L += 1
    return L


def design_lw(N):
    """L_w as stated in DESIGN §2: 128 for N <= 16, 160 for N <= 20."""
    return 128 if N <= 16 else 160


def constants(N, moduli=None):
    """All exact constants for N moduli (DESIGN §2)."""
    ps = list(moduli) if moduli is not None else REAL_MODULI[:N]
    if moduli is None and not (N_MIN <= N <= N_MAX):
        raise ValueError(f"N={N} outside {N_MIN}..{N_MAX}")
    P = prod(ps)
    H = (P - 1) // 2
    T = math.isqrt(H)
    M = [P // p for p in ps]
    q = [pow(Ml % p, -1, p) for Ml, p in zip(M, ps)]
    L32 = limbs_for(P)
    return {
        "N": N, "moduli": ps, "P": P, "H": H, "T": T, "M": M, "q": q,
        "L32": L32, "Lw": LIMB * L32,
        # exponent range of a' = mu*2^eps: 2^eps <= |a'| <= T (§3.3, §3.4)
        "eps_max": T.bit_length() - 1,
        "crt_peak": crt_peak_exact(ps),
    }


def crt_peak_exact(ps):
    """Exact max of x + M_l*y over every reachable state of the §3.6 loop.

    Before step l, x = (sum_{j<l} M_j y_j) mod P. Every M_j (j<l) is divisible
    by Q_l = prod_{j>=l} p_j, and by the CRT x ranges over all multiples of Q_l
    in [0, P). So max x = P - Q_l, and y_l ranges over [0, p_l).
    """
    P = prod(ps)
    best = 0
    for l, p in enumerate(ps):
        Q = prod(ps[l:])
        best = max(best, (P - Q) + (P // p) * (p - 1))
    return best


def pow2_table(p, emax):
    """2^e mod p for e in 0..emax (DESIGN §3.4)."""
    return [pow(2, e, p) for e in range(emax + 1)]


def scaled_moduli(res_bits, N):
    """Greedy moduli from 2^res_bits, for TLA+ models (same rule as the real table)."""
    return greedy_moduli(2**res_bits, N)


def scaled_constants(res_bits, acc_bits, N, limb_bits=None, k_align=1):
    """Scaled analogues of the §2 constants for a TLA+ model.

    res_bits: signed residue width (8 real); acc_bits: signed accumulator width
    (32 real); limb_bits: CRT limb width (32 real; default res_bits*4).
    """
    ps = scaled_moduli(res_bits, N)
    c = constants(N, ps)
    half = 2**(res_bits - 1)
    kc = (2**(acc_bits - 1) - 1) // (half * half)
    if kc < 1:
        raise ValueError(f"acc_bits={acc_bits} cannot hold one {res_bits}-bit product: K_c = 0")
    kcp = (kc // k_align) * k_align
    if kcp < 1:
        raise ValueError(f"K_c={kc} < k_align={k_align}: K_c' = 0")
    lb = limb_bits if limb_bits is not None else 4 * res_bits
    return {
        "moduli": ps, "P": c["P"], "H": c["H"], "T": c["T"], "M": c["M"], "q": c["q"],
        "K_c": kc, "K_c_prime": kcp,
        "limb_bits": lb, "limbs": limbs_for(c["P"], lb), "crt_peak": c["crt_peak"],
        "word_bits": limbs_for(c["P"], 1),      # least L_w with 2P <= 2^L_w
    }


# ---------------------------------------------------------------------------
# §3.2 shift rule (exact integers)


def _le_scaled(a, ea, b, eb):
    """a*2^ea <= b*2^eb for integers a, b >= 0 and any integer exponents."""
    e = min(ea, eb)
    return (a << (ea - e)) <= (b << (eb - e))


def shift_u(S, T, k, rounding):
    """Largest integer u with the §3.2 rule (S > 0):
         RN:    4*S*2^(2u) <= (2T - kappa)^2 * 2^F
         TRUNC:   S*2^(2u) <=        T^2    * 2^F
    """
    if S <= 0:
        raise ValueError("S must be positive")
    if rounding == RN:
        lhs, R = 4 * S, (2 * T - kappa(k))**2
        if 2 * T - kappa(k) <= 0:
            raise ValueError("RN budget 2T - kappa is not positive")
    elif rounding == TRUNC:
        lhs, R = S, T * T
    else:
        raise ValueError(rounding)
    return largest_u(lhs, R)


def largest_u(lhs, R):
    """Largest integer u with lhs * 2^(2u) <= R * 2^F (lhs, R > 0)."""
    u = (R.bit_length() + F_BITS - lhs.bit_length()) // 2
    while not _le_scaled(lhs, 2 * u, R, F_BITS):
        u -= 1
    while _le_scaled(lhs, 2 * (u + 1), R, F_BITS):
        u += 1
    return u


def rn_feasible_kmax(N):
    """Largest k with kappa = ceil(sqrt(k)) < T (§3.2 plan-time rule): (T-1)^2."""
    T = constants(N)["T"]
    return (T - 1)**2


def shift_bounds(N, kmax, rounding, kmin=1):
    """Exact min/max of sigma = u - gamma over gamma in [GAMMA_MIN, GAMMA_MAX],
    k in [kmin, kmax] and every finite nonzero row (S > 0); a zero row gives 0.

    S = sum ceil(x_h^2 2^F) with |x_h| in [1, 2) for the max entry, so
    S_min = 2^F (one entry 2^gamma, rest 0; any k) and S_max(k) = k*TERM_MAX
    (every |x_h| = 2 - 2^-52), both attainable. u decreases in S and (RN) in
    kappa, so sigma_max = u(S_min, kmin) - GAMMA_MIN and
    sigma_min = u(S_max(k_hi), k_hi) - GAMMA_MAX with k_hi = kmax, clamped to
    rn_feasible_kmax(N) in RN mode (larger k is rejected at plan time).
    """
    c = constants(N)
    T = c["T"]
    k_hi = kmax if rounding == TRUNC else min(kmax, rn_feasible_kmax(N))
    if k_hi < kmin:
        raise ValueError("empty k range")
    u_max = shift_u(2**F_BITS, T, kmin, rounding)
    u_min = shift_u(k_hi * TERM_MAX, T, k_hi, rounding)
    return {
        "N": N, "rounding": rounding, "kmin": kmin, "kmax": kmax, "k_hi": k_hi,
        "u_max": u_max, "u_min": u_min,
        "sigma_max": max(0, u_max - GAMMA_MIN),
        "sigma_min": min(0, u_min - GAMMA_MAX),
    }


# ---------------------------------------------------------------------------
# §3.8 error guidance


def error_guidance(N, k, rounding=RN):
    """Worst-case relative error guidance and the §3.8 bound pieces.

    All coefficients are relative to ||a_i||_2 ||b_j||_2.
      RN    term1 = 2 sqrt(k) s / (T - kappa/2)
            term2 = k 2^(-sigma-tau-2) <= k s^2 / (T - kappa/2)^2
      TRUNC term1 = 4 sqrt(k) s / T
            term2 = k 2^(-sigma-tau)   <= 4 k s^2 / T^2
    with s = sqrt(1 + k 2^-F), the ceiling slack of S (2^-sigma < 2||a|| s /
    (T - kappa/2) exactly; DESIGN §3.8 states it without s). Plus u|D|.
    'guidance' is DESIGN's table quantity 2 sqrt(k)/T (RN).
    """
    c = constants(N)
    T, kap = c["T"], kappa(k)
    s2 = 1 + k / 2**F_BITS
    s = math.sqrt(s2)
    rk = math.sqrt(k)
    out = {"N": N, "k": k, "rounding": rounding, "kappa": kap, "T": T,
           "guidance": 2 * rk / T, "unit_roundoff": 2.0**-53}
    if rounding == RN:
        d = T - kap / 2
        out.update(term1=2 * rk * s / d, term2=k * s2 / d**2,
                   kappa_slack=kap / (2 * T), rn_feasible=kap < T)
    else:
        out.update(term1=4 * rk * s / T, term2=4 * k * s2 / T**2,
                   kappa_slack=0.0, rn_feasible=True)
    return out


# ---------------------------------------------------------------------------
# §6f accuracy-aware planning: the reference of imd_err_estimate (libimmerdink_host)

ERR_NORMWISE, ERR_ELEMENTWISE, ERR_ABSOLUTE = 0, 1, 2
ERR_DEFAULT_FAIL_PROB = 2.0**-40   # review 2026-10-05 (was 2^-20)
ERR_UNDERFLOW = Fraction(1, 2**1075)  # RN to a subnormal: |fl(y) - y| <= u|y| + 2^-1075
DISPATCHES = ("NOOP", "EMULATE", "DIRECT", "SCALE")   # IMD_DISPATCH_* order


def sqrt_enclosure(x, bits=100):
    """(lo, hi) rationals with lo <= sqrt(x) <= hi and hi - lo <= 2^-bits (x >= 0 rational)."""
    x = Fraction(x)
    if x < 0:
        raise ValueError("sqrt of a negative number")
    num, den = x.numerator, x.denominator
    r = math.isqrt(num * den * 4**bits)          # sqrt(x) = sqrt(num den 4^bits) / (den 2^bits)
    scale = den * 2**bits
    return Fraction(r, scale), Fraction(r + 1, scale)


def error_estimate(N, k, rounding, dispatch="EMULATE", outputs=1, metric=ERR_NORMWISE, *,
                   cond=0.0, norm_product=0.0, l1_ratio_a=0.0, l1_ratio_b=0.0,
                   fail_prob=0.0, target=0.0):
    """DESIGN §6f.2: the rigorous bound and the typical estimate of one configuration.

    Rigorous (exact rationals; square roots by enclosures): returns 'bound' and 'unit_bound'
    as (lo, hi) with lo <= the exact real value <= hi (hi - lo tiny), or None for +inf.
      EMULATE: E = 2 delta (nu_a + nu_b) s/d + 4 delta^2 k s^2/d^2   (PROOFS L3a + L1d)
               delta = 1/2, d = T - kappa/2 (RN; kappa >= T: None) | delta = 1, d = T (TRUNC)
               s = sqrt(1 + k 2^-F), nu = min(l1_ratio, sqrt k) (0: sqrt k)
        NORMWISE E (1+u) + u;  ELEMENTWISE E rho (1+u) + u;
        ABSOLUTE (E (1+u) + u) norm_product + 2^-1075 (a gradual underflow of D)
      DIRECT:  gamma_k = k u / (1 - k u) (None if k u >= 1/2): NORMWISE gamma_k,
               ELEMENTWISE gamma_k rho, ABSOLUTE gamma_k norm_product + 2k 2^-1075 (fma underflows)
      NOOP, SCALE, k = 0: 0.
    Typical (a float; NOT a guarantee): RN EMULATE only,
      E_typ = min(E, s/d sqrt(8 ln(2M/p)) + k s^2/d^2), M = max(outputs, 1), p = fail_prob
      (0: 2^-40), composed per metric as E; otherwise the bound (its float hi)."""
    if not N_MIN <= N <= N_MAX or rounding not in (RN, TRUNC) or dispatch not in DISPATCHES:
        raise ValueError("N, rounding or dispatch")
    if k < 0 or outputs < 0 or metric not in (ERR_NORMWISE, ERR_ELEMENTWISE, ERR_ABSOLUTE):
        raise ValueError("k, outputs or metric")
    for v in (l1_ratio_a, l1_ratio_b):
        if not (v == 0 or (math.isfinite(v) and v >= 1)):
            raise ValueError("l1_ratio must be 0 or >= 1")
    if metric == ERR_ELEMENTWISE and not (math.isfinite(cond) and cond >= 1):
        raise ValueError("ELEMENTWISE needs cond >= 1")
    if metric == ERR_ABSOLUTE and not (math.isfinite(norm_product) and norm_product > 0):
        raise ValueError("ABSOLUTE needs norm_product > 0")
    if not (math.isfinite(target) and target >= 0):
        raise ValueError("target must be finite and >= 0 (0: none)")
    if not (fail_prob == 0 or 0 < fail_prob < 1):
        raise ValueError("fail_prob must be 0 or in (0, 1)")
    u = UNIT_ROUNDOFF
    out = {"N": N, "k": k, "rounding": rounding, "dispatch": dispatch, "outputs": outputs,
           "metric": metric}
    zero = (Fraction(0), Fraction(0))
    if dispatch in ("NOOP", "SCALE") or k == 0:
        out.update(unit_bound=zero, bound=zero, unit_typical=0.0, typical=0.0)
        return out
    rho, npd = Fraction(cond), Fraction(norm_product)
    if dispatch == "DIRECT":
        if 2 * k * u >= 1:          # gamma_k >= 1 there: no bound (as the C)
            out.update(unit_bound=None, bound=None, unit_typical=math.inf, typical=math.inf)
            return out
        g = k * u / (1 - k * u)
        b = g * (rho if metric == ERR_ELEMENTWISE else npd if metric == ERR_ABSOLUTE else 1)
        if metric == ERR_ABSOLUTE:
            b += 2 * k * ERR_UNDERFLOW   # PROOFS L3c: k steps, each <= 2^-1075 (1 + gamma_k)
        out.update(unit_bound=(g, g), bound=(b, b), unit_typical=float(g), typical=float(b))
        return out
    T, kap = constants(N)["T"], kappa(k)
    if rounding == RN and kap >= T:
        out.update(unit_bound=None, bound=None, unit_typical=math.inf, typical=math.inf)
        return out
    delta = Fraction(1, 2) if rounding == RN else Fraction(1)
    d = Fraction(2 * T - kap, 2) if rounding == RN else Fraction(T)
    s2 = 1 + Fraction(k, 2**F_BITS)
    s_lo, s_hi = sqrt_enclosure(s2)
    rk_lo, rk_hi = sqrt_enclosure(k)

    def nu(v, rk):
        return rk if v == 0 else min(Fraction(v), rk)

    def unit(s, rk):
        return 2 * delta * (nu(l1_ratio_a, rk) + nu(l1_ratio_b, rk)) * s / d + \
            4 * delta**2 * k * s2 / d**2

    def compose(E):
        if metric == ERR_NORMWISE:
            return E * (1 + u) + u
        if metric == ERR_ELEMENTWISE:
            return E * rho * (1 + u) + u
        return (E * (1 + u) + u) * npd + ERR_UNDERFLOW

    E = (unit(s_lo, rk_lo), unit(s_hi, rk_hi))
    out.update(unit_bound=E, bound=(compose(E[0]), compose(E[1])))
    if rounding == RN:
        M = max(outputs, 1)
        p = fail_prob if fail_prob > 0 else ERR_DEFAULT_FAIL_PROB
        s, df = math.sqrt(float(s2)), float(d)
        e_typ = min(float(E[1]), s / df * math.sqrt(8 * math.log(2 * M / p)) +
                    k * float(s2) / (df * df))
        out.update(unit_typical=e_typ, typical=float(compose(Fraction(e_typ))))
    else:
        out.update(unit_typical=float(E[1]), typical=float(out["bound"][1]))
    return out


# ---------------------------------------------------------------------------
# §3.5 epilogue input range and a proven division-free reduction


def residue_range(p):
    return -(p // 2), (p + 1) // 2 - 1


def epilogue_range(p, chunk=K_C_PRIME):
    """Reachable int32 input of the §3.5 epilogue for modulus p: v = W + acc,
    W in [0, p), acc = sum of `chunk` products of symmetric residues."""
    lo_r, hi_r = residue_range(p)
    pmax = max(lo_r * lo_r, hi_r * hi_r)
    pmin = lo_r * hi_r
    return pmin * chunk, pmax * chunk + (p - 1)


def gm_magic(d, nbits=32):
    """Granlund-Montgomery (1994) Thm 4.2: if 2^s <= m*d <= 2^s + 2^(s-nbits)
    then floor(n/d) = floor(m*n / 2^s) for all 0 <= n < 2^nbits.
    Returns the smallest s >= nbits (and m) satisfying it. Used only to show
    why the emitted realization is NOT this form: for most odd moduli m needs
    33 bits, so m*n needs a 65-bit product (cert_barrett)."""
    for s in range(nbits, nbits + 40):
        m = ceil_div(2**s, d)
        if m * d - 2**s <= 2**(s - nbits):
            return m, s
    raise ValueError("no magic")


MULHI_BITS = 64                      # q = umulhi64(m, n): 64x64 -> high 64 bits


def barrett_plan(p):
    """One valid realization of the §3.5/§4 epilogue reduction (DESIGN leaves the
    method open), in machine words only:
      n = v + bias,  bias = p * ceil(2^31/p)   (a multiple of p, so n = v mod p)
      q = umulhi64(m, n) = floor(m*n / 2^64),  m = ceil(2^64/p) < 2^64
      v mod p = n - p*q   (u32 arithmetic; p*q <= n)
    bias > 2^31, so n is in [0, 2^32) for every REACHABLE v (epilogue_range),
    not for every int32 v (cert_barrett checks the reachable range fits).
    Exactness: let e = m*p - 2^64 in [0, p) and n = Qp + r. Then
    m*n/2^64 = Q + (r + n*e/2^64)/p, and r <= p-1, n*e < 2^64 give q = Q.
    p = 256 uses the mask v & 255 (two's complement)."""
    if p == 256:
        return {"p": p, "kind": "mask"}
    bias = p * ceil_div(2**31, p)
    m = ceil_div(2**MULHI_BITS, p)
    return {"p": p, "kind": "umulhi64", "bias": bias, "m": m}


def barrett_mod(v, plan):
    """Bit-exact model of the device reduction (word widths asserted)."""
    if plan["kind"] == "mask":
        assert INT32_MIN <= v <= INT32_MAX
        return (v & 0xFFFFFFFF) & 255
    n = v + plan["bias"]
    assert 0 <= n < 2**32 and 0 < plan["m"] < 2**MULHI_BITS
    q = (plan["m"] * n) >> MULHI_BITS           # umulhi64(m, n)
    r = n - plan["p"] * q
    assert 0 <= plan["p"] * q <= n < 2**32
    return r


# ---------------------------------------------------------------------------
# §3.6 CRT reconstruction exactly as specified


def crt_reconstruct(W, c, track=None):
    """§3.6: x <- 0; for l: y=(q_l W_l) mod p_l; x += M_l y; if x >= P: x -= P;
    z = x if x <= H else x - P. `track` (list) receives every peak x + M_l y."""
    P, H = c["P"], c["H"]
    x = 0
    for Wl, p, Ml, ql in zip(W, c["moduli"], c["M"], c["q"]):
        assert 0 <= Wl < p
        y = (ql * Wl) % p
        x = x + Ml * y
        if track is not None:
            track.append(x)
        if x >= P:
            x -= P
        assert 0 <= x < P
    return x if x <= H else x - P


# ---------------------------------------------------------------------------
# §4a workspace


def _check_shape(m, n, k, b, N, g, mode, vec=VEC_GENERATE):
    if vec not in (VEC_GENERATE, VEC_RESIDENT):
        raise ValueError(vec)
    for name, v in (("m", m), ("n", n), ("k", k), ("b", b)):
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{name}={v} must be a nonnegative int")
    if not (N_MIN <= N <= N_MAX):
        raise ValueError(f"N={N}")
    if g == 0:
        g = N                         # §6 API: residue_group 0 = N
    if not (1 <= g <= N):
        raise ValueError(f"g={g} outside [1, N]")
    if mode not in (EMULATE, DIRECT):
        raise ValueError(mode)
    return g


def plan_device_bytes(m, n, k, b, vec=VEC_GENERATE):
    """DESIGN v1.2 §4a "plan device bytes" (imd_plan_device_bytes): the resident scatter
    vectors, 8 (2m + 2n + 2k + 3b), for a non-empty output with vec_mode RESIDENT; else 0."""
    if vec not in (VEC_GENERATE, VEC_RESIDENT):
        raise ValueError(vec)
    if vec == VEC_GENERATE or b * m * n == 0:
        return 0
    return 8 * (2 * m + 2 * n + 2 * k + 3 * b)


def _check_packed(packed):
    packed = frozenset(packed)
    if not packed <= {"A", "B"}:
        raise ValueError(f"packed={sorted(packed)}: post-swap roles A, B only")
    return packed


def workspace_size(m, n, k, b, N, g, mode=EMULATE, vec=VEC_GENERATE, packed=()):
    """DESIGN §4a, transcribed literally as a closed form:
    sum of align(bytes) over sA sB fA fB vec RA RB W; 0 if b*m*n = 0;
    only align(vec) in DIRECT mode, when k = 0, or when K is empty (k = 1 after lowering:
    extent-1 labels are dropped, so k = 1 <=> K-empty, dispatched to DIRECT; DESIGN v1.1 §4/§4a).
    v1.2: with vec = RESIDENT the vec term is 0 everywhere (the vectors are plan memory).
    v1.7 (§6b.6): `packed` = the POST-SWAP roles ("A", "B") read from packed operands; their
    sX, fX and RX terms are 0 (the caller's packed buffer holds them)."""
    g = _check_shape(m, n, k, b, N, g, mode, vec)
    packed = _check_packed(packed)
    if b * m * n == 0:
        return 0
    A = lambda x: align_up(x, REGION_ALIGN)  # noqa: E731
    if 8 * (2 * m + 2 * n + 2 * k + 3 * b) > SIZE_T_MAX:   # either mode needs the vectors
        raise OverflowError("scatter vectors do not fit size_t")
    vec = 0 if vec == VEC_RESIDENT else A(8 * (2 * m + 2 * n + 2 * k + 3 * b))
    if mode == DIRECT or k <= 1:
        total = vec
    else:
        kh, nh = k_hat_of(k), align_up(n, N_ALIGN)
        pa, pb = "A" not in packed, "B" not in packed      # 1: packed in-line (in the workspace)
        total = (pa * A(2 * b * m) + pb * A(2 * b * n) + pa * A(b * m) + pb * A(b * n) + vec
                 + pa * A(g * b * m * kh) + pb * A(g * b * nh * kh) + A(N * b * m * nh))
    if total > SIZE_T_MAX:
        raise OverflowError("workspace does not fit size_t")
    return total


# Region table (§4a order). Each entry: (name, element bytes, extent function).
_REGIONS = [
    ("sA", lambda m, n, k, b, N, g, kh, nh: 2 * b * m),
    ("sB", lambda m, n, k, b, N, g, kh, nh: 2 * b * n),
    ("fA", lambda m, n, k, b, N, g, kh, nh: 1 * b * m),
    ("fB", lambda m, n, k, b, N, g, kh, nh: 1 * b * n),
    ("vec", lambda m, n, k, b, N, g, kh, nh: 8 * (2 * m + 2 * n + 2 * k + 3 * b)),
    ("RA", lambda m, n, k, b, N, g, kh, nh: g * b * m * kh),
    ("RB", lambda m, n, k, b, N, g, kh, nh: g * b * nh * kh),
    ("W", lambda m, n, k, b, N, g, kh, nh: N * b * m * nh),
]


def region_offsets(m, n, k, b, N, g, mode=EMULATE, vec=VEC_GENERATE, packed=()):
    """[(name, start, end)] laid out in §4a order at 256-B-aligned offsets (no vec region
    with vec = RESIDENT; v1.7: no sX, fX, RX for a packed post-swap role X in `packed`)."""
    g = _check_shape(m, n, k, b, N, g, mode, vec)
    packed = _check_packed(packed)
    if b * m * n == 0:
        return []
    kh, nh = k_hat_of(k), align_up(n, N_ALIGN)
    names = ["vec"] if (mode == DIRECT or k <= 1) else [r[0] for r in _REGIONS]
    if vec == VEC_RESIDENT:
        names = [x for x in names if x != "vec"]
    for X in packed:
        names = [x for x in names if x not in ("s" + X, "f" + X, "R" + X)]
    out, off = [], 0
    for name, f in _REGIONS:
        if name not in names:
            continue
        nbytes = f(m, n, k, b, N, g, kh, nh)
        out.append((name, off, off + nbytes))
        off = align_up(off + nbytes, REGION_ALIGN)
    return out


# ---------------------------------------------------------------------------
# §6b.4 packed operand, byte layout format 1 (DESIGN v1.7, D1)

PACK_FORMAT = 1
_PACK_REGIONS = ("sig", "flag", "res")


def pack_layout(b, r, k, N):
    """DESIGN §6b.4, format 1: [(name, offset, bytes)] of the regions sig (int16 sigma of row
    (lambda, rho) at lambda r + rho), flag (u8, same index) and res (int8 residues of plane l,
    row (lambda, rho), k-bar at l b r k_hat + (lambda r + rho) k_hat + k-bar, then a zeroed tail
    of (ceil(r)_16 - r) k_hat bytes), in that order at 256-aligned offsets, and pack_bytes =
    the sum of the aligned region sizes. ([], 0) if b r k = 0. OverflowError when a size or
    the total exceeds uint64 (the library: INVALID_VALUE)."""
    for name, v in (("b", b), ("r", r), ("k", k)):
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{name}={v} must be a nonnegative int")
    if not (N_MIN <= N <= N_MAX):
        raise ValueError(f"N={N}")
    if b * r * k == 0:
        return [], 0
    kh = k_hat_of(k)
    sizes = {"sig": 2 * b * r, "flag": b * r,
             "res": N * b * r * kh + (align_up(r, N_ALIGN) - r) * kh}
    out, off = [], 0
    for name in _PACK_REGIONS:
        nbytes = sizes[name]
        if nbytes > SIZE_T_MAX:
            raise OverflowError(f"pack region {name} exceeds uint64")
        out.append((name, off, nbytes))
        off = align_up(off + nbytes, REGION_ALIGN)
        if off > SIZE_T_MAX:
            raise OverflowError("pack_bytes exceeds uint64")
    return out, off


def pack_ws_bytes(b, r, k):
    """The workspace imd_pack needs (an implementation choice of D1, DESIGN §6b.5
    pack_ws_bytes): the operand's scatter vectors, rows r, k-bar k and batches b as int64,
    8 (r + k + b) aligned to 256; 0 for an empty pack (b r k = 0)."""
    if b * r * k == 0:
        return 0
    v = 8 * (r + k + b)
    if v > SIZE_T_MAX or align_up(v, REGION_ALIGN) > SIZE_T_MAX:
        raise OverflowError("pack workspace exceeds uint64")
    return align_up(v, REGION_ALIGN)


# ---------------------------------------------------------------------------
# End-to-end real-width emulation of one output element (§3.2-§3.8)


def ilogb(x):
    return math.frexp(x)[1] - 1


def scale_row(vals, N, rounding):
    """§3.2 for one row (or column): returns (sigma, S, gamma)."""
    T = constants(N)["T"]
    k = len(vals)
    nz = [v for v in vals if v != 0.0]
    if not nz:
        return 0, 0, None
    gamma = ilogb(max(abs(v) for v in nz))
    S = 0
    for v in nz:
        t = Fraction(v)**2 * Fraction(2)**(F_BITS - 2 * gamma)  # x_h^2 2^F, exact
        S += ceil_div(t.numerator, t.denominator)
    u = shift_u(S, T, k, rounding)
    return u - gamma, S, gamma


def quantize(v, sigma, rounding):
    """§3.3: a' = RN_even(v 2^sigma) or trunc(v 2^sigma), returned as (a', mu, eps)."""
    fr = Fraction(v)
    if fr == 0:
        return 0, 0, 0
    n, d = fr.numerator, fr.denominator      # v = n 2^-t, d = 2^t
    t = d.bit_length() - 1
    while n % 2 == 0:                        # n odd, so |n| < 2^53 for any double
        n //= 2
        t -= 1
    e = sigma - t
    if e >= 0:
        mu, eps = n, e
    else:
        q = Fraction(n, 2**(-e))
        mu = round(q) if rounding == RN else math.trunc(q)   # round(): ties-to-even
        eps = 0
    return mu * 2**eps, mu, eps


def emulate_element(a, b, N, rounding, chunk):
    """Exact model of §3.2-§3.7 for one (i, j), alpha = 1, beta = 0.
    Uses an artificially small K-chunk to exercise the §3.5 recurrence."""
    c = constants(N)
    k = len(a)
    if rounding == RN and kappa(k) >= c["T"]:
        raise ValueError("RN rejected at plan time: kappa >= T")
    sig, Sa, ga = scale_row(a, N, rounding)
    tau, Sb, gb = scale_row(b, N, rounding)
    qa = [quantize(v, sig, rounding) for v in a]
    qb = [quantize(v, tau, rounding) for v in b]
    emax = c["eps_max"]
    W = []
    for p in c["moduli"]:
        tab = pow2_table(p, emax)
        ra, rb = [], []
        for (x, mu, eps) in qa + qb:
            assert abs(mu) <= 2**MANT and 0 <= eps <= emax, (mu, eps)
            r = smod((mu % p) * tab[eps], p)
            assert r == smod(x, p) and -128 <= r <= 127
            (ra if len(ra) < k else rb).append(r)
        Wl = None
        for c0 in range(0, k, chunk):
            acc = sum(x * y for x, y in zip(ra[c0:c0 + chunk], rb[c0:c0 + chunk]))
            assert abs(acc) <= 128 * 128 * chunk
            Wl = acc % p if Wl is None else (Wl + acc) % p
        W.append(Wl)
    z = crt_reconstruct(W, c)
    return {"sigma": sig, "tau": tau, "Sa": Sa, "Sb": Sb, "ga": ga, "gb": gb,
            "a_q": [x for x, _, _ in qa], "b_q": [x for x, _, _ in qb], "z": z,
            "D": float(Fraction(z) / Fraction(2)**(sig + tau)), "c": c}


# ---------------------------------------------------------------------------
# Certificates

CHECKS = []


def check(section):
    def deco(fn):
        CHECKS.append((section, fn))
        return fn
    return deco


class CertFail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise CertFail(msg)


@check("§2 moduli")
def cert_moduli():
    """greedy_moduli(256,20) equals the DESIGN table; pairwise coprime; all <= 256."""
    need(REAL_MODULI == DESIGN_MODULI, f"greedy {REAL_MODULI} != DESIGN {DESIGN_MODULI}")
    for i, p in enumerate(REAL_MODULI):
        need(2 <= p <= 256, f"p={p}")
        for q in REAL_MODULI[:i]:
            need(math.gcd(p, q) == 1, f"gcd({p},{q}) != 1")
    need(241 in REAL_MODULI, "241 must be present ([OZ2] erratum)")
    # greedy property: every skipped candidate shares a factor with an earlier pick
    for c in range(REAL_MODULI[-1], 257):
        if c not in REAL_MODULI:
            need(any(math.gcd(c, p) > 1 for p in REAL_MODULI if p > c), f"{c} skipped wrongly")
    return f"{len(REAL_MODULI)} moduli, pairwise coprime, greedy"


@check("§2 CRT identity")
def cert_crt_identity():
    """M_l*q_l = delta_lj (mod p_j) for all l, j; q_l in [1, p_l); M_l p_l = P."""
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        for l, (Ml, ql, pl) in enumerate(zip(c["M"], c["q"], c["moduli"])):
            need(1 <= ql < pl, f"N={N} q_{l}={ql}")
            need(Ml * pl == c["P"], "M_l p_l = P")
            need(Ml * (pl - 1) < c["P"], "M_l*y < P")   # DESIGN §3.6 annotation (follows from M_l p_l = P)
            for j, pj in enumerate(c["moduli"]):
                need((Ml * ql) % pj == (1 if j == l else 0), f"N={N} l={l} j={j}")
    return "N=2..20, all (l, j)"


@check("§2 P, H, T")
def cert_pht():
    """P even; H = floor((P-1)/2) = P/2 - 1; T = isqrt(H): T^2 <= H < (T+1)^2."""
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        P, H, T = c["P"], c["H"], c["T"]
        need(P % 2 == 0 and H == P // 2 - 1, f"N={N} parity")
        need(T * T <= H < (T + 1)**2, f"N={N} T")
    return "N=2..20"


@check("§2 L_w / limbs")
def cert_limbs():
    """2P <= 2^(32*L32) minimal; DESIGN's L_w (128 for N<=16, 160 for N<=20) holds and is tight at N=17."""
    rows = []
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        P, L = c["P"], c["L32"]
        need(2 * P <= 2**(32 * L), f"N={N} 2P fits")
        need(L == 1 or 2 * P > 2**(32 * (L - 1)), f"N={N} L32 minimal")
        need(2 * P <= 2**design_lw(N), f"N={N} DESIGN L_w={design_lw(N)} too small")
        rows.append(L)
    need(2 * constants(17)["P"] > 2**128, "128 bits must fail at N=17 (tightness)")
    need(2 * constants(16)["P"] <= 2**127, "N=16 also fits signed int128")
    return "L32 by N=2..20: " + "".join(str(x) for x in rows)


@check("§3.6 running-sum peak")
def cert_crt_peak():
    """Exact reachable peak of x + M_l*y (crt_peak_exact) is < 2P <= 2^Lw and needs all L32 limbs."""
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        P, pk = c["P"], c["crt_peak"]
        need(pk < 2 * P, f"N={N} peak >= 2P")
        need(pk <= 2**c["Lw"] - 1, f"N={N} peak exceeds Lw")
        need(pk >= 2**(c["Lw"] - LIMB), f"N={N} one limb fewer would do for the exact peak")
    # brute-force the reachable-peak formula on the scaled tables
    for res in (3, 4):
        for N in (2, 3, 4):
            ps = scaled_moduli(res, N)
            c = constants(N, ps)
            best = 0
            for z in range(c["P"]):
                tr = []
                crt_reconstruct([z % p for p in ps], c, tr)
                best = max(best, max(tr))
            need(best == c["crt_peak"], f"scaled peak {ps}: {best} != {c['crt_peak']}")
    return ("exact reachable peak < 2P <= 2^Lw for N=2..20, 2P - peak = "
            f"{min(2 * constants(N)['P'] - constants(N)['crt_peak'] for N in range(N_MIN, N_MAX + 1))}"
            " at the closest N; formula brute-forced on scaled tables")


@check("§3.6 CRT reconstruction")
def cert_crt_roundtrip():
    """crt_reconstruct(residues(z)) = z: exhaustive for N=2, boundaries+random for N<=20."""
    c = constants(2)
    for z in range(-c["H"] - 1, c["H"] + 1):
        need(crt_reconstruct([z % p for p in c["moduli"]], c) == z, f"N=2 z={z}")
    rng = random.Random(20261002)
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        H = c["H"]
        zs = [0, 1, -1, H, -H, H - 1, -H + 1, -H - 1] + [rng.randint(-H, H) for _ in range(300)]
        for z in zs:
            need(crt_reconstruct([z % p for p in c["moduli"]], c) == z, f"N={N} z={z}")
    return "N=2 exhaustive over [-P/2, P/2-1]; N=3..20 boundaries + 300 random"


@check("§2/§3.5 K_c, K_c'")
def cert_kc():
    """128^2 K_c <= 2^31-1 < 128^2 (K_c+1); K_c' = floor(K_c/64)*64 (v1.13 §4b: a multiple of
    K4's 64-wide K tile, so every full chunk starts and ends on a 64-byte phase; a multiple of
    16 too, the CUTLASS pointer rule) and maximal under that rule; epilogue W+acc fits int32."""
    need(128 * 128 * K_C <= INT32_MAX < 128 * 128 * (K_C + 1), "K_c")
    need(K_C == INT32_MAX // (128 * 128), "K_c formula")
    need(K_C_PRIME % 16 == 0 and K_C_PRIME % K_CHUNK_ALIGN == 0 and K_C_PRIME <= K_C and
         K_C_PRIME == (K_C // K_CHUNK_ALIGN) * K_CHUNK_ALIGN, "K_c'")
    need(K_C - K_C_PRIME < K_CHUNK_ALIGN, "K_c' maximal")
    need(K_C_PRIME == 131008 == 2047 * 64, "K_c' value (DESIGN §2 table)")
    # the pinned extremes of the 256 GEMM's epilogue input (DESIGN §3.5, PROOFS L-epilogue)
    need(epilogue_range(256) == (-2129666048, 2146435327), "v range p=256 literal")
    for p in REAL_MODULI:
        lo, hi = epilogue_range(p)
        need(INT32_MIN <= lo and hi <= INT32_MAX, f"p={p} epilogue range [{lo},{hi}] not int32")
        # at full K_c the add-mod (W + acc) would still fit
        lo2, hi2 = epilogue_range(p, K_C)
        need(INT32_MIN <= lo2 and hi2 <= INT32_MAX, f"p={p} at K_c")
    # the 256 GEMM is the binding one: one more step of K_c overflows int32
    need(128 * 128 * (K_C + 1) > INT32_MAX, "tight")
    # pin epilogue_range: closed forms, and extreme residue products by brute force
    need(epilogue_range(256) == (-128 * 127 * K_C_PRIME, 128 * 128 * K_C_PRIME + 255), "v range p=256")
    for p in REAL_MODULI:
        lo_r, hi_r = residue_range(p)
        prods = [x * y for x in range(lo_r, hi_r + 1) for y in (lo_r, hi_r, 0, 1, -1)]
        need(epilogue_range(p) == (min(prods) * K_C_PRIME, max(prods) * K_C_PRIME + p - 1),
             f"epilogue_range p={p} vs brute-force residue products")
        if p % 2:
            h = (p - 1) // 2
            need(epilogue_range(p) == (-h * h * K_C_PRIME, h * h * K_C_PRIME + p - 1), f"closed form p={p}")
    return f"K_c={K_C}, K_c'={K_C_PRIME}, max W+acc = {epilogue_range(256)[1]}"


# v1.13 §4b boundary k list: around one, two and four K_c' (old and new), the measured long-k
# shapes (196608, 196624) and the one-chunk windows k_hat in (j K_c', j 131056].
K_BOUNDARY = [1, 16, 17, 64, 65, 131007, 131008, 131009, 131040, 131041, 131056, 131057,
              131071, 131072, 131073, 196608, 196624, 262016, 262017, 262112, 262113, 262144,
              262160, 262200, 393024, 393025, 393168, 524032, 524033]


def k_chunk_list(k_hat, chunk=K_C_PRIME):
    """§3.5 the K-chunks of a row: [(k0, kc)], k0 = c chunk, the last one ending at k_hat."""
    return [(k0, min(chunk, k_hat - k0)) for k0 in range(0, k_hat, chunk)]


def k4_row_phase(row_base, k_hat, k0, kc, tile=K_CHUNK_ALIGN):
    """v1.13 §4b: the byte phase (mod 64) of row r's main-loop K4 loads in the chunk
    [k0, k0 + kc). CUTLASS 2.x loads the chunk's residue tile (kc mod 64 elements) first, so
    every later 64-element load starts at row_base + k0 + (kc mod 64) + 64 t, i.e. at phase
    (row_base + chunk end) mod 64 with row_base = r k_hat (mod 64) for 256-aligned regions."""
    return (row_base + k0 + kc % tile) % tile


@check("v1.13 §4b K4 row phase")
def cert_row_phase():
    """With K_c' = 131008 every full K-chunk starts and ends on a multiple of 64, so the
    main-loop load phase of row r in chunk c, (r k_hat + e_c) mod 64 with e_c the chunk end,
    is r k_hat mod 64 in a full chunk and (r + 1) k_hat mod 64 in the last one (e_c = k_hat):
    the set of phases over the rows, hence the phase class, depends on k_hat alone. The old
    K_c' = 131056 shifted chunk c by 48 (c + 1) (the measured slow chunk 0 of k_hat = 196608).
    k_chunks gains one exactly in the windows k_hat in (j K_c', j 131056]; this holds while
    the windows are disjoint, j 48 < K_c' (j <= 2729, k_hat < ~3.58e8), which covers the
    checked list (j < 5); beyond that the counts can differ by more than one."""
    n = 0
    for k in K_BOUNDARY + list(range(1, 2000, 7)):
        kh = k_hat_of(k)
        chunks = k_chunk_list(kh)
        need(len(chunks) == ceil_div(kh, K_C_PRIME) and sum(c[1] for c in chunks) == kh, f"k={k}")
        for ci, (k0, kc) in enumerate(chunks):
            need(k0 % 16 == 0 and kc % 16 == 0, f"k={k} chunk {ci}: CUTLASS 16-B rule")
            if ci < len(chunks) - 1:
                need(kc == K_C_PRIME and k0 % 64 == 0 and (k0 + kc) % 64 == 0, f"k={k} full chunk {ci}")
            end = kh if ci == len(chunks) - 1 else 0     # e_c mod 64: 0 for a full chunk
            for r in range(8):
                need(k4_row_phase(r * kh, kh, k0, kc) == (r * kh + end) % 64, f"k={k} chunk {ci} row {r}")
                n += 1
        old = k_chunk_list(kh, 131056)
        if len(old) > 1:   # the old chunk length shifts chunk 0's phase by 48 (not by k_hat)
            need(k4_row_phase(0, kh, *old[0]) == 48, f"k={k}: old chunk 0 phase")
        win = any(j * K_C_PRIME < kh <= j * 131056 for j in range(1, 5))
        need((len(chunks) == len(old) + 1) == win and len(chunks) - len(old) in (0, 1), f"k={k} window")
    # the measured long-k case (k4align measurements): 196608 = 3 * 2^16, two chunks
    need(k4_row_phase(0, 196608, *k_chunk_list(196608, 131056)[0]) == 48 and
         k4_row_phase(0, 196608, *k_chunk_list(196608, 131056)[1]) == 0, "196608 old phases")
    need([k4_row_phase(0, 196608, *c) for c in k_chunk_list(196608)] == [0, 0], "196608 new phases")
    return f"{len(K_BOUNDARY)} boundary k + 286 small k, {n} (row, chunk) phases = (r k_hat + e_c) mod 64, e_c = 0 or k_hat"


@check("§3.5 mod reduction realization")
def cert_barrett():
    """Emitted epilogue reduction (mask for 256; bias + umulhi64(ceil(2^64/p), n)
    otherwise) is exact on the whole reachable range, in machine words:
    m < 2^64, n = v + bias in [0, 2^32) for v in epilogue_range(p), and the
    sufficient exactness condition n_max * (m p - 2^64) < 2^64 (proof in
    barrett_plan). Samples at every extreme and around multiples of p confirm
    the bit-exact model. Also shows the 32-bit GM multiply-shift form would
    need a 65-bit product (why it is not emitted)."""
    rng = random.Random(7)
    wide = []
    for p in REAL_MODULI:
        plan = barrett_plan(p)
        lo, hi = epilogue_range(p)
        if plan["kind"] == "mask":
            need(p == 256, "mask only for 256")
        else:
            m, bias = plan["m"], plan["bias"]
            need(bias % p == 0 and bias >= 2**31, f"bias p={p}")
            need(1 <= m < 2**MULHI_BITS, f"m={m} not a 64-bit word, p={p}")
            nlo, nhi = lo + bias, hi + bias
            need(0 <= nlo and nhi < 2**32, f"reachable n not in [0,2^32) for p={p}")
            need(INT32_MAX + bias >= 2**32, "bias docstring: not every int32 v fits")
            e = m * p - 2**MULHI_BITS
            need(0 <= e < p, f"e p={p}")
            need(nhi * e < 2**MULHI_BITS, f"exactness condition fails p={p}")
            gm_m, gm_s = gm_magic(p)
            if (gm_m * nhi).bit_length() > 64:
                wide.append(p)
        samples = [lo, lo + 1, hi, hi - 1, 0, -1, 1, p - 1, p, -p] + \
                  [rng.randint(lo, hi) for _ in range(2000)]
        for base in (lo, hi, 0):                 # around multiples of p near extremes
            q0 = base // p
            for qq in (q0 - 1, q0, q0 + 1):
                samples += [qq * p - 1, qq * p, qq * p + p - 1]
        for v in samples:
            if lo <= v <= hi:
                need(barrett_mod(v, plan) == v % p, f"p={p} v={v}")
    need(len(wide) >= 1, "expected the 32-bit GM form to need > 64-bit products")
    return (f"all 20 moduli, umulhi64 exact on reachable n < 2^32; "
            f"GM 32-bit form would need a 65-bit product for {len(wide)} moduli")


@check("§3.6 y=(q W) mod p reuses ceil(2^64/p)")
def cert_crt_y_reduction():
    """The §3.6 step y = (q_l W_l) mod p_l reuses the §3.5 constant m = ceil(2^64/p) with
    bias 0: n = q_l W_l < 2^16 (q_l < p_l, W_l < p_l <= 256), and umulhi64(m, n) = n div p
    for EVERY n in [0, 2^16) and every p != 256 (exhaustive; p = 256 uses n & 255).
    Also implied by barrett_plan's condition n * (m p - 2^64) < 2^64, re-checked here."""
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        for p, q in zip(c["moduli"], c["q"]):
            need(0 <= q < p and (p - 1) * q < 2**16, f"q W >= 2^16 for N={N} p={p}")
    for p in REAL_MODULI:
        plan = barrett_plan(p)
        if plan["kind"] == "mask":
            need(all((n & 255) == n % p for n in range(2**16)), "mask 256")
            continue
        m = plan["m"]
        need((2**16 - 1) * (m * p - 2**MULHI_BITS) < 2**MULHI_BITS, f"condition p={p}")
        for n in range(2**16):
            q = (m * n) >> MULHI_BITS
            if q != n // p or n - p * q != n % p:
                raise CertFail(f"p={p} n={n}")
    return "exhaustive over n in [0, 2^16) for all 20 moduli (19 umulhi64 + mask for 256)"


@check("§3.4 residues fit int8")
def cert_residues():
    """smod(x, p) in [-floor(p/2), ceil(p/2)-1] subset of int8, congruent, for every class."""
    for p in REAL_MODULI:
        lo, hi = residue_range(p)
        need(-128 <= lo and hi <= 127, f"p={p}")
        seen = set()
        for x in range(-2 * p, 2 * p):
            r = smod(x, p)
            need(lo <= r <= hi and (r - x) % p == 0, f"p={p} x={x}")
            seen.add(r)
        need(len(seen) == p, f"p={p} not a full residue system")
    return "p=256 -> [-128,127], odd p -> [-(p-1)/2,(p-1)/2]"


@check("§3.3/§3.4 2^e mod p table")
def cert_pow2():
    """a' = mu 2^eps with |mu| <= 2^53, 0 <= eps <= floor(log2 T); table path equals direct residue."""
    rng = random.Random(11)
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        emax = c["eps_max"]
        need(2**emax <= c["T"] < 2**(emax + 1), f"N={N} eps_max")
        for p in c["moduli"]:
            tab = pow2_table(p, emax)
            for e in range(emax):
                need(tab[e + 1] == (2 * tab[e]) % p, f"recurrence p={p}")
            need(p != 256 or all(t == 0 for t in tab[8:]), "2^e = 0 mod 256 for e >= 8")
            need(all(0 <= t < p for t in tab), "range")
            for _ in range(40):
                eps = rng.randint(0, emax)
                mu = rng.randint(-2**MANT, 2**MANT)
                need(smod((mu % p) * tab[eps], p) == smod(mu * 2**eps, p), f"p={p}")
    return f"eps_max from {constants(2)['eps_max']} (N=2) to {constants(20)['eps_max']} (N=20)"


@check("§3.2 S width")
def cert_s_width():
    """Each term ceil(x^2 2^60) <= 2^62-1023 < 2^63; S fits uint128 for every k < 2^66."""
    x = 2**MANT - 1                      # x = (2^53-1) 2^-52 = 2 - 2^-52
    t = Fraction(x * x, 2**(2 * (MANT - 1))) * 2**F_BITS
    need(ceil_div(t.numerator, t.denominator) == TERM_MAX, "TERM_MAX")
    need(TERM_MAX < 2**63, "term fits int64")
    kS = (2**128 - 1) // TERM_MAX
    need(kS >= 2**66 - 1, f"uint128 S only for k <= {kS}")
    need(math.frexp(5e-324)[1] - 1 == GAMMA_MIN and math.frexp(1.7976931348623157e308)[1] - 1 == GAMMA_MAX,
         "ilogb range")
    return f"S <= k*(2^62-1023) < 2^128 for k <= {kS} (2^66 + {kS - 2**66})"


@check("§3.2 kappa and RN feasibility")
def cert_kappa():
    """kappa(k) = ceil(sqrt k); rn_feasible_kmax(N) = (T-1)^2 is the exact boundary."""
    for k in list(range(1, 5000)) + [2**40, 2**40 + 1, 2**62, (2**31 - 1)**2, (2**31 - 1)**2 + 1]:
        kp = kappa(k)
        need((kp - 1)**2 < k <= kp * kp, f"kappa({k})")
    for N in range(N_MIN, N_MAX + 1):
        T = constants(N)["T"]
        kx = rn_feasible_kmax(N)
        need(kappa(kx) < T <= kappa(kx + 1), f"N={N}")
    return f"N=2: k <= {rn_feasible_kmax(2)}; N=3: k <= {rn_feasible_kmax(3)}; N=4: k <= {rn_feasible_kmax(4)}"


@check("§3.2 int16 shift range")
def cert_shift_range():
    """sigma fits int16 for k <= 2^40 (and for every k < 2^66) in both roundings; sigma+tau fits int16."""
    lo, hi = 0, 0
    for N in range(N_MIN, N_MAX + 1):
        for rnd in (RN, TRUNC):
            for kmax in (KMAX_CERT, (2**128 - 1) // TERM_MAX):
                sb = shift_bounds(N, kmax, rnd)
                need(INT16_MIN <= sb["sigma_min"] and sb["sigma_max"] <= INT16_MAX, f"N={N} {rnd}")
                if kmax == KMAX_CERT:
                    lo, hi = min(lo, sb["sigma_min"]), max(hi, sb["sigma_max"])
    need(INT16_MIN <= 2 * lo and 2 * hi <= INT16_MAX, "sigma+tau int16")
    need(-1100 <= lo and hi <= 1160, f"DESIGN 'about [-1100, 1160]' vs [{lo}, {hi}]")
    return f"sigma in [{lo}, {hi}] over N=2..20, k<=2^40, RN+TRUNC"


@check("§3.2 shift-bound attainment")
def cert_shift_attain():
    """shift_bounds equals brute force over extremal rows for small kmax (both roundings)."""
    tiny, big = 5e-324, 1.7976931348623157e308
    vals = [0.0, tiny, 2 * tiny, 3 * tiny, 1.0, 1.5, big, -big, big / 2, 2.0**-1022]
    rng = random.Random(3)
    for N in (2, 3, 8, 20):
        for rnd in (RN, TRUNC):
            kmax = 4
            sb = shift_bounds(N, kmax, rnd)
            lo, hi = 0, 0
            for k in range(1, kmax + 1):
                rows = [[rng.choice(vals) for _ in range(k)] for _ in range(60)]
                rows += [[big] * k, [-big] * k, [tiny] + [0.0] * (k - 1)]
                for row in rows:
                    s, _, _ = scale_row(row, N, rnd)
                    lo, hi = min(lo, s), max(hi, s)
            need((lo, hi) == (sb["sigma_min"], sb["sigma_max"]),
                 f"N={N} {rnd}: brute [{lo},{hi}] vs {sb['sigma_min'], sb['sigma_max']}")
    # monotonicity of u in S and in k (the endpoints argument)
    for N in (2, 9, 20):
        T = constants(N)["T"]
        for rnd in (RN, TRUNC):
            kx = KMAX_CERT if rnd == TRUNC else min(KMAX_CERT, rn_feasible_kmax(N))
            ks = sorted({1, 2, 3, kx} | {2**i for i in range(0, 41) if 2**i <= kx})
            prev = None
            for k in ks:
                u = shift_u(k * TERM_MAX, T, k, rnd)
                need(prev is None or u <= prev, "u not monotone in k")
                need(shift_u(2**F_BITS, T, k, rnd) >= u, "u not monotone in S")
                prev = u
    # Attainment at real k ranges, where RN's u depends on kappa:
    #  sigma_max is the sigma of the single-entry row [2^-1074] (any k, so k=1);
    #  sigma_min is the sigma of k_hi copies of DBL_MAX: S = k_hi * term(DBL_MAX)
    #  (all terms equal), term computed from the double, not from TERM_MAX.
    tf = Fraction(big)**2 * Fraction(2)**(F_BITS - 2 * ilogb(big))
    term_big = ceil_div(tf.numerator, tf.denominator)
    need(term_big == TERM_MAX, "TERM_MAX is the DBL_MAX term")
    for N in range(N_MIN, N_MAX + 1):
        T = constants(N)["T"]
        for rnd in (RN, TRUNC):
            sb = shift_bounds(N, KMAX_CERT, rnd)
            need(sb["sigma_max"] == scale_row([tiny], N, rnd)[0], f"N={N} {rnd} sigma_max not attained")
            khi = sb["k_hi"]
            need(khi == (KMAX_CERT if rnd == TRUNC else min(KMAX_CERT, rn_feasible_kmax(N))), "k_hi")
            need(sb["sigma_min"] == shift_u(khi * term_big, T, khi, rnd) - ilogb(big),
                 f"N={N} {rnd} sigma_min not attained")
    # brute force over real rows at N=2 where kappa reaches T-1 (RN rejects beyond)
    kx2 = rn_feasible_kmax(2)
    for rnd, kmaxes in ((RN, (1, 2, 3, 100, kx2)), (TRUNC, (1, 2, 100, kx2, kx2 + 7))):
        for kmax in kmaxes:
            sb = shift_bounds(2, kmax, rnd)
            lo, hi = 0, 0
            for k in sorted(x for x in {1, 2, kmax // 2, kmax - 1, kmax} if 1 <= x <= kmax):
                for row in ([tiny] + [0.0] * (k - 1), [big] * k, [-big] * k, [1.0] * k):
                    sgm = scale_row(row, 2, rnd)[0]
                    lo, hi = min(lo, sgm), max(hi, sgm)
            need((lo, hi) == (sb["sigma_min"], sb["sigma_max"]),
                 f"N=2 {rnd} kmax={kmax}: brute [{lo},{hi}] vs {sb['sigma_min'], sb['sigma_max']}")
    return ("brute force matches at kmax=4 (N=2,3,8,20) and kmax up to (T-1)^2 (N=2); "
            "per-N sigma range attained for k<=2^40; u monotone in S and k")


def _budget(T, k, rounding):
    """(c, R) of the §3.2 rule c*S*2^(2u) <= R*2^F."""
    return (4, (2 * T - kappa(k))**2) if rounding == RN else (1, T * T)


def _row_guarantee(row, N, rounding):
    """Every §3.2/§3.3/§3.8 per-row claim for one row, in exact rationals:
      premise    ||x||^2 2^F <= S < ||x||^2 2^F + k'  (S is the ceiling sum, k' nonzeros)
      rule       u = sigma + gamma is the LARGEST integer with c S 2^(2u) <= R 2^F
      consequence c ||a||^2 2^(2 sigma) <= R   (what the norm proof uses)
      guarantee  ||a'||_2 <= T
      §3.8       2^-sigma < 2 ||a|| s / d,  s^2 = 1 + k 2^-F, d = T - kappa/2 (RN) or T (TRUNC)
    Returns whether DESIGN's uncorrected 2^-sigma <= 2||a||/d holds (None for a zero row)."""
    T = constants(N)["T"]
    k = len(row)
    sig, S, gam = scale_row(row, N, rounding)
    if S == 0:
        need(sig == 0 and all(v == 0.0 for v in row), "zero row")
        return None
    na2 = sum(Fraction(v)**2 for v in row)
    nx2F = na2 * Fraction(2)**(F_BITS - 2 * gam)            # ||x||^2 2^F
    nz = sum(1 for v in row if v != 0.0)
    need(nx2F <= S < nx2F + nz, f"N={N} premise ||x||^2 2^F <= S (ceiling) violated")
    c, R = _budget(T, k, rounding)
    u = sig + gam
    need(_le_scaled(c * S, 2 * u, R, F_BITS) and not _le_scaled(c * S, 2 * u + 2, R, F_BITS),
         f"N={N} {rounding} u not maximal")
    need(c * na2 * Fraction(2)**(2 * sig) <= R, f"N={N} {rounding} consequence c||a||^2 2^2sigma <= R")
    aq = [quantize(v, sig, rounding)[0] for v in row]
    need(sum(x * x for x in aq) <= T * T, f"N={N} {rounding} ||a'|| > T")
    d2 = Fraction(2 * T - kappa(k), 2)**2 if rounding == RN else Fraction(T)**2
    inv = Fraction(1) / Fraction(2)**(2 * sig)              # (2^-sigma)^2
    need(inv < 4 * na2 * (1 + Fraction(k, 2**F_BITS)) / d2, "corrected 2^-sigma bound")
    return inv <= 4 * na2 / d2


@check("§3.2/§3.3 norm guarantee")
def cert_shift_rule_guarantee():
    """On structured random rows (_row_guarantee): S premise; u maximal; consequence;
    ||a'||_2 <= T; 2^-sigma < 2||a|| s/d."""
    rng = random.Random(5)
    n = 0
    for N in (2, 3, 4, 8, 14, 20):
        T = constants(N)["T"]
        for rnd in (RN, TRUNC):
            for k in (1, 2, 3, 5, 16, 40):
                if rnd == RN and kappa(k) >= T:
                    continue
                for _ in range(25):
                    _row_guarantee(_rand_row(rng, k), N, rnd)
                    n += 1
    return f"{n} rows"


# Rows on which DESIGN §3.8's uncorrected 2^-sigma <= 2||a||/(T - kappa/2) is false
# (k = 2, RN; found by the reviewer, reproduced by _boundary_rows below).
DESIGN_38_COUNTEREXAMPLES = [
    (14, [1.4954333905227286, 2.881201632831676e-08], 53),
    (20, [1.1373509086410998, 2.4919464713946127e-08], 76),
]


def _boundary_rows(N, rounding, side, k=2, tries=4000):
    """Two-entry rows X1^2 + X2^2 (X = x 2^52 integers) whose y = ||x||^2 2^F sits
    within 2^-16 of the rule boundary V = R 2^F / (c 2^(2u)) for the one u with
    V in [2^F, 2^(F+2)):
      side 'below': y < V < S  (the ceiling pushes S over V; u drops by one)
      side 'above': floor-sum <= V < y  (a FLOOR-sum S would keep u: unsound)
    Yields rows (as doubles; exact since X < 2^53)."""
    T = constants(N)["T"]
    c, R = _budget(T, k, rounding)
    for u in range(-80, 80):
        V = Fraction(R * 2**F_BITS, c * 2**(2 * u)) if u >= 0 else Fraction(R * 2**F_BITS * 2**(-2 * u), c)
        if not (2**F_BITS <= V < 2**(F_BITS + 2)) or V.denominator == 1:
            continue
        t = V * 2**44                                  # target for X1^2 + X2^2
        found = 0
        for j in range(tries):
            if side == "below":
                X1 = math.isqrt(math.floor(t) - 2**52 - j * 2**40)
                rem = t - X1 * X1
                X2 = math.isqrt(math.floor(rem))
                if X2 * X2 == rem:
                    X2 -= 1
            else:
                X1 = math.isqrt(math.floor(t)) - j
                X2 = math.isqrt(math.floor(t - X1 * X1)) + 1
            if not (2**52 <= X1 < 2**53 and 0 < X2 < X1):
                continue
            y = Fraction(X1 * X1 + X2 * X2, 2**44)
            Sc = sum(ceil_div(X * X, 2**44) for X in (X1, X2))
            Sf = sum((X * X) // 2**44 for X in (X1, X2))
            ok = (y < V < Sc) if side == "below" else (Sf <= V < y)
            if ok:
                found += 1
                yield [X1 / 2**52, X2 / 2**52] + [0.0] * (k - 2)
                if found >= 8:
                    break


@check("§3.2 ceiling in S on boundary rows (adversarial)")
def cert_s_ceiling_boundary():
    """Adversarial rows on the rounding boundary of S (_boundary_rows), both roundings:
    every §3.2/§3.3 claim holds (_row_guarantee). 'above' rows show the ceiling is
    necessary (a floor-sum S gives a u with c||x||^2 2^(2u) > R 2^F); 'below' rows
    and DESIGN_38_COUNTEREXAMPLES show DESIGN §3.8's uncorrected
    2^-sigma <= 2||a||/(T - kappa/2) is false (the sqrt(1 + k 2^-F) factor is needed)."""
    n_rows, floor_bad, design_false = 0, 0, []
    for N in (2, 3, 8, 14, 20):
        T = constants(N)["T"]
        for rnd in (RN, TRUNC):
            c, R = _budget(T, 2, rnd)
            for side in ("below", "above"):
                for row in _boundary_rows(N, rnd, side):
                    ok = _row_guarantee(row, N, rnd)
                    n_rows += 1
                    if side == "below" and not ok:
                        design_false.append((N, rnd))
                    if side == "above":
                        Sf = sum(math.floor(Fraction(v)**2 * 2**F_BITS) for v in row)   # gamma = 0
                        uf = largest_u(c * Sf, R)
                        nx2 = sum(Fraction(v)**2 for v in row)
                        if c * nx2 * Fraction(2)**(2 * uf) > R:
                            floor_bad += 1
    for N, row, sig in DESIGN_38_COUNTEREXAMPLES:
        need(scale_row(row, N, RN)[0] == sig, f"regression sigma N={N}")
        need(_row_guarantee(row, N, RN) is False, f"regression N={N}: uncorrected bound holds?")
    need(n_rows >= 40, f"only {n_rows} boundary rows generated")
    need(floor_bad > 0, "floor-sum S never unsound: ceiling necessity not demonstrated")
    need(len(design_false) > 0, "uncorrected §3.8 inequality never fails on 'below' rows")
    return (f"{n_rows} boundary rows OK; floor-S unsound on {floor_bad}; uncorrected §3.8 "
            f"2^-sigma bound false on {len(design_false)} (+2 regression rows)")


def _rand_double(rng, emin, emax):
    mant = rng.getrandbits(MANT - 1) | (1 << (MANT - 1))
    return math.ldexp(mant, rng.randint(emin, emax) - (MANT - 1)) * rng.choice((1, -1))


def _rand_row(rng, k, emin=-200, emax=200):
    fam = rng.randrange(6)
    if fam == 0:   # general spread
        return [_rand_double(rng, emin, emax) for _ in range(k)]
    if fam == 1:   # narrow exponent band, some zeros
        e = rng.randint(emin, emax)
        return [0.0 if rng.random() < 0.2 else _rand_double(rng, e - 3, e) for _ in range(k)]
    if fam == 2:   # all at max mantissa (S_max-like)
        e = rng.randint(emin, emax)
        return [rng.choice((1, -1)) * math.ldexp(2**MANT - 1, e - MANT + 1) for _ in range(k)]
    if fam == 3:   # small odd integers times a common power: ties in RN likely
        e = rng.randint(emin, emax)
        return [math.ldexp(rng.choice((1, -1)) * (2 * rng.randint(0, 40) + 1), e) for _ in range(k)]
    if fam == 4:   # one dominant entry
        e = rng.randint(emin, emax)
        return [math.ldexp(1.0, e)] + [_rand_double(rng, e - 60, e - 1) for _ in range(k - 1)]
    return [0.0] * k if rng.random() < 0.3 else \
        [_rand_double(rng, emin, emax) if rng.random() < 0.5 else 0.0 for _ in range(k)]


@check("§3.2-§3.8 end-to-end exactness and error contract")
def cert_end_to_end():
    """Exact real-width model: z = sum a'b' exactly; sum|a'||b'| <= T^2 <= H;
    |D - sum ab| <= term1*s + k 2^(-sigma-tau-2) + u|D| (RN; TRUNC analogue)."""
    rng = random.Random(42)
    count = 0
    for N in (2, 3, 4, 6, 8, 14, 16, 20):
        c = constants(N)
        T, H = c["T"], c["H"]
        for rnd in (RN, TRUNC):
            for k in (2, 3, 7, 16, 33):
                for it in range(40):
                    if it == 39:   # subnormal row against a huge column
                        a = [math.ldexp(rng.randint(1, 2**20), -1074) for _ in range(k)]
                        b = [_rand_double(rng, 900, 1000) for _ in range(k)]
                    else:
                        a, b = _rand_row(rng, k), _rand_row(rng, k)
                    r = emulate_element(a, b, N, rnd, chunk=3)
                    exact_q = sum(x * y for x, y in zip(r["a_q"], r["b_q"]))
                    need(r["z"] == exact_q, f"N={N} CRT result != A'B'")
                    need(sum(abs(x * y) for x, y in zip(r["a_q"], r["b_q"])) <= T * T <= H,
                         f"N={N} uniqueness bound")
                    sb = shift_bounds(N, k, rnd)
                    for s in (r["sigma"], r["tau"]):
                        need(sb["sigma_min"] <= s <= sb["sigma_max"], "sigma outside shift_bounds")
                    _row_guarantee(a, N, rnd)
                    _row_guarantee(b, N, rnd)
                    # §3.8 contract with the PUBLISHED error_guidance pieces (floats,
                    # widened by 2^-40 relative to cover their own rounding):
                    #   |D - sum ab| <= term1 ||a|| ||b|| + second + u|D|
                    #   second = k 2^(-sigma-tau-2) (RN) | k 2^(-sigma-tau) (TRUNC)
                    #   second <= term2 ||a|| ||b||
                    eg = error_guidance(N, k, rnd)
                    widen = 1 + Fraction(1, 2**40)
                    t1, t2 = Fraction(eg["term1"]) * widen, Fraction(eg["term2"]) * widen
                    exact = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
                    D = Fraction(r["D"])
                    err = abs(D - exact)
                    sh = Fraction(2)**(-(r["sigma"] + r["tau"]))
                    second = k * sh / 4 if rnd == RN else k * sh
                    na2 = sum(Fraction(v)**2 for v in a)
                    nb2 = sum(Fraction(v)**2 for v in b)
                    if na2 and nb2:
                        need(second * second <= t2 * t2 * na2 * nb2, f"N={N} {rnd} k={k}: second > term2")
                    rest = err - second - UNIT_ROUNDOFF * abs(D)
                    need(rest <= 0 or rest * rest <= t1 * t1 * na2 * nb2,
                         f"N={N} {rnd} k={k}: error contract violated")
                    count += 1
    return f"{count} elements, N in 2..20, chunk=3"


@check("§3.2 kappa slack is necessary (mutation)")
def cert_kappa_slack_needed():
    """Adversarial equal-entry rows v*[1]*k, v = 1 + j/2^10: the real RN rule keeps
    ||a'||_2 <= T on all of them, while RN with TRUNC's budget (kappa slack removed,
    4S2^2u <= (2T)^2 2^F) breaks it. Shows the slack is necessary at real width."""
    found, checked = None, 0
    for N in (2, 3):
        T = constants(N)["T"]
        for k in range(2, 65):
            for j in range(1024):
                v = Fraction(2**10 + j, 2**10)          # gamma = 0, x = v
                t = v * v * 2**F_BITS
                S = k * ceil_div(t.numerator, t.denominator)
                for mutated in (False, True):
                    R = (2 * T)**2 if mutated else (2 * T - kappa(k))**2
                    u = largest_u(4 * S, R)
                    a = round(v * Fraction(2)**u)      # ties-to-even
                    bad = k * a * a > T * T
                    if not mutated:
                        need(not bad, f"real RN rule violated: N={N} k={k} v={v}")
                        checked += 1
                    elif bad and found is None:
                        found = (N, k, v, k * a * a, T * T)
    need(found is not None, "mutation not detected: slack necessity not demonstrated")
    N, k, v, lhs, rhs = found
    return f"{checked} rows OK; mutant fails at N={N} k={k} v={v}: ||a'||^2={lhs} > T^2={rhs}"


@check("§3.3 quantization rounding (unit)")
def cert_quantize_unit():
    """RN is round-half-to-EVEN and TRUNC rounds toward zero (DESIGN §3.3; bitwise
    determinism §1 depends on the exact tie rule); a' = mu 2^eps with mu odd or 0
    on the exact path; subnormal and huge-exponent edge cases."""
    xs = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -0.75, 0.75, 3.5, -3.5, 1.25, -1.25]
    want = {RN: [0, 2, 2, 0, -2, -2, -1, 1, 4, -4, 1, -1],
            TRUNC: [0, 1, 2, 0, -1, -2, 0, 0, 3, -3, 1, -1]}
    n = 0
    for rnd in (RN, TRUNC):
        for sigma in (5, 0, -7, 1072):
            for x, w in zip(xs, want[rnd]):
                v = math.ldexp(x, -sigma)                  # v 2^sigma = x exactly
                need(Fraction(v) * Fraction(2)**sigma == Fraction(x), "test value not exact")
                got = quantize(v, sigma, rnd)
                need(got[0] == w, f"{rnd} quantize({x} at sigma={sigma}) = {got[0]}, want {w}")
                n += 1
    tiny = 5e-324
    for rnd in (RN, TRUNC):
        need(quantize(tiny, 1074, rnd) == (1, 1, 0), "2^-1074 * 2^1074")
        need(quantize(tiny, 1073, rnd)[0] == 0, "half of the smallest subnormal")
        need(quantize(-3 * tiny, 1073, rnd)[0] == (-2 if rnd == RN else -1), "-1.5 from subnormal")
        need(quantize(3.0, 100, rnd) == (3 * 2**100, 3, 100), "exact path mu 2^eps")
        need(quantize(-12.0, 3, rnd) == (-96, -3, 5), "exact path strips factors of 2")
        need(quantize(0.0, 50, rnd) == (0, 0, 0), "zero")
        n += 6
    return f"{n} cases"


@check("§3.3 TRUNC toward zero is necessary (mutation)")
def cert_trunc_toward_zero_needed():
    """Equal-entry NEGATIVE rows -v*[1]*k, v = 1 + j/2^10, under TRUNC: the real rule
    (trunc toward zero) keeps ||a'||_2 <= T on all, while floor (toward -inf) breaks it."""
    found, checked = None, 0
    for N in (2, 3):
        T = constants(N)["T"]
        for k in range(1, 65):
            for j in range(1024):
                v = Fraction(2**10 + j, 2**10)
                t = v * v * 2**F_BITS                       # gamma = 0, so sigma = u
                sig = shift_u(k * ceil_div(t.numerator, t.denominator), T, k, TRUNC)
                if j % 97 == 0:                             # spot-check against scale_row
                    need(scale_row([-float(v)] * k, N, TRUNC)[0] == sig, "closed-form sigma")
                a = quantize(-float(v), sig, TRUNC)[0]
                need(k * a * a <= T * T, f"TRUNC rule violated: N={N} k={k} v=-{v}")
                checked += 1
                af = math.floor(-v * Fraction(2)**sig)
                if found is None and k * af * af > T * T:
                    found = (N, k, -v, k * af * af, T * T)
    need(found is not None, "floor never breaks ||a'|| <= T: necessity not demonstrated")
    N, k, v, lhs, rhs = found
    return f"{checked} rows OK; floor fails at N={N} k={k} v={v}: ||a'||^2={lhs} > T^2={rhs}"


@check("§3.8 DESIGN guidance table")
def cert_guidance_table():
    """2 sqrt(k)/T matches the DESIGN table (2 s.f.); N=16 rows: < u at 1e4, < 2u (not u) at 2^17."""
    table = {6: (1.8e-5, 6.5e-5), 8: (7.6e-8, 2.8e-7), 9: (5.0e-9, 1.8e-8),
             10: (3.3e-10, 1.2e-9), 12: (1.5e-12, 5.5e-12), 14: (7.4e-15, 2.7e-14)}
    for N, vals in table.items():
        for k, t in zip((10**4, 2**17), vals):
            g = error_guidance(N, k)["guidance"]
            half_digit = 0.05 * 10**math.floor(math.log10(t))
            need(abs(g - t) <= half_digit * 1.0001, f"N={N} k={k}: {g:.3e} vs {t}")
    u = 2.0**-53
    need(error_guidance(16, 10**4)["guidance"] < u, "N=16 k=1e4 below u")
    need(error_guidance(16, 2**17)["guidance"] < 2 * u, "N=16 k=2^17 below 2u")
    # term1/term2 against exact rationals (relative 2^-45): RN d = T - kappa/2, TRUNC d = T
    for N in (2, 8, 14, 20):
        T = constants(N)["T"]
        for k in (1, 2, 33, 10**4, 2**17, rn_feasible_kmax(2)):
            for rnd in (RN, TRUNC):
                if rnd == RN and kappa(k) >= T:
                    continue
                eg = error_guidance(N, k, rnd)
                s2 = 1 + Fraction(k, 2**F_BITS)
                if rnd == RN:
                    d2 = Fraction(2 * T - kappa(k), 2)**2
                    want1, want2 = 4 * k * s2 / d2, k * s2 / d2          # term1^2, term2
                else:
                    want1, want2 = 16 * k * s2 / T**2, 4 * k * s2 / T**2
                for got, want in ((Fraction(eg["term1"])**2, want1), (Fraction(eg["term2"]), want2)):
                    need(abs(got - want) <= want / 2**45, f"error_guidance N={N} k={k} {rnd}")
    # informational: at k=2^17, N=16 guidance exceeds u (DESIGN wording)
    g = error_guidance(16, 2**17)["guidance"]
    return f"table OK; N=16 k=2^17 guidance {g:.2e} = {g / u:.2f}u"


def _native_fma_dot(a, b):
    """§3.7 native path in exact arithmetic: acc = +0; acc = RN(a_h b_h + acc) in order
    (float(Fraction) is correctly rounded)."""
    acc = 0.0
    for x, y in zip(a, b):
        acc = float(Fraction(x) * Fraction(y) + Fraction(acc))
    return acc


@check("§6f composed error bound (rigorous estimate of imd_err_estimate)")
def cert_error_estimate():
    """error_estimate's rigorous bound holds on emulated elements (emulate_element, both
    roundings) and on the native FMA chain (DIRECT), for every metric, with the TRUE
    statistics as the caller's: nu = ||a||_1/||a||_2 (an upper rational), rho =
    ||a|| ||b|| / |x| and norm_product = ||a|| ||b|| (upper rationals). The check uses the
    bound's LOWER enclosure, in exact rationals. Also: with nu = 0 (sqrt k) the unit bound is
    error_guidance's term1 + term2; the enclosures are ordered and tight (<= 2^-80 relative)."""
    rng = random.Random(61)
    n = 0
    for N in (3, 6, 8, 14, 20):
        T = constants(N)["T"]
        for rnd in (RN, TRUNC):
            for k in (2, 5, 16, 33):
                if rnd == RN and kappa(k) >= T:
                    continue
                eg = error_guidance(N, k, rnd)
                e0 = error_estimate(N, k, rnd)["unit_bound"]
                ref = Fraction(eg["term1"]) + Fraction(eg["term2"])
                need(abs(e0[0] - ref) <= ref / 2**40 and e0[0] <= e0[1] <= e0[0] * (1 + Fraction(1, 2**80)),
                     f"N={N} {rnd} k={k}: unit bound vs error_guidance / enclosure")
                for _ in range(12):
                    a = _rand_row(rng, k, -60, 60)
                    b = _rand_row(rng, k, -60, 60)
                    r = emulate_element(a, b, N, rnd, chunk=4)
                    _check_composed(N, k, rnd, "EMULATE", a, b, r["D"])
                    n += 1
    # review 2026-10-05: larger k at small N, where the bound is closest to the error, and
    # constant non-representable rows (their quantization errors add coherently)
    for N in (3, 4, 5):
        for rnd in (RN, TRUNC):
            for k in (600, 1200):
                for fam in ("const", "const_neg", "rand"):
                    if fam == "rand":
                        a, b = _rand_row(rng, k, -20, 20), _rand_row(rng, k, -20, 20)
                    else:
                        va, vb = 0.1 * (1 + rng.random()), 0.3 * (1 + rng.random())
                        a = [va] * k
                        b = [vb if fam == "const" else -vb] * k
                    r = emulate_element(a, b, N, rnd, chunk=257)
                    _check_composed(N, k, rnd, "EMULATE", a, b, r["D"])
                    n += 1
    for k in (1, 2, 7, 64):            # DIRECT: the native chain
        for _ in range(40):
            a, b = _rand_row(rng, k, -60, 60), _rand_row(rng, k, -60, 60)
            _check_composed(8, k, RN, "DIRECT", a, b, _native_fma_dot(a, b))
            n += 1
    return (f"{n} elements (EMULATE N in 3..20 both roundings, k up to 1200 at N = 3..5 with "
            "constant rows; DIRECT), 3 metrics each")


def _check_composed(N, k, rnd, dispatch, a, b, D):
    exact = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
    err = abs(Fraction(D) - exact)
    na2 = sum(Fraction(v)**2 for v in a)
    nb2 = sum(Fraction(v)**2 for v in b)
    if na2 == 0 or nb2 == 0:
        need(err == 0, "zero row: exact zero")
        return
    na_lo = sqrt_enclosure(na2)[0]
    nb_lo = sqrt_enclosure(nb2)[0]
    nu_a = float(sum(abs(Fraction(v)) for v in a) / na_lo)   # >= the true ratio (rounded up below)
    nu_b = float(sum(abs(Fraction(v)) for v in b) / nb_lo)
    nu_a, nu_b = math.nextafter(nu_a, math.inf), math.nextafter(nu_b, math.inf)
    nab_hi = sqrt_enclosure(na2 * nb2)[1]
    npd = math.nextafter(float(nab_hi), math.inf)
    for nus in ((0.0, 0.0), (max(nu_a, 1.0), max(nu_b, 1.0))):
        kw = dict(l1_ratio_a=nus[0], l1_ratio_b=nus[1])
        bn = error_estimate(N, k, rnd, dispatch, 1, ERR_NORMWISE, **kw)["bound"][0]
        need(err * err <= bn * bn * na2 * nb2, f"N={N} {rnd} {dispatch} k={k}: NORMWISE")
        ba = error_estimate(N, k, rnd, dispatch, 1, ERR_ABSOLUTE, norm_product=npd, **kw)["bound"][0]
        need(err <= ba, f"N={N} {rnd} {dispatch} k={k}: ABSOLUTE")
        if exact != 0:
            rho = math.nextafter(float(nab_hi / abs(exact)), math.inf)
            if math.isfinite(rho):
                be = error_estimate(N, k, rnd, dispatch, 1, ERR_ELEMENTWISE, cond=max(rho, 1.0),
                                    **kw)["bound"][0]
                need(err <= be * abs(exact), f"N={N} {rnd} {dispatch} k={k}: ELEMENTWISE")


@check("§2-§3.6 scaled TLA+ parameters")
def cert_scaled_tla():
    """scaled_constants reproduces the real constants at (8, 32) and the parameters the
    TLA+ model headers state (MC_OzakiCRT, MC_CRTRealization, MC_OzakiScaling);
    it refuses accumulators too narrow for one product (K_c = 0)."""
    real = scaled_constants(RES_BITS, ACC_BITS, N_MAX, limb_bits=LIMB, k_align=K_CHUNK_ALIGN)
    need(real["moduli"] == REAL_MODULI and real["K_c"] == K_C and real["K_c_prime"] == K_C_PRIME,
         "real-width scaled_constants")
    for N in range(N_MIN, N_MAX + 1):
        sc = scaled_constants(RES_BITS, ACC_BITS, N, limb_bits=LIMB)
        need(sc["limbs"] == constants(N)["L32"], f"N={N} limbs")
    # (res_bits, acc_bits, N) -> (moduli, P, H, K_c) as in MC_OzakiCRT.tla's header
    crt = {(2, 5, 2): ([4, 3], 12, 5, 3), (3, 6, 4): ([8, 7, 5, 3], 840, 419, 1),
           (3, 6, 3): ([8, 7, 5], 280, 139, 1), (3, 7, 4): ([8, 7, 5, 3], 840, 419, 3),
           (4, 8, 5): ([16, 15, 13, 11, 7], 240240, 120119, 1)}
    for (r, acc, N), (ps, P, H, kc) in crt.items():
        sc = scaled_constants(r, acc, N)
        need((sc["moduli"], sc["P"], sc["H"], sc["K_c"]) == (ps, P, H, kc), f"OzakiCRT ({r},{acc},{N}): {sc}")
    # MC_CRTRealization: minimal word bits and exact peak
    for (r, N), (wb, pk) in {(3, 4): (11, 1497), (4, 4): (17, 65857)}.items():
        sc = scaled_constants(r, 2 * r + 1, N)
        need((sc["word_bits"], sc["crt_peak"]) == (wb, pk), f"CRTRealization ({r},{N}): {sc}")
        need(sc["crt_peak"] >= 2**(wb - 1), "one bit fewer really overflows")
    # MC_OzakiScaling toy moduli: <<8,7>> T 5, <<8,7,5>> T 11
    for N, (P, H, T) in {2: (56, 27, 5), 3: (280, 139, 11)}.items():
        sc = scaled_constants(3, 7, N)
        need((sc["P"], sc["H"], sc["T"]) == (P, H, T), f"OzakiScaling N={N}")
    for bad in ((3, 5, 2), (8, 15, 2), (8, 32, 2, None, 2**18)):
        try:
            scaled_constants(*bad)
        except ValueError:
            continue
        raise CertFail(f"scaled_constants{bad} accepted K_c' = 0")
    return f"{len(crt) + 4} TLA+ parameter sets; real (8,32) reproduces K_c, K_c', moduli"


@check("§2/§3.3 DESIGN text numbers")
def cert_design_text():
    """log2 P values; |a'| <= T reaches 2^77 at N=20; kappa/2 slack < 2^-20 for N>=8 (k bound)."""
    for N, v in ((8, 63.58), (9, 71.41), (10, 79.24), (14, 110.16), (15, 117.78), (16, 125.38)):
        need(round(math.log2(constants(N)["P"]), 2) == v, f"log2 P at N={N}")
    T20 = constants(20)["T"]
    need(2**77 <= T20 < 2**78, "T(20)")
    need(constants(16)["T"] < 2**63 <= constants(17)["T"], "int64 a' up to N=16 only")
    # kappa/(2T) < 2^-20  <=>  kappa < T 2^-19 ; largest k at N=8:
    T8 = constants(8)["T"]
    kap_ok = (2 * T8 - 1) // 2**20          # largest kappa with kappa * 2^20 < 2T
    need(kap_ok * 2**20 < 2 * T8 <= (kap_ok + 1) * 2**20, "kap_ok")
    k_ok = kap_ok * kap_ok                  # largest k with kappa(k) <= kap_ok
    need(kappa(2**17) * 2**20 < 2 * T8, "slack at k=2^17, N=8")
    return f"log2P OK; kappa/2 slack < 2^-20 at N=8 only for k <= {k_ok} (~2^{math.log2(k_ok):.1f})"


@check("§4a workspace")
def cert_workspace():
    """Closed form = sum of aligned regions; regions disjoint, 256-aligned, in order;
    sub-buffer strides 16-B aligned; zero/DIRECT/k=0 rules; monotone in every argument."""
    ms = [0, 1, 15, 16, 17, 100, 257]
    ks = [0, 1, 2, 15, 16, 17, 131008, 131009, 131056, 131057, 196624, 262200]
    bs = [0, 1, 3]
    n_cases = 0
    for N in (2, 9, 14, 20):
        for g in sorted({1, 2, N // 2 or 1, N}):
            for mode in (EMULATE, DIRECT):
                for vec in (VEC_GENERATE, VEC_RESIDENT):
                    for m in ms:
                        for n in ms:
                            for k in ks:
                                for b in bs:
                                    ws = workspace_size(m, n, k, b, N, g, mode, vec)
                                    regs = region_offsets(m, n, k, b, N, g, mode, vec)
                                    _check_layout(m, n, k, b, N, g, mode, ws, regs, vec)
                                    n_cases += 1
    # monotonicity (EMULATE, k >= 1): a planner may price with it
    for N in (2, 14):
        for g in (1, N):
            base = (17, 33, 5, 2)
            ws0 = workspace_size(*base, N, g)
            for i in range(4):
                bigger = list(base)
                bigger[i] += 1
                need(workspace_size(*bigger, N, g) >= ws0, "not monotone")
            if g < N:
                need(workspace_size(*base, N, g + 1) >= ws0, "not monotone in g")
            if N < N_MAX:
                need(workspace_size(*base, N + 1, g) >= ws0, "not monotone in N")
    # Hand-computed worked examples (bytes typed from DESIGN §4a by hand, not
    # from the code): m, n, k all different, b > 1, g < N, k and n not % 16.
    #  (17,33,5,3,N=9,g=2): k^=16 n^=48; sA 102->256 sB 198->256 fA 51->256
    #   fB 99->256 vec 8*119=952->1024 RA 2*3*17*16=1632->1792
    #   RB 2*3*48*16=4608 W 9*3*17*48=22032->22272; total 30720
    #  (1000,3,40,2,N=20,g=7): k^=48 n^=16; sA 4000->4096 sB 12->256
    #   fA 2000->2048 fB 6->256 vec 8*2092=16736->16896 RA 7*2*1000*48=672000
    #   RB 7*2*16*48=10752 W 20*2*1000*16=640000; total 1346304
    #  DIRECT / k = 0 (17,33,0,3): vec 8*(34+66+0+9)=872 -> 1024
    worked = [
        ((17, 33, 5, 3, 9, 2, EMULATE), 30720,
         [("sA", 0, 102), ("sB", 256, 454), ("fA", 512, 563), ("fB", 768, 867),
          ("vec", 1024, 1976), ("RA", 2048, 3680), ("RB", 3840, 8448), ("W", 8448, 30480)]),
        ((1000, 3, 40, 2, 20, 7, EMULATE), 1346304,
         [("sA", 0, 4000), ("sB", 4096, 4108), ("fA", 4352, 6352), ("fB", 6400, 6406),
          ("vec", 6656, 23392), ("RA", 23552, 695552), ("RB", 695552, 706304),
          ("W", 706304, 1346304)]),
        ((17, 33, 5, 3, 9, 2, DIRECT), 1024, [("vec", 0, 952)]),
        ((17, 33, 0, 3, 9, 0, EMULATE), 1024, [("vec", 0, 872)]),
        #  K-empty (k = 1) dispatches to DIRECT: vec 8*(34+66+2+9)=888 -> 1024
        ((17, 33, 1, 3, 9, 0, EMULATE), 1024, [("vec", 0, 888)]),
        ((17, 0, 5, 3, 9, 2, EMULATE), 0, []),
        #  v1.2 RESIDENT (17,33,5,3,N=9,g=2): the GENERATE example without vec (1024):
        #   sA 0 sB 256 fA 512 fB 768 RA 1024..2656 RB 2816..7424 W 7424..29456; 29696
        ((17, 33, 5, 3, 9, 2, EMULATE, VEC_RESIDENT), 29696,
         [("sA", 0, 102), ("sB", 256, 454), ("fA", 512, 563), ("fB", 768, 867),
          ("RA", 1024, 2656), ("RB", 2816, 7424), ("W", 7424, 29456)]),
        ((17, 33, 5, 3, 9, 2, DIRECT, VEC_RESIDENT), 0, []),
        ((17, 33, 0, 3, 9, 0, EMULATE, VEC_RESIDENT), 0, []),
    ]
    for args, want, regs_want in worked:
        need(workspace_size(*args) == want, f"worked example {args}: {workspace_size(*args)} != {want}")
        need(region_offsets(*args) == regs_want, f"worked example {args} offsets")
    # plan device bytes: (17,33,5,3) RESIDENT 8*(34+66+10+9) = 952; GENERATE / empty 0
    need(plan_device_bytes(17, 33, 5, 3, VEC_RESIDENT) == 952, "plan device bytes")
    need(plan_device_bytes(17, 33, 5, 3) == 0 and plan_device_bytes(0, 33, 5, 3, VEC_RESIDENT) == 0,
         "plan device bytes (GENERATE / empty)")
    ex = workspace_size(8192, 8192, 8192, 1, 14, 14)
    return f"{n_cases} shapes, {len(worked)} hand-computed examples; 8192^3 N=14: {ex / 2**30:.3f} GiB"


@check("§6b.4 packed layout")
def cert_pack_layout():
    """DESIGN v1.7 §6b.4 (format 1) and §6b.6: pack_bytes = sum of the aligned regions sig,
    flag, res (256-aligned, in order, gaps < 256); res = N b r k_hat + (ceil(r)_16 - r) k_hat;
    every K4 operand pointer into a pack (plane l, batch lambda, row rho, K-chunk c, any
    residue group) 16-byte aligned; the post-swap B role (n_hat = ceil(r)_16 rows per block,
    batch stride r k_hat) reads only inside res; empty packs are 0 bytes; overflow past
    uint64 raises (INVALID_VALUE); the packed workspace (§6b.6) is §4a without sX, fX, RX of
    each packed role. Hand-computed examples."""
    n_cases = 0
    for N in (2, 9, 14, 20):
        for b in (0, 1, 3, 7):
            for r in (0, 1, 15, 16, 17, 33):
                for k in (0, 1, 2, 15, 16, 17, 131008, 131009, 131056, 131057, 131072):
                    regs, total = pack_layout(b, r, k, N)
                    n_cases += 1
                    if b * r * k == 0:
                        need(regs == [] and total == 0 and pack_ws_bytes(b, r, k) == 0,
                             "empty pack")
                        continue
                    kh, nh = ceil_div(k, 16) * 16, ceil_div(r, 16) * 16
                    lit = {"sig": 2 * b * r, "flag": b * r,
                           "res": N * b * r * kh + (nh - r) * kh}
                    need([x[0] for x in regs] == ["sig", "flag", "res"], "region order")
                    off = 0
                    for name, start, nbytes in regs:
                        need(nbytes == lit[name], f"{name} bytes")
                        need(start == off and start % 256 == 0, f"{name} offset")
                        off = ceil_div(start + nbytes, 256) * 256
                    need(total == off and total - (regs[-1][1] + regs[-1][2]) < 256, "total")
                    res0 = regs[2][1]
                    # every 16-byte K4 access: plane l (any group start g0 = l), batch lambda,
                    # row rho (role B: up to n_hat - 1), K-chunk c
                    for l in sorted({0, 1 % N, N - 1}):
                        for lam in sorted({0, b - 1}):
                            for rho in sorted({0, r - 1, nh - 1}):
                                for c in range(ceil_div(k, K_C_PRIME)):
                                    a = res0 + l * b * r * kh + (lam * r + rho) * kh + c * K_C_PRIME
                                    need(a % 16 == 0, "K4 pointer alignment")
                                    # v1.13 §4b: the chunk offset keeps the row's 64-B phase
                                    need((a - c * K_C_PRIME) % 64 == a % 64, "chunk phase")
                    # role B: the last byte K4 can read (plane N-1, batch b-1, row n_hat-1,
                    # k_hat-1) is inside res; the bound is tight (the tail is exactly needed)
                    last = res0 + (N - 1) * b * r * kh + ((b - 1) * r + nh - 1) * kh + kh - 1
                    need(last == res0 + lit["res"] - 1, "B-role reads end exactly at the tail")
                    need(pack_ws_bytes(b, r, k) == ceil_div(8 * (r + k + b), 256) * 256, "pack ws")
    # Hand-computed (DESIGN §6b.4 typed by hand):
    #  (b=3, r=17, k=5, N=9): k^=16; sig 102 -> 256; flag 51 -> 256;
    #   res 9*3*17*16 + (32-17)*16 = 7344 + 240 = 7584 -> 7680; total 8192
    #  (1, 1, 1000, 14): k^=1008; sig 2 -> 256; flag 1 -> 256; res 14*1008 + 15*1008 = 29232
    #   -> 29440; total 29952
    #  (2, 32, 16, 2): sig 128 -> 256; flag 64 -> 256; res 2*2*32*16 = 2048; total 2560
    worked = [((3, 17, 5, 9), 8192, [("sig", 0, 102), ("flag", 256, 51), ("res", 512, 7584)]),
              ((1, 1, 1000, 14), 29952, [("sig", 0, 2), ("flag", 256, 1), ("res", 512, 29232)]),
              ((2, 32, 16, 2), 2560, [("sig", 0, 128), ("flag", 256, 64), ("res", 512, 2048)]),
              ((0, 5, 5, 8), 0, []), ((2, 3, 0, 8), 0, [])]
    for args, want, regs_want in worked:
        regs, total = pack_layout(*args)
        need(total == want and regs == regs_want, f"pack worked example {args}: {regs} {total}")
    need(pack_ws_bytes(3, 17, 5) == 256 and pack_ws_bytes(1, 1, 1000) == 8192, "pack ws examples")
    # overflow: N b r k_hat beyond 2^64
    for args in ((1, 2**40, 2**30, 20), (2**20, 2**20, 2**30, 2)):
        try:
            pack_layout(*args)
            need(False, f"no overflow for {args}")
        except OverflowError:
            pass
    # §6b.6 packed workspace: (17,33,5,3,N=9,g=2), cert_workspace's example (30720 bytes):
    #  packed A: no sA (256), fA (256), RA (1792): 28416; sB 0 fB 256 vec 512 RB 1536 W 6144
    #  packed B: no sB, fB, RB (4608): 25600; sA 0 fA 256 vec 512 RA 1536 W 3328
    #  both: vec 0, W 1024: 23296
    pw = [(("A",), 28416, [("sB", 0, 198), ("fB", 256, 355), ("vec", 512, 1464),
                           ("RB", 1536, 6144), ("W", 6144, 28176)]),
          (("B",), 25600, [("sA", 0, 102), ("fA", 256, 307), ("vec", 512, 1464),
                           ("RA", 1536, 3168), ("W", 3328, 25360)]),
          (("A", "B"), 23296, [("vec", 0, 952), ("W", 1024, 23056)])]
    for packed, want, regs_want in pw:
        need(workspace_size(17, 33, 5, 3, 9, 2, packed=packed) == want, f"packed ws {packed}")
        need(region_offsets(17, 33, 5, 3, 9, 2, packed=packed) == regs_want,
             f"packed offsets {packed}: {region_offsets(17, 33, 5, 3, 9, 2, packed=packed)}")
    for packed in ((), ("A",), ("B",), ("A", "B")):   # DIRECT / k = 0 / K-empty: unchanged
        for args in ((17, 33, 5, 3, 9, 2, DIRECT), (17, 33, 0, 3, 9, 0), (17, 33, 1, 3, 9, 0)):
            need(workspace_size(*args, packed=packed) == workspace_size(*args), "vec-only rule")
    ex = workspace_size(8192, 8192, 8192, 1, 14, 14, packed=("A", "B"))
    return (f"{n_cases} (b, r, k, N) layouts, {len(worked) + len(pw)} hand-computed examples; "
            f"8192^3 N=14 both packed: {ex / 2**30:.3f} GiB workspace")


# DESIGN §4a table typed literally (independent of _REGIONS / workspace_size).
def _design_region_bytes(m, n, k, b, N, g):
    kh, nh = ceil_div(k, 16) * 16, ceil_div(n, 16) * 16
    return {"sA": 2 * b * m, "sB": 2 * b * n, "fA": b * m, "fB": b * n,
            "vec": 8 * (2 * m + 2 * n + 2 * k + 3 * b),
            "RA": g * b * m * kh, "RB": g * b * nh * kh, "W": N * b * m * nh}


def _check_layout(m, n, k, b, N, g, mode, ws, regs, vec=VEC_GENERATE):
    gg = N if g == 0 else g
    if b * m * n == 0:
        need(ws == 0 and regs == [], "zero-size rule")
        return
    lit = _design_region_bytes(m, n, k, b, N, gg)
    res = vec == VEC_RESIDENT     # v1.2: no vec region
    if mode == DIRECT or k <= 1:
        need([r[0] for r in regs] == ([] if res else ["vec"]), "DIRECT/k=0/K-empty rule")
        need(ws == (0 if res else align_up(lit["vec"], 256)), "vec only")
    else:
        order = ["sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"]
        need([r[0] for r in regs] == [x for x in order if not (res and x == "vec")], "region order")
    total, prev_end = 0, 0
    for name, start, end in regs:
        need(end - start == lit[name], f"{name} size {end - start} != DESIGN {lit[name]}")
        need(start % 256 == 0, f"{name} misaligned")
        need(start >= prev_end, f"{name} overlaps")
        need(start == total, "offset = sum of previous aligned sizes")
        total += ceil_div(end - start, 256) * 256
        prev_end = end
    need(total == ws, f"sum {total} != closed form {ws}")
    if mode == EMULATE and k > 1:
        # CUTLASS alignment 16: every operand/aux pointer the K4 launches form
        # (workspace base is 256-aligned, §5) is 16-B aligned: plane (l, lambda)
        # base + row i/j + chunk c*K_c' for A' (RA), B' (RB) and W planes; and (v1.13 §4b, a
        # performance contract) c*K_c' is a multiple of 64.
        off = {r[0]: r[1] for r in regs}
        kh, nh = ceil_div(k, 16) * 16, ceil_div(n, 16) * 16
        rows_a = sorted({0, 1, m - 1})
        rows_b = sorted({0, 1, nh - 1})
        chunks = range(ceil_div(k, K_C_PRIME))
        for lam in sorted({0, b - 1}):
            for l in sorted({0, gg - 1}):
                for c in chunks:
                    # v1.13 §4b: a chunk start leaves the row's 64-B load phase unchanged
                    need((c * K_C_PRIME) % K_CHUNK_ALIGN == 0, "chunk start 64-B phase")
                    for i in rows_a:
                        need((off["RA"] + ((l * b + lam) * m + i) * kh + c * K_C_PRIME) % 16 == 0, "A' ptr")
                    for j in rows_b:
                        need((off["RB"] + ((l * b + lam) * nh + j) * kh + c * K_C_PRIME) % 16 == 0, "B' ptr")
            for l in sorted({0, N - 1}):
                for i in rows_a:
                    need((off["W"] + ((l * b + lam) * m + i) * nh) % 16 == 0, "W ptr")


# ---------------------------------------------------------------------------
# Outputs


def run_checks():
    fails = 0
    for section, fn in CHECKS:
        name = fn.__name__
        try:
            detail = fn()
            print(f"PASS  {section:<45} {name}: {detail}")
        except CertFail as e:
            fails += 1
            print(f"FAIL  {section:<45} {name}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {section:<45} {name}: {type(e).__name__}: {e}")
    print(f"{len(CHECKS) - fails}/{len(CHECKS)} certificates passed")
    return fails == 0


def print_table():
    print(f"{'N':>2} {'log2P':>8} {'log2T':>7} {'L32':>3} {'eps_max':>7} "
          f"{'rn_feasible_kmax':>24} {'2sqrt(k)/T k=1e4':>17} {'k=2^17':>9} "
          f"{'sigma RN k<=2^40':>17} {'sigma TRUNC':>13}")
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        kx = rn_feasible_kmax(N)
        kxs = str(kx) if kx < 10**12 else f"2^{math.log2(kx):.2f}"
        srn = shift_bounds(N, KMAX_CERT, RN)
        str_ = shift_bounds(N, KMAX_CERT, TRUNC)
        print(f"{N:>2} {math.log2(c['P']):>8.3f} {math.log2(c['T']):>7.3f} {c['L32']:>3} "
              f"{c['eps_max']:>7} {kxs:>24} {error_guidance(N, 10**4)['guidance']:>17.2e} "
              f"{error_guidance(N, 2**17)['guidance']:>9.2e} "
              f"{'[%d,%d]' % (srn['sigma_min'], srn['sigma_max']):>17} "
              f"{'[%d,%d]' % (str_['sigma_min'], str_['sigma_max']):>13}")


def build_json():
    S = str
    per_n = {}
    for N in range(N_MIN, N_MAX + 1):
        c = constants(N)
        per_n[S(N)] = {
            "moduli": [S(p) for p in c["moduli"]],
            "P": S(c["P"]), "H": S(c["H"]), "T": S(c["T"]),
            "M": [S(x) for x in c["M"]], "q": [S(x) for x in c["q"]],
            "L32": S(c["L32"]), "Lw": S(c["Lw"]), "Lw_design": S(design_lw(N)),
            "crt_peak": S(c["crt_peak"]), "eps_max": S(c["eps_max"]),
            "rn_feasible_kmax": S(rn_feasible_kmax(N)),
            "rn_budget_sq_kappa1": S((2 * c["T"] - 1)**2),
            "shift_bounds_k_le_2^40": {
                r: {key: S(v) for key, v in shift_bounds(N, KMAX_CERT, r).items()
                    if key in ("sigma_min", "sigma_max", "k_hi")} for r in (RN, TRUNC)},
        }
    emax = constants(N_MAX)["eps_max"]
    return {
        "generator": "spec/params.py",
        "design": "docs/DESIGN.md v1 sections 2-4a",
        "note": "all integers are decimal strings; regenerate with: python3 spec/params.py --json",
        "N_range": [S(N_MIN), S(N_MAX)],
        "moduli": [S(p) for p in REAL_MODULI],
        "K_c": S(K_C), "K_c_prime": S(K_C_PRIME), "F": S(F_BITS),
        "term_max": S(TERM_MAX), "gamma_range": [S(GAMMA_MIN), S(GAMMA_MAX)],
        "region_align": S(REGION_ALIGN), "k_align": S(K_ALIGN), "n_align": S(N_ALIGN),
        "k_chunk_align": S(K_CHUNK_ALIGN),
        "workspace_regions": [r[0] for r in _REGIONS],
        "pow2_mod": {S(p): [S(t) for t in pow2_table(p, emax)] for p in REAL_MODULI},
        "epilogue_note": ("v = W + acc in [v_min, v_max]; p = 256: v & 255; else n = v + bias "
                          "(u32), q = umulhi64(m, n), v mod p = n - p*q (certified by cert_barrett)"),
        "epilogue": {S(p): dict({key: S(v) for key, v in barrett_plan(p).items()},
                                v_min=S(epilogue_range(p)[0]), v_max=S(epilogue_range(p)[1]))
                     for p in REAL_MODULI},
        "per_N": per_n,
    }


def _limbs(x, count, bits=LIMB):
    """x as `count` little-endian `bits`-bit limbs (asserts that x fits)."""
    if not 0 <= x < 2**(bits * count):
        raise ValueError(f"{x} does not fit {count} x {bits}-bit limbs")
    return [(x >> (bits * i)) & (2**bits - 1) for i in range(count)]


def build_header():
    """C++17 header with the device constant tables (DESIGN §2, §3.2-§3.6, §4, §4a).

    Every value is taken from the functions certified by --check; nothing is typed by
    hand. Per-N tables are indexed by N directly (rows 0..N_MIN-1 are zero), multi-word
    integers are little-endian u32 limbs padded to the N_MAX width, and the per-modulus
    tables are indexed by the modulus position l (0-based, so p_1 is index 0)."""
    max_limbs = constants(N_MAX)["L32"]
    emax = constants(N_MAX)["eps_max"]
    sig_lo = min(shift_bounds(N, KMAX_CERT, r)["sigma_min"]
                 for N in range(N_MIN, N_MAX + 1) for r in (RN, TRUNC))
    sig_hi = max(shift_bounds(N, KMAX_CERT, r)["sigma_max"]
                 for N in range(N_MIN, N_MAX + 1) for r in (RN, TRUNC))
    zero_limbs = [0] * max_limbs

    def u32(x):
        return f"{x}u"

    def u64(x):
        return f"{x}ull"

    def row(xs, fmt, per_line=None):
        items = [fmt(x) for x in xs]
        if per_line is None or len(items) <= per_line:
            return "{" + ", ".join(items) + "}"
        lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
        return "{\n      " + ",\n      ".join(lines) + "}"

    per_n = {N: constants(N) for N in range(N_MIN, N_MAX + 1)}

    def per_n_rows(fn, empty):
        return [fn(per_n[N]) if N in per_n else empty for N in range(N_MAX + 1)]

    plans = [barrett_plan(p) for p in REAL_MODULI]
    ranges = [epilogue_range(p) for p in REAL_MODULI]
    out = []
    w = out.append
    w("// GENERATED by spec/params.py --emit-header. DO NOT EDIT.")
    w("// Regenerate: python3 spec/params.py --emit-header include/immerdink/detail/constants.hpp")
    w("// (the CMake build regenerates it and a test checks that this committed copy is current).")
    w("//")
    w("// Device constant tables of docs/DESIGN.md v1.1 (§2, §3.2-§3.6, §4, §4a), every value")
    w("// certified by `python3 spec/params.py --check`. Multi-word integers are little-endian")
    w("// u32 limbs padded to kMaxLimbs; per-N rows are indexed by N (rows below kNMin are 0);")
    w("// per-modulus entries are indexed by the 0-based modulus position l (p_1 is index 0).")
    w("// C++17, usable from host and device code (see src/device/constants.cuh).")
    w("#pragma once")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace imd::constants {")
    w("")
    w(f"inline constexpr int kNMin = {N_MIN};")
    w(f"inline constexpr int kNMax = {N_MAX};")
    w(f"inline constexpr int kLimbBits = {LIMB};")
    w(f"inline constexpr int kMaxLimbs = {max_limbs};          // L32 at N = kNMax")
    w(f"inline constexpr int kF = {F_BITS};                  // §3.2 fraction bits of S")
    w(f"inline constexpr int64_t kKc = {K_C};          // §2 max exact K-chunk (.satfinite)")
    w(f"inline constexpr int64_t kKcPrime = {K_C_PRIME};     // §3.5 chunk length used, floor(K_c/64)*64")
    w(f"inline constexpr int kKChunkAlign = {K_CHUNK_ALIGN};       // v1.13 §4b K_c' multiple (K4 K tile)")
    w(f"inline constexpr int kRegionAlign = {REGION_ALIGN};       // §4a align(x) = ceil(x)_256")
    w(f"inline constexpr int kKAlign = {K_ALIGN};            // §4 k_hat = ceil(k)_16")
    w(f"inline constexpr int kNAlign = {N_ALIGN};            // §4 n_hat = ceil(n)_16")
    w(f"inline constexpr int kGammaMin = {GAMMA_MIN};       // §3.2 ilogb range, finite nonzero FP64")
    w(f"inline constexpr int kGammaMax = {GAMMA_MAX};")
    w(f"inline constexpr uint64_t kTermMax = {u64(TERM_MAX)};  // §3.2 max ceil(x^2 2^F)")
    w(f"inline constexpr int kSigmaMin = {sig_lo};       // §3.2 sigma range, N = 2..20, k <= 2^40, RN+TRUNC")
    w(f"inline constexpr int kSigmaMax = {sig_hi};")
    w(f"inline constexpr int kEpsMax = {emax};             // §3.3 eps_max at N = kNMax (pow2_mod width - 1)")
    w(f"inline constexpr int32_t kModMaskModulus = 256;  // §3.5 p = 256 reduces by v & 255")
    w("")
    w("struct Tables {")
    w("  // §2 moduli p_l, l = 0..kNMax-1 (a prefix of length N is used).")
    w("  uint32_t moduli[kNMax];")
    w("  // §3.5 exact v mod p for v in [epi_vmin, epi_vmax] (cert_barrett):")
    w("  //   p = 256: v & 255;  otherwise n = u32(v + epi_bias), q = umulhi64(epi_magic, n),")
    w("  //   v mod p = n - p*q.  epi_bias = p*ceil(2^31/p), epi_magic = ceil(2^64/p); both 0 for 256.")
    w("  uint32_t epi_bias[kNMax];")
    w("  uint64_t epi_magic[kNMax];")
    w("  int32_t epi_vmin[kNMax];")
    w("  int32_t epi_vmax[kNMax];")
    w("  // §3.4 2^e mod p_l, e = 0..kEpsMax.")
    w("  uint8_t pow2_mod[kNMax][kEpsMax + 1];")
    w("  // Per N (row N): limb count L32 (2P <= 2^(32 L32)), eps_max = floor(log2 T),")
    w("  // P = prod p_l, H = P/2 - 1, T = isqrt(H), M_l = P/p_l, q_l = M_l^-1 mod p_l.")
    w("  int32_t L32[kNMax + 1];")
    w("  int32_t eps_max[kNMax + 1];")
    w("  uint32_t P[kNMax + 1][kMaxLimbs];")
    w("  uint32_t H[kNMax + 1][kMaxLimbs];")
    w("  uint32_t T[kNMax + 1][kMaxLimbs];")
    w("  uint32_t M[kNMax + 1][kNMax][kMaxLimbs];")
    w("  uint32_t q[kNMax + 1][kNMax];")
    w("};")
    w("")
    w("inline constexpr Tables kTables = {")
    w(f"    /* moduli */ {row(REAL_MODULI, u32, 10)},")
    w(f"    /* epi_bias */ {row([pl.get('bias', 0) for pl in plans], u32, 5)},")
    w(f"    /* epi_magic */ {row([pl.get('m', 0) for pl in plans], u64, 4)},")
    w(f"    /* epi_vmin */ {row([lo for lo, _ in ranges], str, 5)},")
    w(f"    /* epi_vmax */ {row([hi for _, hi in ranges], str, 5)},")
    w("    /* pow2_mod */ {")
    for p in REAL_MODULI:
        w(f"      {row(pow2_table(p, emax), str, 26)},")
    w("    },")
    w(f"    /* L32 */ {row(per_n_rows(lambda c: c['L32'], 0), str)},")
    w(f"    /* eps_max */ {row(per_n_rows(lambda c: c['eps_max'], 0), str)},")
    for name in ("P", "H", "T"):
        w(f"    /* {name} */ {{")
        for N in range(N_MAX + 1):
            v = _limbs(per_n[N][name], max_limbs) if N in per_n else zero_limbs
            w(f"      {row(v, u32)},  // N = {N}")
        w("    },")
    w("    /* M */ {")
    for N in range(N_MAX + 1):
        ms = per_n[N]["M"] if N in per_n else []
        rows_ = [_limbs(x, max_limbs) for x in ms] + [zero_limbs] * (N_MAX - len(ms))
        w(f"      {{  // N = {N}")
        for r in rows_:
            w(f"        {row(r, u32)},")
        w("      },")
    w("    },")
    w("    /* q */ {")
    for N in range(N_MAX + 1):
        qs = per_n[N]["q"] if N in per_n else []
        w(f"      {row(qs + [0] * (N_MAX - len(qs)), u32, 20)},  // N = {N}")
    w("    },")
    w("};")
    w("")
    w("}  // namespace imd::constants")
    return "\n".join(out) + "\n"


def write_if_changed(path, text):
    """Write `text` to `path` unless it already has exactly that content."""
    try:
        with open(path) as f:
            if f.read() == text:
                return False
    except FileNotFoundError:
        pass
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit-header", metavar="PATH",
                    help="write the C++ device constant header (DESIGN §2-§4a) to PATH")
    ap.add_argument("--scaled", nargs=3, type=int, metavar=("RES_BITS", "ACC_BITS", "N"))
    args = ap.parse_args(argv)
    if args.emit_header:
        changed = write_if_changed(args.emit_header, build_header())
        print(f"{'wrote' if changed else 'unchanged'} {args.emit_header}")
    if not (args.check or args.table or args.json or args.scaled):
        if args.emit_header:
            return 0
        ap.print_help()
        return 2
    ok = True
    if args.table:
        print_table()
    if args.scaled:
        r, a, n = args.scaled
        print(json.dumps(scaled_constants(r, a, n), indent=1))
    if args.json:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_out.json")
        with open(path, "w") as f:
            json.dump(build_json(), f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")
    if args.check:
        ok = run_checks()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
