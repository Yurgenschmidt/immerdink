"""Exact, vectorized engine of the C2 study (NOTES-C1 §4.3 M-a) and of the M-d1 convergence
runs: numpy only, single-threaded, no GPU and no CUDA harness (bench/accuracy/common.py
imports tests/harness, so it is not used here).

Operands are held as ROW-FIBER matrices: A (m x k) is the GEMM view's A, Bt (n x k) the
transpose of its B (column j of B is row j of Bt), both C-contiguous, so every pass over
either operand runs along K and can be blocked by rows (<= BLOCK rows per block).

Bit-identity with spec/oracle.py (self_test() checks it before any record is written):
  sigma    §3.2 shift of every row: an FP64 pre-pass gives S's N-independent part per row with
           a rigorous enclosure of the exact S = sum ceil(x_h^2 2^F); u is read off the
           enclosure with frexp (floor(log4) of an exact float), and every row whose u the
           enclosure does not decide is recomputed exactly by oracle.scale_vector.
  kweights oracle.kweights' rule (NOTES-C1 §4.5 steps 1-5) on the GEMM view: exponent
           extrema per K index (union extrema over several partners), the lower median of
           Delta, the deadband, the max_abs clamp and the mandatory FP64-exact clamp, then
           the guard from the sigma engine on the weighted rows (exact: the clamp makes
           ldexp exact, checked by a round trip).
  emulate  D = RN(z 2^-(sigma + tau)) with z = sum a'b' exact: a' = RN / trunc(a 2^sigma)
           are integer-valued floats (exact), split into balanced 20-bit limbs and multiplied
           with FP64 BLAS over K chunks of 4096 (every partial sum is an integer < 2^53, so
           exact in any order); the limb diagonals are carried into base-2^20 digits in int64
           and rounded once, half-even, from a 64-bit window plus a sticky bit (round_fast;
           checked against the Python-integer path, whose int -> float conversion is
           correctly rounded; results below the normal range fall back to
           oracle.round_dyadic). sum |a'||b'| <= H (the CRT precondition, under which the CRT
           returns z exactly, OzakiCRT CRTExact) is asserted on every output.
  exact    x = sum a b exactly: each row split error-free into 20-bit digits at its own
           exponent, the same exact BLAS, Python-integer combination; returned as
           hi = RN(x) and lo = RN(x - hi).
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "spec"))
import oracle as o  # noqa: E402

F = o.F
U = 2.0 ** -53
LIMB = 20                 # limb / digit bits: |limb| <= 2^20, 4096 products < 2^52
KCHUNK = 4096             # K chunk of the exact BLAS (every partial sum < 2^53)
BLOCK = 256               # rows per block of every full-operand pass (NOTES-C1 §4.3 step 3)
NEG, POS = -(1 << 20), 1 << 20   # "undefined" sentinels of the exponent extrema


# ---------------------------------------------------------------------------------------
# §3.2 shifts of every row
# ---------------------------------------------------------------------------------------

class RowStats:
    """N-independent part of §3.2 for every row of X (r x k): S_fp = sum (|x_h| 2^-gamma)^2
    in FP64 (any summation order), gamma = ilogb(max |x_h|) over the finite entries, the
    count of nonzero finite entries, the flag (a non-finite entry, §3.1) and zero rows."""

    def __init__(self, X):
        r, k = X.shape
        self.k = k
        self.S = np.zeros(r)
        self.gamma = np.zeros(r, dtype=np.int64)
        self.nnz = np.zeros(r, dtype=np.int64)
        self.flag = np.zeros(r, dtype=bool)
        for i0 in range(0, r, BLOCK):
            Xb = X[i0:i0 + BLOCK]
            fin = np.isfinite(Xb)
            Ab = np.where(fin, np.abs(Xb), 0.0)
            mx = Ab.max(axis=1)
            g = np.frexp(np.where(mx == 0, 1.0, mx))[1].astype(np.int64) - 1
            Y = np.ldexp(Ab, -g[:, None])
            self.S[i0:i0 + BLOCK] = (Y * Y).sum(axis=1)
            self.gamma[i0:i0 + BLOCK] = g
            self.nnz[i0:i0 + BLOCK] = (Ab != 0).sum(axis=1)
            self.flag[i0:i0 + BLOCK] = ~fin.all(axis=1)
        self.zero = self.nnz == 0


def _floor_log4(y):
    """floor(log4 y) for positive finite floats y, exactly (frexp gives floor(log2 y))."""
    return np.floor_divide(np.frexp(y)[1].astype(np.int64) - 1, 2)


def shifts(X, st: RowStats, plan: o.Plan, out_fallback=None):
    """sigma (int64) of every row of X at the plan (its k must equal the row length):
    bit-identical to oracle.scale_vector(row, plan)[0]. out_fallback, if a list, receives
    the indices of the rows recomputed exactly.

    Enclosure: S_real = sum x_h^2 2^F (x = |a| 2^-gamma) and S = sum ceil(x_h^2 2^F) lie in
    [S_real, S_real + nnz]; |S_fp 2^F - S_real| <= gamma_k S_fp 2^F + k 2^-1014 for any
    summation order (squares and the 2^-gamma scaling underflow only below 2^-1022).
    With S_fp 2^F >= 2^F, every slack below is relative: eps = (k + 4) 2^-52 + 2^-48."""
    assert X.shape[1] == st.k == plan.k
    k = st.k
    eps = (k + 4) * 2.0 ** -52 + 2.0 ** -48
    S2 = np.where(st.zero, 1.0, st.S) * 2.0 ** F
    lo = S2 * (1 - eps)
    hi = S2 * (1 + eps + (st.nnz + 2) * 2.0 ** -F)
    L, R = plan.shift_lhs_mul, float(plan.shift_rhs)
    r_lo = R * (1 - 2.0 ** -50) / (L * hi) * (1 - 2.0 ** -50)
    r_hi = R * (1 + 2.0 ** -50) / (L * lo) * (1 + 2.0 ** -50)
    u_lo, u_hi = _floor_log4(r_lo), _floor_log4(r_hi)
    u = u_hi.copy()
    amb = np.nonzero((u_lo != u_hi) & ~st.zero)[0]
    for i in amb:
        u[i] = o.scale_vector(X[i].tolist(), plan)[4]
    if out_fallback is not None:
        out_fallback.extend(int(i) for i in amb)
    sig = u - st.gamma
    sig[st.zero] = 0
    return sig


def u_of(sig, st: RowStats):
    """u = sigma + gamma (the §3.2 integer before the gamma offset); 0 for zero rows."""
    return np.where(st.zero, 0, sig + st.gamma)


def norms(X):
    """Row 2-norms of X (FP64, scaled against overflow/underflow; finite entries)."""
    out = np.zeros(X.shape[0])
    for i0 in range(0, X.shape[0], BLOCK):
        Xb = np.where(np.isfinite(X[i0:i0 + BLOCK]), X[i0:i0 + BLOCK], 0.0)
        mx = np.abs(Xb).max(axis=1)
        mx = np.where(mx == 0, 1.0, mx)
        Y = Xb / mx[:, None]
        out[i0:i0 + BLOCK] = mx * np.sqrt((Y * Y).sum(axis=1))
    return out


# ---------------------------------------------------------------------------------------
# K-weights (oracle.kweights on the GEMM view)
# ---------------------------------------------------------------------------------------

def extrema(Xs):
    """(c, m, defined) per K index over every row of every matrix in Xs (each r x k): max /
    min ilogb over the finite nonzero entries (union extrema across the matrices)."""
    k = Xs[0].shape[1]
    c = np.full(k, NEG, dtype=np.int64)
    mn = np.full(k, POS, dtype=np.int64)
    for X in Xs:
        for i0 in range(0, X.shape[0], BLOCK):
            Xb = X[i0:i0 + BLOCK]
            ok = np.isfinite(Xb) & (Xb != 0)
            E = np.frexp(np.where(ok, Xb, 1.0))[1].astype(np.int64) - 1
            c = np.maximum(c, np.where(ok, E, NEG).max(axis=0))
            mn = np.minimum(mn, np.where(ok, E, POS).min(axis=0))
    return c, mn, c > NEG


def weights(ext_x, ext_y, deadband, max_abs=o.KW_MAX_ABS):
    """NOTES-C1 §4.5 steps 2-4 (oracle.kweights) from the extrema of X (ext_x) and of every
    partner (ext_y, union): returns (w, info) with w int64 per K index."""
    cX, mX, dX = ext_x
    cY, mY, dY = ext_y
    both = dX & dY
    delta = cY - cX
    dd = np.sort(delta[both])
    med = int(dd[(dd.size - 1) // 2]) if dd.size else None
    if med is None:
        w0 = np.zeros(cX.size, dtype=np.int64)
    else:
        w0 = np.where(~both | (np.abs(delta - med) <= deadband), 0,
                      np.floor_divide(delta - med, 2))
    wa = np.clip(w0, -max_abs, max_abs)
    hi = np.full(cX.size, max_abs, dtype=np.int64)
    hi = np.where(dX, np.minimum(hi, 1023 - cX), hi)
    hi = np.maximum(np.where(dY, np.minimum(hi, mY + 1022), hi), 0)
    lo = np.full(cX.size, -max_abs, dtype=np.int64)
    lo = np.where(dX, np.maximum(lo, -1022 - mX), lo)
    lo = np.minimum(np.where(dY, np.maximum(lo, cY - 1023), lo), 0)
    w = np.maximum(lo, np.minimum(hi, wa))
    info = dict(median=med, w0=w0, clamped=int(np.sum(w != w0)),
                clamped_max_abs=int(np.sum(wa != w0)), clamped_fp64=int(np.sum(w != wa)),
                num_weighted=int(np.sum(w != 0)), max_abs_w=int(np.abs(w).max()) if w.size else 0,
                defined=int(both.sum()))
    return w, info


def scale(X, w, sign=1):
    """X 2^(sign w) along K (a copy), checked exact by a round trip on the finite entries
    (the FP64 clamp's promise). Returns (Xw, violations)."""
    Xw = np.empty_like(X)
    bad = 0
    sw = (sign * w).astype(np.int64)
    for i0 in range(0, X.shape[0], BLOCK):
        Xb = X[i0:i0 + BLOCK]
        Y = np.ldexp(Xb, sw[None, :])
        back = np.ldexp(Y, -sw[None, :])
        fin = np.isfinite(Xb)
        bad += int(np.sum(fin & ((back != Xb) | ~np.isfinite(Y))))
        Xw[i0:i0 + BLOCK] = Y
    return Xw, bad


def gain_min(sig0, sig1, st: RowStats):
    """min over nonzero finite rows of sigma(weighted) - sigma(plain); None if none."""
    ok = ~st.zero & ~st.flag
    return int((sig1 - sig0)[ok].min()) if ok.any() else None


# ---------------------------------------------------------------------------------------
# Exact integer products (limb BLAS)
# ---------------------------------------------------------------------------------------

def _limbs(Xi, nl):
    """Balanced LIMB-bit limbs of an integer-valued float array (|Xi| < 2^(LIMB nl - 1)),
    most significant first: Xi = sum_s L_s 2^(LIMB (nl - 1 - s))."""
    R = Xi.copy()
    out = []
    for s in range(nl - 1):
        sh = LIMB * (nl - 1 - s)
        d = np.rint(R * 2.0 ** -sh)                    # exact scalings: integers, |R| < 2^79
        R = R - d * 2.0 ** sh
        out.append(d)
    assert np.all(np.abs(R) <= 1 << LIMB), "limb split not exact"
    out.append(R)                                      # the last limb: R itself (an integer)
    return out


def nlimbs(maxabs):
    b = int(maxabs).bit_length() if maxabs >= 1 else 1
    return max(1, -(-(b + 1) // LIMB))


def _stacked_products(LA, LB):
    """{s + t: sum P_st} with P_st = LA[s] @ LB[t] (LA: r x k limbs, LB: k x c limbs), exact:
    one stacked GEMM per K chunk, every chunk's sums < 2^53, accumulated in int64."""
    r, k = LA[0].shape
    c = LB[0].shape[1]
    SA, SB = np.vstack(LA), np.hstack(LB)
    acc = np.zeros((SA.shape[0], SB.shape[1]), dtype=np.int64)
    for h0 in range(0, k, KCHUNK):
        P = SA[:, h0:h0 + KCHUNK] @ SB[h0:h0 + KCHUNK, :]
        assert np.all(np.abs(P) < 2.0 ** 53)
        acc += P.astype(np.int64)
    Q = {}
    for s in range(len(LA)):
        for t in range(len(LB)):
            Q[s + t] = Q.get(s + t, 0) + acc[s * r:(s + 1) * r, t * c:(t + 1) * c]
    return Q


def _combine(Q, dmax):
    """Python-int object array sum_d Q_d 2^(LIMB (dmax - d))."""
    X = None
    for d, q in Q.items():
        t = q.astype(object) * (1 << (LIMB * (dmax - d)))
        X = t if X is None else X + t
    return X


def int_matmul(Ai, Bi, raw=False):
    """Exact product (object array of Python ints) of integer-valued float matrices; raw:
    the limb diagonals (Q, dmax) instead (round_fast's input)."""
    na = nlimbs(np.abs(Ai).max()) if Ai.size else 1
    nb = nlimbs(np.abs(Bi).max()) if Bi.size else 1
    Q = _stacked_products(_limbs(Ai, na), _limbs(Bi, nb))
    return (Q, na + nb - 2) if raw else _combine(Q, na + nb - 2)


def _carry(Dg):
    """Carry-propagate base-2^LIMB digits (int64, most significant first) in place: every
    digit but the first ends in [0, 2^LIMB); the value sum_i Dg[i] 2^(LIMB (n - 1 - i)) is
    unchanged."""
    for i in range(Dg.shape[0] - 1, 0, -1):
        c = Dg[i] >> LIMB                              # floor division by 2^LIMB
        Dg[i] -= c << LIMB
        Dg[i - 1] += c


def round_fast(Q, dmax, ex):
    """RN(z 2^ex) for z = sum_d Q[d] 2^(LIMB (dmax - d)) (int64 diagonals, |Q_d| < 2^61),
    vectorized and exact: normalized base-2^LIMB digits of |z| (int64 carries), the 64-bit
    window under the leading bit plus a sticky bit, round-half-even to 53 bits, one exact
    ldexp. +0.0 for z = 0 (the oracle's choice). Results below the normal range fall back
    to oracle.round_dyadic (one rounding). Checked against round_scaled by self_test()."""
    shape = np.shape(ex)
    nd = dmax + 4                                      # 3 extra leading digits absorb carries
    Dg = np.zeros((nd,) + shape, dtype=np.int64)
    for d, q in Q.items():
        assert np.all(np.abs(q) < 2 ** 61)
        Dg[3 + d] += q
    _carry(Dg)
    neg = Dg[0] < 0                                    # the rest is >= 0: the sign of z
    Dg[:, neg] = -Dg[:, neg]
    _carry(Dg)                                         # digits of |z|, all in [0, 2^LIMB)
    assert np.all(Dg >= 0) and np.all(Dg < 1 << LIMB)
    nz = Dg != 0
    zero = ~nz.any(axis=0)
    p = np.argmax(nz, axis=0)                          # leading nonzero digit
    pad = np.concatenate([Dg, np.zeros((4,) + shape, dtype=np.int64)])
    suf = np.flip(np.cumsum(np.flip(pad != 0, axis=0), axis=0), axis=0) > 0  # any at >= i
    take = lambda A, j: np.take_along_axis(A, (p + j)[None], axis=0)[0]  # noqa: E731
    U64 = np.uint64
    L0, L1, L2, L3 = (take(pad, j).astype(U64) for j in range(4))
    sticky = take(suf, 4)
    bl = np.frexp(np.where(zero, 1.0, L0.astype(np.float64)))[1].astype(np.int64)  # 1..LIMB
    W = (L0 << (64 - bl).astype(U64)) | (L1 << (44 - bl).astype(U64)) | \
        (L2 << (24 - bl).astype(U64))
    sh = bl - 4
    W |= np.where(sh <= 0, L3 << np.maximum(-sh, 0).astype(U64),
                  L3 >> np.maximum(sh, 0).astype(U64))
    lost = L3 & ((U64(1) << np.maximum(sh, 0).astype(U64)) - U64(1))
    sticky = sticky | ((sh > 0) & (lost != 0))
    M = W >> U64(11)
    rem = W & U64(0x7FF)
    up = (rem > U64(0x400)) | ((rem == U64(0x400)) & (sticky | ((M & U64(1)) == U64(1))))
    M = M + up.astype(U64)
    g = bl - 1 + LIMB * (nd - 1 - p)                   # bit index of z's leading bit
    e = g - 52 + ex
    with np.errstate(over="ignore"):
        D = np.ldexp(M.astype(np.float64), np.where(zero, 0, e))
    D = np.where(neg, -D, D)
    D[zero] = 0.0
    tiny = ~zero & (g + ex < -1022)
    if tiny.any():
        Z = _combine(Q, dmax)
        for idx in zip(*np.nonzero(tiny)):
            D[idx] = o.round_dyadic(int(Z[idx]), int(ex[idx]))
    return D


def quantize(X, sig, rounding):
    """a' = RN_even / trunc(a 2^sigma) per row (integer-valued floats, exact; |a'| <= 2^78).
    The scaling multiplies by 2^sigma (the same IEEE result as ldexp, faster) whenever 2^sigma
    is a normal double."""
    if sig.size and -1022 <= sig.min() and sig.max() <= 1023:
        Y = X * np.ldexp(1.0, sig)[:, None]
    else:
        Y = np.ldexp(X, sig[:, None])
    return np.rint(Y) if rounding == "RN" else np.trunc(Y)


def emulate(Ar, Bc, sig, tau, plan: o.Plan, rounding, check=True):
    """D (r x c) of the sampled rows Ar (r x k) and columns Bc (c x k, row fibers) at the
    given shifts: D = RN(z 2^-(sigma+tau)), z = sum a'b' exact. Asserts the CRT
    precondition sum|a'||b'| <= H on every output (check=True) and returns
    (D, max sum|a'||b'| / H)."""
    Aq = quantize(Ar, sig, rounding)
    Bq = quantize(Bc, tau, rounding).T
    ratio = None
    if check:
        Sabs = np.abs(Aq) @ np.abs(Bq)                 # FP64: certify most outputs at once
        k = Ar.shape[1]
        up = Sabs * (1 + (k + 4) * 2.0 ** -52)
        if np.any(up > float(plan.H) * (1 - 2.0 ** -52)):   # exact where FP64 cannot decide
            Z = int_matmul(np.abs(Aq), np.abs(Bq))
            mx = max(Z.ravel().tolist())
            assert mx <= plan.H, f"CRT precondition violated: {mx} > H"
            ratio = mx / plan.H
        else:
            ratio = float(up.max()) / plan.H
    Q, dmax = int_matmul(Aq, Bq, raw=True)
    return round_fast(Q, dmax, -(sig[:, None] + tau[None, :])), ratio


def round_scaled(Z, ex):
    """RN(Z 2^ex) elementwise for an object array of Python ints: int -> float is correctly
    rounded and ldexp is exact in the normal range; subnormal results fall back to
    oracle.round_dyadic (one rounding, as §3.7)."""
    Dz = Z.astype(np.float64)
    D = np.ldexp(Dz, ex)
    tiny = ((D != 0) & (np.abs(D) < 2.0 ** -1022)) | ((D == 0) & (Dz != 0))
    for i, j in zip(*np.nonzero(tiny)):
        D[i, j] = o.round_dyadic(int(Z[i, j]), int(ex[i, j]))
    return D


def _digits(X):
    """Error-free split of each row of X (finite): X[i] = 2^e_i sum_s D_s 2^(-LIMB s)."""
    mx = np.abs(X).max(axis=1)
    e = np.frexp(np.where(mx == 0, 1.0, mx))[1].astype(np.int64)
    R = np.ldexp(X, -e[:, None])
    assert np.array_equal(np.ldexp(R, e[:, None]), X), "row dynamic range beyond FP64"
    out = []
    s = 0
    while np.any(R != 0):
        s += 1
        assert s <= 60
        d = np.rint(np.ldexp(R, LIMB * s))
        R = R - np.ldexp(d, -LIMB * s)
        out.append(d)
    return e, out or [np.zeros_like(X)]


def exact_ref(Ar, Bc):
    """x = Ar Bc^T exactly: (hi, lo, X, xe) with x_ij = X_ij 2^xe_ij (X Python ints),
    hi = RN(x), lo = RN(x - hi)."""
    ea, DA = _digits(Ar)
    eb, DB = _digits(Bc)
    Q = _stacked_products(DA, [d.T for d in DB])
    dmax = len(DA) + len(DB) - 2
    X = _combine(Q, dmax)
    xe = ea[:, None] + eb[None, :] - LIMB * (dmax + 2)
    hi = round_scaled(X, xe)
    back = np.vectorize(lambda v, s: o.dyadic(float(v)), otypes=[object])(hi, xe)
    rem = X - np.vectorize(lambda md, s: (md[0] << (md[1] - s)) if md[1] >= s else
                           md[0] // (1 << (s - md[1])), otypes=[object])(back, xe)
    lo = round_scaled(rem, xe)
    return hi, lo, X, xe


def errors(D, hi, lo, na, nb):
    """(absolute error, normwise, elementwise) of D against x = hi + lo; elementwise is NaN
    where x = 0. FP64 evaluation of |(D - hi) - lo|: a few ulps of the error itself."""
    E = np.abs((D - hi) - lo)
    nw = E / (na[:, None] * nb[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        ew = np.where(hi != 0, E / np.abs(hi), np.nan)
    return E, nw, ew


# ---------------------------------------------------------------------------------------
# Envelope rule (NOTES-C1 Appendix B), FP64-evaluated with outward rounding
# ---------------------------------------------------------------------------------------

def envelope_d(A, sigF, uF, st_a, EB, rows, plan: o.Plan, s):
    """Appendix B B1-B3 for the given rows of A: d_i = min(floor(max(lambda(R_i), 0) / 2),
    76 - uF_i), R_i = sum_h U_ih E^B_h. U = ceil(|a| 2^(sigmaF + min(s, 0))) 2^max(s, 0) as
    in A1; the envelopes EB hold the ceilings without the 2^max(s, 0) factor. R_i is summed
    in FP64 and rounded up by (1 + (k + 4) 2^-52), and lambda is taken from an outward-rounded
    quotient, so d is never above the exact rule's (it may be one below it at a boundary:
    still sound; every sampled output's sum|a'||b'| <= H is asserted exactly anyway)."""
    smin = min(s, 0)
    k = A.shape[1]
    ca = np.ceil(np.ldexp(np.abs(A[rows]), (sigF[rows] + smin)[:, None]))
    Rp = (ca @ EB) * (1 + (k + 4) * 2.0 ** -52)
    H = float(plan.H) * (1 - 2.0 ** -50)
    with np.errstate(divide="ignore"):
        y = np.where(Rp > 0, H / np.where(Rp > 0, Rp, 1.0) * (1 - 2.0 ** -50), 1.0)
    lam = np.frexp(y)[1].astype(np.int64) - 1 + 2 * smin     # floor(log2(H / R')) + 2 min(s,0)
    lam = np.where(Rp > 0, lam, 10 ** 6)
    d = np.minimum(np.floor_divide(np.maximum(lam, 0), 2), 76 - uF[rows])
    d = np.where(st_a.zero[rows], 0, d)
    return np.maximum(d, 0)


def envelope_vec(A, sigF, plan: o.Plan, s):
    """E_h = max_i ceil(|a_ih| 2^(sigmaF_i + min(s, 0))) over every row (exact floats)."""
    smin = min(s, 0)
    E = np.zeros(A.shape[1])
    for i0 in range(0, A.shape[0], BLOCK):
        Ab = np.abs(np.where(np.isfinite(A[i0:i0 + BLOCK]), A[i0:i0 + BLOCK], 0.0))
        E = np.maximum(E, np.ceil(np.ldexp(Ab, (sigF[i0:i0 + BLOCK] + smin)[:, None])).max(axis=0))
    return E


def envelope_s(plan: o.Plan):
    """Appendix B constant s(N) = 62 - floor(log2 T)."""
    return 62 - (plan.T.bit_length() - 1)


# ---------------------------------------------------------------------------------------
# Self-test against the oracle (run before any record is written)
# ---------------------------------------------------------------------------------------

def _rows_near_boundary(plan, k, rng, count):
    """Rows whose exact S sits at S*-1, S*, S*+1 for S* = floor(R / (L 4^u0)), the largest S
    with shift u0: built from entries 2^-e (each contributes exactly 4^(30 - e) to S for
    e <= 30) on top of c ones (gamma = 0), padded with zeros to length k."""
    L, R = plan.shift_lhs_mul, plan.shift_rhs
    out = []
    for _ in range(count):
        c0 = int(rng.integers(1, 4))
        u0 = o.largest_u(L * (c0 << F), R)
        Sstar = R // (L << (2 * u0)) if u0 >= 0 else (R << (-2 * u0)) // L
        for dlt in (-1, 0, 1):
            S = Sstar + dlt
            c = S >> F
            rem = S - (c << F)
            row = [1.0] * c
            p = 0
            while rem:
                dig = rem & 3
                row += [2.0 ** -(30 - p)] * dig      # (2^-(30-p))^2 2^60 = 4^p
                rem >>= 2
                p += 1
            if len(row) > k:
                continue
            sg = rng.choice([-1.0, 1.0], len(row))
            v = np.zeros(k)
            pos = rng.permutation(k)[:len(row)]
            v[pos] = np.array(row) * sg
            out.append(v)
    return out


def self_test(rng_seed=20261007, nrows=2100, verbose=True):
    """>= 2000 rows (random, near the u boundary, subnormal, single-entry, zero, wide range)
    at N in {3, 6, 8, 14, 20}, RN and TRUNC, against oracle.scale_vector: 0 mismatches;
    the K-weight engine against oracle.kweights_gemm; emulate() and exact_ref() against
    oracle.contract / exact_dot. Raises AssertionError on any mismatch; returns counts."""
    rng = np.random.default_rng(rng_seed)
    cnt = dict(rows=0, distinct=0, fallback=0, near=0, kw=0, emu=0, ref=0)
    per_k = {37: nrows // 2, 600: nrows * 3 // 10, 3600: nrows * 3 // 20, 16384: nrows // 50}
    for k, nk in per_k.items():
        rows = []
        for t in range(nk):
            kind = t % 8
            if kind == 0:
                v = (rng.random(k) - 0.5) * np.exp(4 * rng.standard_normal(k))
            elif kind == 1:
                v = np.ldexp(rng.integers(1, 2**20, k).astype(float), -1074) * \
                    rng.choice([-1.0, 1.0], k)                          # subnormal
            elif kind == 2:
                v = np.zeros(k)
                v[rng.integers(k)] = rng.choice([-1, 1]) * np.ldexp(1 + rng.random(),
                                                                    int(rng.integers(-1074, 1024)))
            elif kind == 3:
                v = np.zeros(k)                                         # zero row
            elif kind == 4:
                v = rng.standard_normal(k) * np.ldexp(1.0, rng.integers(-1000, 1000, k))
            elif kind == 5:
                v = np.full(k, np.ldexp(1 + rng.random(), int(rng.integers(-1074, 1000))))
            elif kind == 6:
                v = rng.uniform(-1, 1, k)
                v[rng.integers(k)] = np.inf
            else:
                v = rng.uniform(-1, 1, k) * 10.0 ** rng.uniform(-300, 300)
            rows.append(v)
        for N in (3, 6, 8, 14, 20):
            for rnd in ("RN", "TRUNC"):
                try:
                    plan = o.Plan(N, k, rnd)
                except o.NotSupported:
                    continue
                near = _rows_near_boundary(plan, k, rng, 4)
                X = np.array(rows + near)
                st = RowStats(X)
                fb = []
                sig = shifts(X, st, plan, fb)
                for i in range(X.shape[0]):
                    ref = o.scale_vector(X[i].tolist(), plan)[0]
                    assert sig[i] == ref, (k, N, rnd, i, sig[i], ref)
                cnt["rows"] += X.shape[0]
                cnt["near"] += len(near)
                cnt["fallback"] += len(fb)
        cnt["distinct"] += len(rows)
    assert cnt["distinct"] >= 2000 and cnt["fallback"] > 0, cnt

    # K-weights: the engine's w, median, clamps and guard against oracle.kweights_gemm
    for t in range(60):
        m, n, k = int(rng.integers(1, 5)), int(rng.integers(1, 5)), int(rng.integers(2, 40))
        cen = rng.integers(-60, 60, k)
        A = rng.standard_normal((m, k)) * np.ldexp(1.0, cen + rng.integers(-5, 6, (m, k)))
        B = rng.standard_normal((k, n)) * np.ldexp(1.0, (cen[:, None] * (-1) ** t) +
                                                   rng.integers(-5, 6, (k, n)))
        if t % 5 == 0:
            A[rng.integers(m), rng.integers(k)] = 0.0
            B[rng.integers(k), rng.integers(n)] = np.ldexp(1.0, int(rng.integers(-1074, -1023)))
        if t % 7 == 0:
            A[0, 0] = np.ldexp(1.5, 1020)
        db, N, rnd = int(rng.choice([0, 2, 3, 5])), int(rng.choice([6, 8, 14])), \
            str(rng.choice(["RN", "TRUNC"]))
        wo, rep = o.kweights_gemm(A.tolist(), B.tolist(), N, rnd, deadband=db)
        Bt = np.ascontiguousarray(B.T)
        w, info = weights(extrema([A]), extrema([Bt]), db)
        assert w.tolist() == wo and info["clamped"] == rep["clamped"], (t, w, wo)
        assert (rep["median"] == o.INT32_MIN) == (info["median"] is None)
        if info["median"] is not None:
            assert info["median"] == rep["median"]
        plan = o.Plan(N, k, rnd)
        Aw, ba = scale(A, w, 1)
        Bw, bb = scale(Bt, w, -1)
        assert ba == bb == 0
        sa, sb = RowStats(A), RowStats(Bt)
        ga = gain_min(shifts(A, sa, plan), shifts(Aw, RowStats(Aw), plan), sa)
        gb = gain_min(shifts(Bt, sb, plan), shifts(Bw, RowStats(Bw), plan), sb)
        G = None if ga is None or gb is None else ga + gb
        assert (o.INT32_MIN if G is None else G) == rep["gain"], (t, G, rep["gain"])
        cnt["kw"] += 1

    # round_fast against the Python-integer path on random diagonals: ties to even (with and
    # without sticky bits), carries across digits, both signs, wide exponents, subnormals
    nrf = 0
    for t in range(400):
        shape = (8, 9)
        nq = int(rng.integers(1, 7))
        Q = {}
        for d in range(nq):
            mag = int(rng.integers(0, 58))
            Q[d] = (rng.integers(-2 ** 62, 2 ** 62, shape) >> (61 - mag)).astype(np.int64)
            if t % 4 == 0 and d == nq - 1:   # exact ties at the 53-bit rounding position
                Q = {dd: np.zeros(shape, dtype=np.int64) for dd in range(nq)}
                top = rng.integers(2 ** 52, 2 ** 53, shape)
                Q[0] = top.astype(np.int64)
                if nq > 1:
                    Q[1] = np.full(shape, 1 << (LIMB - 1), dtype=np.int64) * \
                        rng.choice([-1, 1, 0], shape)
        ex = rng.integers(-1180, 120, shape) if t % 5 == 0 else rng.integers(-200, 60, shape)
        Z = _combine(Q, nq - 1)
        want = round_scaled(Z, ex)
        got = round_fast(Q, nq - 1, ex)
        assert np.array_equal(want.view(np.int64), got.view(np.int64)), t
        nrf += want.size
    cnt["round_fast"] = nrf

    # emulate() / exact_ref() against oracle.contract / exact_dot
    for t in range(24):
        m, n, k = 3, 4, int(rng.choice([5, 40, 300]))
        A = (rng.random((m, k)) - 0.5) * np.exp(3 * rng.standard_normal((m, k)))
        B = (rng.random((k, n)) - 0.5) * np.exp(3 * rng.standard_normal((k, n)))
        if t % 3 == 1:
            A = A * 1e-310                                  # subnormal inputs and outputs
        N, rnd = int(rng.choice([4, 6, 8, 14, 17, 20])), str(rng.choice(["RN", "TRUNC"]))
        plan = o.Plan(N, k, rnd)
        Bt = np.ascontiguousarray(B.T)
        sig = shifts(A, RowStats(A), plan)
        tau = shifts(Bt, RowStats(Bt), plan)
        D, _ = emulate(A, Bt, sig, tau, plan, rnd)
        tr = {}
        Do = o.contract(A.tolist(), B.tolist(), [[0.0] * n for _ in range(m)], 1.0, 0.0, N,
                        rnd, trace=tr)
        assert list(sig) == tr["sigma"] and list(tau) == tr["tau"]
        assert np.array_equal(D.view(np.int64), np.array(Do).view(np.int64)), (t, N, rnd)
        hi, lo, X, xe = exact_ref(A, Bt)
        for i in range(m):
            for j in range(n):
                mm, ee = o.exact_dot(A[i].tolist(), B[:, j].tolist())
                assert hi[i, j] == o.round_dyadic(mm, ee)
                assert o.dyadic_to_fraction(mm, ee) == o.dyadic_to_fraction(int(X[i, j]),
                                                                            int(xe[i, j]))
        cnt["emu"] += D.size
        cnt["ref"] += D.size
    if verbose:
        print(f"kw_engine self-test: {cnt['distinct']} random rows (k = 37 .. 16384) at 10 "
              f"plans (N 3/6/8/14/20 x RN/TRUNC) + {cnt['near']} rows on the u boundary: "
              f"{cnt['rows']} row-plan checks, {cnt['fallback']} exact fallbacks, 0 sigma "
              f"mismatches; {cnt['round_fast']} vectorized roundings equal to the integer path; "
              f"{cnt['kw']} K-weight "
              f"cases match oracle.kweights; {cnt['emu']} emulated outputs bit-equal to "
              f"oracle.contract; {cnt['ref']} exact references", flush=True)
    return cnt


if __name__ == "__main__":
    self_test()
