"""Fixtures of the C2 study (NOTES-C1 §4.3 M-a): GEMM-view operands as row-fiber matrices
A (m x k) and Bt (n x k) (column j of B is row j of Bt; K is the GEMM view's canonical order,
so K-segmentation's bands are contiguous in it), generated block by block (<= 256 rows per
block, kw_engine.BLOCK) with a seeded generator per block, so a run's peak memory stays near
the operands themselves.

Families (§2.1, §4.3):
  uniform, phi0.5 / phi2 / phi4, decay_same, decay_opp, outliers, cancel: the M5 generator of
      bench/accuracy/contractions.gen and (U - 1/2) e^(phi N(0,1));
  opp_nonsep: anti-correlated, every row and column decaying at its own rate;
  same_sign: U(0, 1) on both sides;
  local_df: DF factors of a 1D chain (locality along the auxiliary index);
  mixed: decay_opp with half the rows dense;
  cc_df, cc_ppl, cc_ring: the synthetic CC family below;
  multi3: X with three partners (decay_opp-like, phi = 4, uniform), IR6;
  batch_mixed: b batches, the first b/4 decaying along K at another rate, IR16.

Synthetic CC family [I] (modelling assumptions, NOTES-C1 §4.3):
  occupied eps: o // 7 core orbitals at -11 +- 0.5 Ha, the rest valence uniform in
      [-1.5, -0.3] Ha; virtual eps: log-spaced over [0.05, 50] Ha;
  DF factors B^Q_pq = s_Q g(eps_p) g(eps_q) N(0, 1), s_Q decaying over 8 decades along Q,
      g(eps) = (1 + |eps|)^-1/2;
  MP2 amplitudes t_ij^ab = (ia|jb) / (eps_i + eps_j - eps_a - eps_b), (ia|jb) = sum_Q B B;
  cc_df   acQ,bdQ->acbd  (K = Q):  A rows (a,c) = B^Q_ac, B columns (b,d) = B^Q_bd (the same
          factor tensor on both sides, as R2);
  cc_ppl  ijcd,abcd->ijab (K = cd): A rows (i,j) = t_ij^cd, B columns (a,b) = (ac|bd) built
          from DF factors over 3v auxiliary functions;
  cc_ring ilad,ldkc->aikc (K = ld): A rows (a,i) = t_il^ad, B columns (k,c) = (ld|kc).
  The CC shapes are their own (m = n is not imposed): the size class of a tier is matched
  by (o, v, naux) as CC_SIZES lists.
"""

import math

import numpy as np

import kw_engine as ke

FAMILIES = ("uniform", "phi0.5", "phi2", "phi4", "decay_same", "decay_opp", "opp_nonsep",
            "outliers", "cancel", "same_sign", "local_df", "mixed", "cc_df", "cc_ppl", "cc_ring")

# tier size class -> (o, v, naux) of each CC contraction (cc_df: m = n = v^2, k = naux;
# cc_ppl: m = o^2, n = k = v^2; cc_ring: m = n = k = o v)
CC_SIZES = {
    ("S", 128, 600): dict(cc_df=(4, 11, 600), cc_ppl=(11, 24, 72), cc_ring=(4, 32, 96)),
    ("S", 128, 3600): dict(cc_df=(4, 11, 3600), cc_ppl=(11, 60, 180), cc_ring=(8, 75, 225)),
    ("S", 512, 3600): dict(cc_df=(8, 23, 3600), cc_ppl=(23, 60, 180), cc_ring=(16, 64, 192)),
    ("S", 512, 16384): dict(cc_df=(8, 23, 16384)),
    ("L", 2048, 3600): dict(cc_df=(16, 45, 3600), cc_ppl=(32, 56, 168), cc_ring=(32, 64, 192)),
    ("L", 2048, 16384): dict(cc_df=(16, 45, 16384)),
    ("L", 4096, 3600): dict(cc_df=(32, 64, 3600), cc_ppl=(48, 64, 192), cc_ring=(32, 128, 384)),
}


def _blocks(rows, seed, tag, fill):
    """Fill `rows` rows block by block: fill(rng, i0, i1) returns rows i0..i1-1."""
    out = None
    for i0 in range(0, rows, ke.BLOCK):
        i1 = min(rows, i0 + ke.BLOCK)
        rng = np.random.default_rng([seed, tag, i0])
        blk = fill(rng, i0, i1)
        if out is None:
            out = np.empty((rows, blk.shape[1]))
        out[i0:i1] = blk
    return out


def synthetic(fam, m, n, k, seed):
    """(A, Bt) of a synthetic family (the GEMM view, K canonical)."""
    h = np.arange(k) / max(k - 1, 1)
    if fam == "uniform":
        f = lambda r, i0, i1, c: r.uniform(-1, 1, (i1 - i0, k))  # noqa: E731
        return _blocks(m, seed, 1, lambda r, a, b: f(r, a, b, 0)), \
            _blocks(n, seed, 2, lambda r, a, b: f(r, a, b, 1))
    if fam == "same_sign":
        return _blocks(m, seed, 1, lambda r, a, b: r.uniform(0, 1, (b - a, k))), \
            _blocks(n, seed, 2, lambda r, a, b: r.uniform(0, 1, (b - a, k)))
    if fam.startswith("phi"):
        phi = float(fam[3:])
        def g(r, a, b):
            return (r.random((b - a, k)) - 0.5) * np.exp(phi * r.standard_normal((b - a, k)))
        return _blocks(m, seed, 1, g), _blocks(n, seed, 2, g)
    if fam in ("decay_same", "decay_opp", "mixed"):
        da = 10.0 ** (-1 - 9 * h)
        db = da if fam == "decay_same" else da[::-1]
        ra = 10.0 ** (-3 * np.arange(m) / max(m - 1, 1))
        rb = 10.0 ** (-3 * np.arange(n) / max(n - 1, 1))

        def fa(r, a, b):
            X = r.uniform(0.5, 1, (b - a, k)) * r.choice([-1.0, 1.0], (b - a, k)) * \
                da[None, :] * ra[a:b, None]
            if fam == "mixed":                      # half the rows dense
                dense = np.arange(a, b) < m // 2
                X[dense] = r.uniform(-1, 1, (int(dense.sum()), k)) * 0.1
            return X
        fb = lambda r, a, b: r.uniform(0.5, 1, (b - a, k)) * r.choice([-1.0, 1.0], (b - a, k)) * \
            db[None, :] * rb[a:b, None]  # noqa: E731
        return _blocks(m, seed, 1, fa), _blocks(n, seed, 2, fb)
    if fam == "opp_nonsep":
        def fa(r, a, b):
            p = r.uniform(0.5, 2.0, b - a)
            return r.uniform(0.5, 1, (b - a, k)) * r.choice([-1.0, 1.0], (b - a, k)) * \
                10.0 ** (-1 - 9 * h[None, :] ** p[:, None])

        def fb(r, a, b):
            p = r.uniform(0.5, 2.0, b - a)
            return r.uniform(0.5, 1, (b - a, k)) * r.choice([-1.0, 1.0], (b - a, k)) * \
                10.0 ** (-1 - 9 * (1 - h[None, :]) ** p[:, None])
        return _blocks(m, seed, 1, fa), _blocks(n, seed, 2, fb)
    if fam == "outliers":
        def f(r, a, b):
            X = r.standard_normal((b - a, k))
            for _ in range(2):    # two outliers per row (of A) / column (of B), 1e3 .. 1e8
                X[np.arange(b - a), r.integers(0, k, b - a)] *= 10.0 ** r.uniform(3, 8, b - a)
            return X
        return _blocks(m, seed, 1, f), _blocks(n, seed, 2, f)
    if fam == "cancel":
        assert k % 2 == 0
        h2 = k // 2

        def fa(r, a, b):
            X = r.uniform(-1, 1, (b - a, k))
            X[:, h2:] = X[:, :h2] * (1 + 1e-6 * r.uniform(-1, 1, (b - a, h2)))
            return X

        def fb(r, a, b):
            X = r.uniform(-1, 1, (b - a, k))
            X[:, h2:] = -X[:, :h2]
            return X
        return _blocks(m, seed, 1, fa), _blocks(n, seed, 2, fb)
    if fam == "local_df":
        # 1D chain of length L: pair (p, q) at positions x_p, x_q; auxiliary function Q at
        # x_Q (sorted: K follows the chain); |B^Q_pq| ~ e^(-|x_p - x_q| / 2) e^(-|x_Q - mid| / 1.5)
        L = 20.0
        xq = np.sort(np.random.default_rng([seed, 9]).uniform(0, L, k))

        def f(r, a, b):
            xp, xr = r.uniform(0, L, b - a), r.uniform(0, L, b - a)
            mid, dist = (xp + xr) / 2, np.abs(xp - xr)
            mag = np.exp(-dist[:, None] / 2.0) * np.exp(-np.abs(xq[None, :] - mid[:, None]) / 1.5)
            return mag * r.uniform(0.5, 1, mag.shape) * r.choice([-1.0, 1.0], mag.shape)
        return _blocks(m, seed, 1, f), _blocks(n, seed, 2, f)
    raise ValueError(fam)


# ---------------------------------------------------------------------------------------
# Synthetic CC family
# ---------------------------------------------------------------------------------------

def cc_orbitals(o, v, seed):
    rng = np.random.default_rng([seed, 31])
    nc = o // 7
    eo = np.sort(np.concatenate([rng.uniform(-11.5, -10.5, nc), rng.uniform(-1.5, -0.3, o - nc)]))
    ev = np.geomspace(0.05, 50.0, v)
    return eo, ev


def cc_factors(eps_p, eps_q, naux, seed, tag):
    """B^Q_pq = s_Q g(eps_p) g(eps_q) N(0, 1), (p, q, Q) with Q last; s_Q over 8 decades."""
    rng = np.random.default_rng([seed, tag])
    sQ = 10.0 ** (-8 * np.arange(naux) / max(naux - 1, 1))
    g = lambda e: (1 + np.abs(e)) ** -0.5  # noqa: E731
    return (g(eps_p)[:, None, None] * g(eps_q)[None, :, None] * sQ[None, None, :] *
            rng.standard_normal((eps_p.size, eps_q.size, naux)))


def mp2_amplitudes(Bov, eo, ev):
    """t[i, j, a, b] = (ia|jb) / (e_i + e_j - e_a - e_b) from Bov[i, a, Q] (FP64)."""
    o, v, nq = Bov.shape
    X = Bov.reshape(o * v, nq)
    iajb = (X @ X.T).reshape(o, v, o, v)
    D = eo[:, None, None, None] - ev[None, :, None, None] + eo[None, None, :, None] - \
        ev[None, None, None, :]
    return (iajb / D).transpose(0, 2, 1, 3)                              # i, j, a, b


def cc(fam, size, seed):
    """(A, Bt, info) of a synthetic CC contraction at size (o, v, naux)."""
    o, v, naux = size
    eo, ev = cc_orbitals(o, v, seed)
    if fam == "cc_df":
        A = cc_factors(ev, ev, naux, seed, 41).reshape(v * v, naux)
        return A, A.copy(), dict(spec="acQ,bdQ->acbd", o=o, v=v, naux=naux)   # Bvv x Bvv
    Bov = cc_factors(eo, ev, naux, seed, 43)
    t = mp2_amplitudes(Bov, eo, ev)                      # i j a b
    if fam == "cc_ppl":
        Bvv = cc_factors(ev, ev, naux, seed, 44).reshape(v * v, naux)
        acbd = (Bvv @ Bvv.T).reshape(v, v, v, v)          # (ac|bd)[a, c, b, d]
        A = np.ascontiguousarray(t.reshape(o * o, v * v))                         # (ij)(cd)
        Bt = np.ascontiguousarray(acbd.transpose(0, 2, 1, 3).reshape(v * v, v * v))  # (ab)(cd)
        return A, Bt, dict(spec="ijcd,abcd->ijab", o=o, v=v, naux=naux)
    if fam == "cc_ring":
        X = Bov.reshape(o * v, naux)
        ldkc = (X @ X.T)                                   # (ld|kc)[(l,d), (k,c)]
        A = np.ascontiguousarray(t.transpose(2, 0, 1, 3).reshape(v * o, o * v))  # (a,i)(l,d)
        Bt = np.ascontiguousarray(ldkc.T)                 # (k,c)(l,d)
        return A, Bt, dict(spec="ilad,ldkc->aikc", o=o, v=v, naux=naux)
    raise ValueError(fam)


def multi3(m, n, k, seed):
    """IR6 synthetic: X (m x k) decaying along K over 9 decades, partners Y1 decay_opp-like
    (anti-correlated), Y2 phi = 4, Y3 uniform (each n x k)."""
    X, Y1 = synthetic("decay_opp", m, n, k, seed)
    _, Y2 = synthetic("phi4", 1, n, k, seed + 1000)
    _, Y3 = synthetic("uniform", 1, n, k, seed + 2000)
    return X, [Y1, Y2, Y3]


def batch_mixed(b, m, n, k, seed):
    """IR16 synthetic: b batches of (A_l m x k, Bt_l n x k), decay_opp-like; batches l < b/4
    decay over 9 decades along K, the others over 3 (another rate)."""
    h = np.arange(k) / max(k - 1, 1)
    As, Bs = [], []
    for lam in range(b):
        dec = 9.0 if lam < b // 4 else 3.0
        rng = np.random.default_rng([seed, 77, lam])
        da = 10.0 ** (-1 - dec * h)
        As.append(rng.uniform(0.5, 1, (m, k)) * rng.choice([-1.0, 1.0], (m, k)) * da[None, :])
        Bs.append(rng.uniform(0.5, 1, (n, k)) * rng.choice([-1.0, 1.0], (n, k)) * da[None, ::-1])
    return As, Bs


# ---------------------------------------------------------------------------------------
# Real operands (psi4, immerdink-eval scripts/c1/psi4_operands.py)
# ---------------------------------------------------------------------------------------

def load_real(path):
    """The .npz of psi4_operands.py: Boo[k,i,Q], Bov[i,a,Q], Bvv[a,c,Q], eps_occ, eps_vir."""
    z = np.load(path)
    return {x: z[x] for x in z.files}


def real_contractions(R, which):
    """GEMM views of the consumer's contractions (ehrenfest ccsd.py, read-only; NOTES-C1
    §4.3 R1-R7) from real factors R. Returns (A, Bt, info); for R6 (A = X's partners) and R7
    (batched) see real_multi / real_batched. Operands that ccsd.py forms as intermediates are
    formed here in FP64 numpy: u = 2t - t^T (eq 18, swap of the virtuals), L = 2(ld|kc) -
    (lc|kd) (eq 17), p1 - p2 / 2 with p1 = (ki|ac), p2 = sum_ld t_li^ad (kd|lc) (eq 13).
    Bare (undressed) integrals and MP2 amplitudes stand in for the t1-dressed ones and the
    converged t2 [I] (§4.3)."""
    Boo, Bov, Bvv = R["Boo"], R["Bov"], R["Bvv"]
    eo, ev = R["eps_occ"], R["eps_vir"]
    o, v, nq = Bov.shape
    Bvo = Bov.transpose(1, 0, 2)                                          # a, i, Q
    if which == "R1":                       # aiQ,bjQ->abij, K = Q
        X = np.ascontiguousarray(Bvo.reshape(v * o, nq))
        return X, X.copy(), dict(spec="aiQ,bjQ->abij", ext=dict(a=v, i=o, b=v, j=o, Q=nq))
    if which == "R2":                       # acQ,bdQ->acbd, K = Q
        X = np.ascontiguousarray(Bvv.reshape(v * v, nq))
        return X, X.copy(), dict(spec="acQ,bdQ->acbd", ext=dict(a=v, c=v, b=v, d=v, Q=nq))
    if which == "R5":                       # kiQ,acQ->kiac, K = Q
        return (np.ascontiguousarray(Boo.reshape(o * o, nq)),
                np.ascontiguousarray(Bvv.reshape(v * v, nq)),
                dict(spec="kiQ,acQ->kiac", ext=dict(k=o, i=o, a=v, c=v, Q=nq)))
    t = mp2_amplitudes(Bov, eo, ev)                                      # i j a b
    if which == "R2b":                      # ijcd,acbd->abij, K = cd
        Xv = Bvv.reshape(v * v, nq)
        acbd = (Xv @ Xv.T).reshape(v, v, v, v)
        A = np.ascontiguousarray(t.reshape(o * o, v * v))
        Bt = np.ascontiguousarray(acbd.transpose(0, 2, 1, 3).reshape(v * v, v * v))
        return A, Bt, dict(spec="ijcd,acbd->abij", ext=dict(i=o, j=o, c=v, d=v, a=v, b=v))
    X = Bov.reshape(o * v, nq)
    ovov = (X @ X.T).reshape(o, v, o, v)                                 # (ia|jb)[i,a,j,b]
    if which == "R3":                       # ilad,ldkc->aikc, K = ld
        u = 2 * t - t.transpose(0, 1, 3, 2)                               # u[i,l,a,d]
        A = np.ascontiguousarray(u.transpose(2, 0, 1, 3).reshape(v * o, o * v))   # (a,i)(l,d)
        L = 2 * ovov - ovov.transpose(0, 3, 2, 1)                         # L[l,d,k,c]
        Bt = np.ascontiguousarray(L.transpose(2, 3, 0, 1).reshape(o * v, o * v))  # (k,c)(l,d)
        return A, Bt, dict(spec="ilad,ldkc->aikc", ext=dict(i=o, l=o, a=v, d=v, k=o, c=v))
    if which == "R4":                       # kjbc,kiac->abij, K = kc
        A = np.ascontiguousarray(t.transpose(1, 2, 0, 3).reshape(o * v, o * v))   # (j,b)(k,c)
        p1 = (Boo.reshape(o * o, nq) @ Bvv.reshape(v * v, nq).T).reshape(o, o, v, v)  # k i a c
        kdlc = ovov                                                       # (kd|lc)[k,d,l,c]
        tl = t.transpose(1, 2, 0, 3).reshape(o * v, o * v)                # (i,a)(l,d)
        W2 = kdlc.transpose(2, 1, 0, 3).reshape(o * v, o * v)             # (l,d)(k,c)
        p2 = (tl @ W2).reshape(o, v, o, v).transpose(2, 0, 1, 3)          # k i a c
        W = p1 - 0.5 * p2
        Bt = np.ascontiguousarray(W.transpose(1, 2, 0, 3).reshape(o * v, o * v))  # (i,a)(k,c)
        return A, Bt, dict(spec="kjbc,kiac->abij", ext=dict(k=o, j=o, b=v, c=v, i=o, a=v))
    raise ValueError(which)


def real_multi(R):
    """R6 (IR6): X = Bov (k,c) x Q, pre-scaled once; partners Bov (l,d) x Q in ldQ,kcQ->ldkc
    and Bvo (a,i) x Q in aiQ,kcQ->aikc (X is the plan's B in both)."""
    Bov = R["Bov"]
    o, v, nq = Bov.shape
    X = np.ascontiguousarray(Bov.reshape(o * v, nq))
    Bvo = np.ascontiguousarray(Bov.transpose(1, 0, 2).reshape(v * o, nq))
    return X, [X.copy(), Bvo], dict(spec=["ldQ,kcQ->ldkc", "aiQ,kcQ->aikc"],
                                    ext=dict(k=o, c=v, l=o, d=v, a=v, i=o, Q=nq))


def real_batched(R):
    """R7 (IR16): iaQ,ibQ->iab, batch i over every occupied orbital (core included)."""
    Bov = R["Bov"]
    return [np.ascontiguousarray(Bov[i]) for i in range(Bov.shape[0])], \
        [np.ascontiguousarray(Bov[i]) for i in range(Bov.shape[0])]
