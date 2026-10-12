#!/usr/bin/env python3
"""M-d1 (NOTES-C1 §4.3, IR3, IR7, Q25): convergence of scheduled iterations on the CPU, every
contraction computed by the bit-exact emulator (bench/accuracy/kw_engine.py, checked against
spec/oracle.py by its self-test). No GPU; time is M-d2's job.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      tools/memcap.sh timeout 600 python3 bench/accuracy/md1_converge.py SYSTEM [--out DIR]

(timeout INSIDE memcap: memcap.sh runs the command in a systemd scope, which a timeout
around memcap.sh does not stop.)

SYSTEM: jacobi | diis | ppl | rho (the rho-miss rate) | davidson | cg.

Systems (NOTES-C1 §4.3):
  jacobi    ACCURACY §3's Jacobi system (bench/accuracy/iterate.py: v = 40, o = 10, rho = 0.7):
            y[ab,ij] = sum_cd W[ab,cd] x[cd,ij], an emulated GEMM view m = k = 1600, n = 100
            (W fixed, so its shifts are computed once per N);
  diis      the same with DIIS(8) (Pulay, on the Jacobi map), with and without a history
            reset at every precision switch (Q25);
  ppl       a contraction-dominated synthetic fixed point (v = 40, o = 10, rho = 0.7):
            x <- (r - P(x) - Q(x)) / D, the PPL-like P(x)_ijab = sum_cd x_ijcd W_cdab
            (m = 100, k = n = 1600; W fixed) and the ring-like Q(x)_ijab = sum_kc x_ikac V_kcjb
            (m = k = n = 400; V fixed), each with its own precision;
  davidson  report-only (IR3): the lowest 4 roots of H = diag(Delta eps_ia) + 2 B B^T (n = 2048,
            DF-like B over 64 auxiliary functions, scaled to ||2 B B^T|| = 1 Ha), block 8,
            subspace <= 48 with restart;
            sigma = H V is an emulated GEMM view m = k = 2048, n = 8;
  cg        report-only (IR3): SPD A = B B^T / k + mu I (n = 2048, DF-like B, cond 1e6), 16
            right-hand sides, CG per right-hand side with one emulated block matvec A P
            (m = k = 2048, n = 16); schedules fixed, monotone tightening, relaxed (decreasing).

Schedules: fixed N in {6, 8, 10, 12, 14}, N = 20 (the drift reference), and the forcing-rule
ramps of NOTES-C1 §4.2 (the rule is the caller's, not the library's): after iteration t each
contraction c gets an ABSOLUTE target eta r_t / C (r_t = the max-abs entry of the residual the
caller has, Delta (x_t+1 - x_t); C = the number of contractions; ABSOLUTE bounds every entry),
with norm_product_c = max ||a_i|| ||b_j|| of the call's current operands (refreshed every
iteration: a pass over the operands, cheaper than the call); N_c = the smallest level >=
imd_err_min_moduli(levels[0] .. levels[-1]) (MISSED: the top level), raised only at the
iteration boundary and never lowered (monotone=True); levels {5, 6, 8, 10, 12, 14};
eta in {0.1, 0.01}; BOUND, and TYPICAL with fail_prob p = P / C (P = 1e-3 over the run's
iterations x contractions calls, §4.2 step 7).

Metrics: iterations to the relative residual 1e-7 and 1e-10 (FP64 residual of the exact
operator, accurate to ~1e-15 here); the final residual in 80-bit long double; drift against
a reference iteration whose contractions are exact and rounded once (at least as accurate as
Dot2 / double-double contractions) and against the N = 20 trajectory; the phases.
Success (§4.3): every ramp reaches the fixed-N_final (= 14) final residual within 2x and
needs at most one more iteration to each tolerance fixed N_final reaches, on jacobi, diis
(both variants) and ppl.

rho-miss rate (IR7): each call's ELEMENTWISE target e (1e-6, 1e-10) is met at the N that
imd_err_min_moduli (BOUND, levels up to 20) picks for cond = max rho_hat of the PREVIOUS
call's output over its certified outputs, rho_hat from the shipped helper
immerdink.precision.rho_from_output (stage 1, DESIGN §6g.11; bit-identical to spec/params.py
rho_recipe), with the caller's norm upper bounds (FP64 row norms raised by
(1 + (k + 4) 2^-52), then one ulp up). A miss: the call's realized max elementwise error
(exact reference) over those outputs exceeds e; the rate counts calls whose lagged request
was MET (a MISSED request, no level meeting it, is reported apart as capped). The records
before the helper shipped (7260a95) used a local version, rho_hat_local, which also raised U
by (1 + 2^-40); `rho --rho-impl local` reproduces them.
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kw_engine as ke  # noqa: E402
import c2_gate as c2g  # noqa: E402
sys.path.insert(0, os.path.join(ke.ROOT, "python"))
from immerdink import precision  # noqa: E402  (loads no library on import)

o = ke.o
LEVELS = [5, 6, 8, 10, 12, 14]
RHO_LEVELS = [5, 6, 8, 10, 12, 14, 16, 18, 20]   # rho-miss: the lag's reliability, not a cap
FAIL_P = 1e-3
LD = np.longdouble


# ---------------------------------------------------------------------------------------
# Emulated contraction with cached shifts of a fixed operand
# ---------------------------------------------------------------------------------------

class Operand:
    """A row-fiber operand (rows x k: rows of A, or rows of B^T); its shifts are cached per
    (N, rounding), so a fixed operand's are computed once per N."""

    def __init__(self, X):
        self.X = np.ascontiguousarray(X)
        self.st = ke.RowStats(self.X)
        self.norm = ke.norms(self.X)
        self.sig = {}

    def shifts(self, N, rnd="RN"):
        if (N, rnd) not in self.sig:
            self.sig[(N, rnd)] = ke.shifts(self.X, self.st, c2g.plan_of(N, self.X.shape[1], rnd))
        return self.sig[(N, rnd)]


def contract(Aop, Bop, N, rnd="RN"):
    """The emulated product at N (bit-identical to oracle.contract), m x n."""
    pl = c2g.plan_of(N, Aop.X.shape[1], rnd)
    return ke.emulate(Aop.X, Bop.X, Aop.shifts(N, rnd), Bop.shifts(N, rnd), pl, rnd)[0]


def exact(Aop, Bop):
    """(hi, lo): the exact product, hi = RN(x) (the reference contraction), lo = RN(x - hi)."""
    return ke.exact_ref(Aop.X, Bop.X)[:2]


def norm_product(Aop, Bop):
    return float(Aop.norm.max() * Bop.norm.max())


# ---------------------------------------------------------------------------------------
# Precision schedules (the caller's rule)
# ---------------------------------------------------------------------------------------

class Schedule:
    """fixed N | ramp (eta, basis) | relaxed (NORMWISE matvec tolerance) | rho (eps)."""

    def __init__(self, perf, kind, N=None, eta=None, basis="BOUND", ncontr=1, iters=100,
                 eps=None, monotone=True, levels=LEVELS):
        self.perf, self.kind, self.N0, self.eta, self.basis = perf, kind, N, eta, basis
        self.ncontr, self.iters, self.eps = ncontr, iters, eps
        self.monotone, self.levels = monotone, list(levels)
        self.floor = {}
        self.name = (f"fixed{N}" if kind == "fixed" else
                     f"ramp_eta{eta}_{basis}" if kind == "ramp" else
                     "relaxed" if kind == "relaxed" else f"rho_eps{eps}")

    def _snap(self, n):
        return next((lv for lv in self.levels if n is not None and lv >= n), self.levels[-1])

    def _mm(self, shape, acc):
        m, n, k = shape
        r = self.perf.host.min_moduli("ik,kj->ij", dict(i=m, k=k, j=n), accuracy=acc,
                                      n_lo=self.levels[0], n_hi=self.levels[-1], lib=self.perf.h)
        self.last_met = r.status == "SUCCESS"         # False: MISSED, the top level is used
        return self._snap(r.num_moduli if r.status == "SUCCESS" else None)

    def choose(self, c, shape, r_abs=None, nprod=None, rel_tol=None, cond=None):
        """N of contraction c (GEMM view shape (m, n, k)) at this iteration boundary."""
        host = self.perf.host
        if self.kind == "fixed":
            return self.N0
        if self.kind == "ramp":
            if not nprod or r_abs is None:
                want = self.levels[0]                  # x = 0: the product is exactly 0
            else:
                p = FAIL_P / (self.iters * self.ncontr) if self.basis == "TYPICAL" else 0.0
                want = self._mm(shape, host.Accuracy(metric="ABSOLUTE",
                                                     target=self.eta * r_abs / self.ncontr,
                                                     norm_product=nprod, basis=self.basis,
                                                     fail_prob=p))
        elif self.kind == "relaxed":
            want = self._mm(shape, host.Accuracy(metric="NORMWISE", target=rel_tol))
        else:
            want = self.levels[-1] if cond is None else \
                self._mm(shape, host.Accuracy(metric="ELEMENTWISE", target=self.eps,
                                              cond=max(1.0, cond)))
        if self.monotone:
            want = max(want, self.floor.get(c, self.levels[0]))
            self.floor[c] = want
        return want


def norm_ub(na, k):
    """Upper bounds of the true row 2-norms from ke.norms' FP64 values (the caller's §6f.1
    obligation): raised by (1 + (k + 4) 2^-52), then one ulp up."""
    return np.nextafter(na * (1 + (k + 4) * 2.0 ** -52), np.inf)


def rho_hat(perf, D, na, nb, N, k, rnd="RN"):
    """The shipped rigorous rho (precision.rho_from_output, alpha = 1, beta = 0, nu = sqrt k)
    of a call's output D, from norm upper bounds of the operands' FP64 row norms na, nb.
    Returns (rho_hat, certified mask)."""
    return precision.rho_from_output(D, norm_ub(na, k), norm_ub(nb, k), N=N, rounding=rnd,
                                     k=k, lib=perf.h.path)


def rho_hat_local(perf, D, na, nb, N, k, rnd="RN"):
    """The local recipe of the first M-d1 runs (7260a95, before the helper shipped): U = the
    NORMWISE bound of imd_err_estimate at (N, rounding, k) (nu = sqrt k) times (1 + 2^-40);
    n_ij >= ||a_i|| ||b_j|| (FP64 norms raised by (1 + (k + 4) 2^-52)); t = nextafter(U n, inf),
    den = nextafter(|D| - t, 0), certified iff D is finite and normal and den > 0,
    rho_hat = nextafter(n / den, inf). Returns (rho_hat, certified mask)."""
    U = perf.host.err_estimate(N, k, rounding=rnd, lib=perf.h,
                               accuracy=perf.host.Accuracy(metric="NORMWISE")).bound
    U *= 1 + 2.0 ** -40
    up = 1 + (k + 4) * 2.0 ** -52
    nab = np.nextafter((na * up)[:, None] * (nb * up)[None, :], np.inf)
    t = np.nextafter(U * nab, np.inf)
    den = np.nextafter(np.abs(D) - t, 0)
    cert = np.isfinite(D) & (np.abs(D) >= 2.0 ** -1022) & (den > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(cert, np.nextafter(nab / np.where(cert, den, 1.0), np.inf), np.inf)
    return rho, cert


# ---------------------------------------------------------------------------------------
# Fixed-point systems: calls(x) -> [(A operand, B operand, post)], y_c = post(A B^T)
# ---------------------------------------------------------------------------------------

def jacobi_setup(v=40, o_=10, rho=0.7, seed=7):
    """bench/accuracy/iterate.py's setup (copied: iterate imports the CUDA harness)."""
    rng = np.random.default_rng(seed)
    ev = np.sort(rng.uniform(0.5, 3.0, v))
    eo = np.sort(rng.uniform(-2.0, -0.5, o_))
    Delta = ev[:, None, None, None] + ev[None, :, None, None] - eo[None, None, :, None] - \
        eo[None, None, None, :]
    G = rng.standard_normal((v * v, v * v))
    Wm = (G + G.T) / 2
    eab = (ev[:, None] + ev[None, :]).reshape(-1)
    Wm *= np.exp(-0.5 * (eab[:, None] + eab[None, :] - 2))
    dmin = Delta.min(axis=(2, 3)).reshape(-1)
    S = Wm / np.sqrt(dmin[:, None] * dmin[None, :])
    lam = np.max(np.abs(np.linalg.eigvalsh(S)))
    Wm *= rho / lam
    r = rng.standard_normal((v, v, o_, o_))
    return Delta, Wm, r


class Jacobi:
    """x[ab, ij] <- (r - W x) / Delta."""
    name = "jacobi"

    def __init__(self):
        Delta, self.Wm, r = jacobi_setup()
        v, o_ = Delta.shape[0], Delta.shape[2]
        self.D = Delta.reshape(v * v, o_ * o_)
        self.r = r.reshape(v * v, o_ * o_)
        self.W = Operand(self.Wm)
        self.rnorm = np.linalg.norm(self.r)

    def calls(self, x):
        return [(self.W, Operand(x.T), None)]

    def update(self, x, ys):
        return (self.r - ys[0]) / self.D

    def apply64(self, x):
        return self.Wm @ x

    def apply_ld(self, x):
        return self.Wm.astype(LD) @ x.astype(LD)


class PPL:
    """x[ij, ab] <- (r - P(x) - Q(x)) / D (see the module docstring)."""
    name = "ppl"

    def __init__(self, v=40, o_=10, rho=0.7, seed=11):
        rng = np.random.default_rng(seed)
        ev = np.sort(rng.uniform(0.5, 3.0, v))
        eo = np.sort(rng.uniform(-2.0, -0.5, o_))
        self.v, self.o = v, o_
        self.D = (ev[None, None, :, None] + ev[None, None, None, :] - eo[:, None, None, None] -
                  eo[None, :, None, None]).reshape(o_ * o_, v * v)
        G = rng.standard_normal((v * v, v * v))
        Wm = (G + G.T) / 2
        eab = (ev[:, None] + ev[None, :]).reshape(-1)
        Wm *= np.exp(-0.5 * (eab[:, None] + eab[None, :] - 2))
        ec = (ev[None, :] - eo[:, None]).reshape(-1)                 # (k,c) and (j,b)
        Vm = rng.standard_normal((o_ * v, o_ * v)) * np.exp(-0.4 * (ec[:, None] + ec[None, :]))
        self.Wm, self.Vm = Wm, Vm
        self.r = rng.standard_normal((o_ * o_, v * v))
        y = rng.standard_normal(self.r.shape)
        for _ in range(60):          # spectral radius of x -> (P + Q)(x) / D, then rescale
            z = self.apply64(y) / self.D
            lam = np.linalg.norm(z) / np.linalg.norm(y)
            y = z / np.linalg.norm(z)
        self.Wm *= rho / lam
        self.Vm *= rho / lam
        self.WT = Operand(self.Wm.T)                                 # B of P: rows (a,b)
        self.VT = Operand(self.Vm.T)                                 # B of Q: rows (j,b)
        self.rnorm = np.linalg.norm(self.r)

    def ring_in(self, x):
        o_, v = self.o, self.v
        return x.reshape(o_, o_, v, v).transpose(0, 2, 1, 3).reshape(o_ * v, o_ * v)   # (ia)(kc)

    def ring_out(self, y):
        o_, v = self.o, self.v
        return y.reshape(o_, v, o_, v).transpose(0, 2, 1, 3).reshape(o_ * o_, v * v)   # (ij)(ab)

    def calls(self, x):
        return [(Operand(x), self.WT, None), (Operand(self.ring_in(x)), self.VT, self.ring_out)]

    def update(self, x, ys):
        return (self.r - ys[0] - ys[1]) / self.D

    def apply64(self, x):
        return x @ self.Wm + self.ring_out(self.ring_in(x) @ self.Vm)

    def apply_ld(self, x):
        xl = x.astype(LD)
        return xl @ self.Wm.astype(LD) + self.ring_out(self.ring_in(xl) @ self.Vm.astype(LD))


def residual(sysm, x):
    return float(np.linalg.norm(sysm.r - sysm.D * x - sysm.apply64(x)) / sysm.rnorm)


def residual_ld(sysm, x):
    res = sysm.r.astype(LD) - sysm.D.astype(LD) * x.astype(LD) - sysm.apply_ld(x)
    return float(np.sqrt(np.sum(res * res)) / sysm.rnorm)


def ys_of(calls, Ns, mode):
    out = []
    for (Aop, Bop, post), N in zip(calls, Ns):
        y = exact(Aop, Bop)[0] if mode == "ref" else contract(Aop, Bop, N)
        out.append(post(y) if post else y)
    return out


def iterate(sysm, sched, iters, diis=0, reset=False, mode="emu", refs=None, keep=False):
    """One run. refs: {name: trajectory} for drift; keep: return the trajectory too."""
    x = np.zeros_like(sysm.r)
    hx, he = [], []
    N_hist, res_hist, traj = [], [], []
    drifts = {nm: [] for nm in (refs or {})}
    r_abs, Nprev, it_tol = None, None, {1e-7: None, 1e-10: None}
    best, stall = math.inf, 0
    for t in range(iters):
        calls = sysm.calls(x)
        Ns = [sched.choose(c, (A.X.shape[0], B.X.shape[0], A.X.shape[1]), r_abs=r_abs,
                           nprod=norm_product(A, B)) for c, (A, B, _) in enumerate(calls)] \
            if mode == "emu" else [None] * len(calls)
        if diis and reset and Nprev is not None and Ns != Nprev:
            hx, he = [], []                       # history reset at a precision switch
        Nprev = Ns
        g = sysm.update(x, ys_of(calls, Ns, mode))
        e = g - x
        r_abs = float(np.abs(e * sysm.D).max())   # the caller's residual Delta (g - x)
        x = g
        if diis:
            hx, he = (hx + [g])[-diis:], (he + [e])[-diis:]
            nh = len(he)
            if nh >= 2:
                Bm = np.zeros((nh + 1, nh + 1))
                Bm[:nh, :nh] = [[np.vdot(a, b) for b in he] for a in he]
                Bm[nh, :nh] = Bm[:nh, nh] = -1
                rhs = np.zeros(nh + 1)
                rhs[nh] = -1
                try:
                    c = np.linalg.solve(Bm, rhs)[:nh]
                    x = sum(ci * xi for ci, xi in zip(c, hx))
                except np.linalg.LinAlgError:
                    pass
        res = residual(sysm, x)
        res_hist.append(res)
        N_hist.append(Ns)
        for tol in it_tol:
            if it_tol[tol] is None and res <= tol:
                it_tol[tol] = t + 1
        for nm, tr in (refs or {}).items():
            if t < len(tr):
                drifts[nm].append(float(np.linalg.norm(x - tr[t]) / np.linalg.norm(tr[t])))
        if keep:
            traj.append(x.copy())
        best, stall = (res, 0) if res < 0.9 * best else (best, stall + 1)
        if stall >= 12 and t >= 25 and not keep:
            break
    rec = dict(iters_run=len(res_hist), it_1e7=it_tol[1e-7], it_1e10=it_tol[1e-10],
               final_residual=residual_ld(sysm, x), residuals=res_hist, N=N_hist,
               phases=phases(N_hist) if mode == "emu" else None,
               drift={nm: [max(d), d[-1]] for nm, d in drifts.items() if d})
    return rec, traj


def phases(Nh):
    """Run-length phases [N per contraction, first iteration, length]."""
    out = []
    for t, N in enumerate(Nh):
        if out and out[-1][0] == list(N):
            out[-1][2] += 1
        else:
            out.append([list(N), t + 1, 1])
    return out


def success(ramp, fixed):
    """§4.3: the ramp's final residual within 2x of fixed N_final's, and at most one more
    iteration to each tolerance fixed N_final reaches."""
    ok_res = ramp["final_residual"] <= 2 * fixed["final_residual"]
    ok_it = all(fixed[t] is None or (ramp[t] is not None and ramp[t] <= fixed[t] + 1)
                for t in ("it_1e7", "it_1e10"))
    return dict(pass_=bool(ok_res and ok_it), residual_within_2x=bool(ok_res),
                iterations_within_1=bool(ok_it),
                residual_ratio=ramp["final_residual"] / fixed["final_residual"],
                extra_iterations={t: (None if fixed[t] is None or ramp[t] is None
                                      else ramp[t] - fixed[t]) for t in ("it_1e7", "it_1e10")})


def fixed_point_study(sysm, perf, iters, log, prov, diis=0):
    """The exact-contraction reference and N = 20 first (their trajectories are the drift
    references), then fixed N = 6..14 and the ramps; success of each ramp against fixed 14."""
    variants = [False] if not diis else [True, False]
    for reset in variants:
        tag = sysm.name + (f"_diis{diis}_" + ("reset" if reset else "noreset") if diis else "")
        t0 = time.process_time()
        sch = lambda **kw: Schedule(perf, ncontr=len(sysm.calls(sysm.r)), iters=iters, **kw)  # noqa: E731
        ref_rec, ref_tr = iterate(sysm, sch(kind="fixed", N=None), iters, diis, reset, "ref",
                                  keep=True)
        n20_rec, n20_tr = iterate(sysm, sch(kind="fixed", N=20), iters, diis, reset,
                                  refs={"ref": ref_tr}, keep=True)
        recs = {"reference_exact": ref_rec, "fixed20": n20_rec}
        refs = {"ref": ref_tr, "N20": n20_tr}
        for sc in ([sch(kind="fixed", N=N) for N in (6, 8, 10, 12, 14)] +
                   [sch(kind="ramp", eta=eta, basis=b) for eta in (0.1, 0.01)
                    for b in ("BOUND", "TYPICAL")]):
            recs[sc.name] = iterate(sysm, sc, iters, diis, reset, refs=refs)[0]
        for nm, rec in recs.items():
            if nm.startswith("ramp"):
                rec["success"] = success(rec, recs["fixed14"])
            log(dict(prov, system=tag, schedule=nm, **rec,
                     cpu_s_system=round(time.process_time() - t0, 1)))
        del ref_tr, n20_tr, refs


def rho_miss(sysm, perf, eps, iters, log, prov, impl="shipped"):
    """IR7: the lagged-rho ELEMENTWISE schedule; miss rate per contraction call. impl:
    "shipped" (precision.rho_from_output) or "local" (rho_hat_local, the first runs)."""
    rho_fn = rho_hat if impl == "shipped" else rho_hat_local
    t0 = time.process_time()
    sc = Schedule(perf, "rho", eps=eps, monotone=False, levels=RHO_LEVELS)
    x = np.zeros_like(sysm.r)
    conds, certs, rows = {}, {}, []
    for t in range(iters):
        calls = sysm.calls(x)
        ys, row = [], []
        for c, (A, B, post) in enumerate(calls):
            sc.last_met = None                         # None: no statistics yet (top level)
            N = sc.choose(c, (A.X.shape[0], B.X.shape[0], A.X.shape[1]), cond=conds.get(c))
            met = sc.last_met
            D = contract(A, B, N)
            hi, lo = exact(A, B)
            with np.errstate(divide="ignore", invalid="ignore"):
                ew = np.where(hi != 0, np.abs((D - hi) - lo) / np.abs(hi), np.inf)
            prev = certs.get(c)
            realized = float(ew[prev].max()) if prev is not None and prev.any() else None
            rho, cert = rho_fn(perf, D, A.norm, B.norm, N, A.X.shape[1])
            conds[c] = float(rho[cert].max()) if cert.any() else None
            certs[c] = cert
            row.append(dict(c=c, N=N, met=met, cond_next=conds[c], certified=int(cert.sum()),
                            outputs=int(cert.size), realized=realized,
                            miss=None if realized is None else bool(realized > eps)))
            ys.append(post(D) if post else D)
        rows.append(row)
        x = sysm.update(x, ys)
        if residual(sysm, x) < 1e-13:
            break
    ms = [r["miss"] for row in rows for r in row if r["miss"] is not None and r["met"]]
    capped = [r["miss"] for row in rows for r in row if r["miss"] is not None and r["met"] is False]
    log(dict(prov, system=sysm.name, schedule=f"rho_eps{eps}", eps=eps, iterations=len(rows),
             rho_impl=impl, levels=RHO_LEVELS, checked=len(ms), misses=int(sum(ms)),
             miss_rate=sum(ms) / len(ms) if ms else None, capped_calls=len(capped),
             capped_misses=int(sum(capped)), per_iteration=rows,
             cpu_s=round(time.process_time() - t0, 1)))


# ---------------------------------------------------------------------------------------
# Block Davidson and CG (report-only, IR3)
# ---------------------------------------------------------------------------------------

def davidson_setup(o_=16, v=128, naux=64, seed=13):
    rng = np.random.default_rng(seed)
    eo = np.sort(np.concatenate([rng.uniform(-11.5, -10.5, o_ // 7),
                                 rng.uniform(-1.5, -0.3, o_ - o_ // 7)]))
    ev = np.geomspace(0.05, 50.0, v)
    d = (ev[None, :] - eo[:, None]).reshape(-1)                      # Delta eps_ia
    g = (1 + np.abs(eo))[:, None, None] ** -0.5 * (1 + np.abs(ev))[None, :, None] ** -0.5
    sQ = 10.0 ** (-2 * np.arange(naux) / naux)
    B = (g * sQ[None, None, :] * rng.standard_normal((o_, v, naux))).reshape(o_ * v, naux)
    BB = B @ B.T
    BB *= 0.5 / np.linalg.norm(BB, 2)            # ||2 B B^T|| = 1 Ha: a two-electron scale
    return np.diag(d) + 2 * BB, d


def run_davidson(sched, H, Hop, d, nroots=4, block=8, vmax=48, iters=80, recompute=False):
    """Block Davidson: V0 = unit vectors at the smallest diagonal entries; sigma = H V emulated
    for the new block only; Rayleigh-Ritz; diagonal-preconditioned corrections; restart to the
    block of Ritz vectors (and their stored sigma) when the subspace would exceed vmax.
    Ramp targets: ABSOLUTE eta ||r|| (r: the residuals the caller has, from sigma).
    recompute: when N rises, collapse the subspace to the current Ritz block and recompute its
    sigma at the new N (products from a lower-N phase otherwise stay in the subspace)."""
    n = H.shape[0]
    V = np.zeros((n, block))
    V[np.argsort(d)[:block], np.arange(block)] = 1.0
    Vall, S = np.zeros((n, 0)), np.zeros((n, 0))
    exact_ev = np.linalg.eigvalsh(H)[:nroots]
    res, Nh, it_tol, rcall = [], [], {1e-7: None, 1e-10: None}, None
    Nprev, X = None, None
    for t in range(iters):
        Vop = Operand(V.T)
        N = sched.choose(0, (n, block, n), r_abs=rcall, nprod=norm_product(Hop, Vop))
        if recompute and Nprev is not None and N != Nprev:
            Vall, S = X.copy(), contract(Hop, Operand(X.T), N)    # fresh products at N
        Nprev = N
        Vall = np.hstack([Vall, V])
        S = np.hstack([S, contract(Hop, Vop, N)])
        Hs = Vall.T @ S
        th, Y = np.linalg.eigh((Hs + Hs.T) / 2)
        X = Vall @ Y[:, :block]
        R = S @ Y[:, :block] - X * th[None, :block]
        rcall = float(np.linalg.norm(R[:, :nroots], axis=0).max())
        true = float(np.linalg.norm(H @ X[:, :nroots] - X[:, :nroots] * th[None, :nroots],
                                    axis=0).max())
        res.append(true)
        Nh.append([N])
        for tl in it_tol:
            if it_tol[tl] is None and true <= tl:
                it_tol[tl] = t + 1
        if true <= 1e-11:
            break
        den = th[None, :block] - d[:, None]
        C = R / np.where(np.abs(den) > 1e-8, den, 1e-8)
        if Vall.shape[1] + block > vmax:
            Vall, S = X.copy(), S @ Y[:, :block]
        for _ in range(2):
            C -= Vall @ (Vall.T @ C)
        V = np.linalg.qr(C)[0]
    return dict(iters_run=len(res), it_1e7=it_tol[1e-7], it_1e10=it_tol[1e-10],
                residuals=res, N=Nh, phases=phases(Nh),
                eig_error=float(np.max(np.abs(th[:nroots] - exact_ev))))


def cg_setup(n=2048, naux=128, nrhs=16, cond=1e6, seed=17):
    rng = np.random.default_rng(seed)
    sQ = 10.0 ** (-1.5 * np.arange(naux) / naux)
    B = rng.standard_normal((n, naux)) * sQ[None, :]
    G = B @ B.T / naux
    mu = np.linalg.eigvalsh(G)[-1] / cond
    return G + mu * np.eye(n), rng.standard_normal((n, nrhs))


def run_cg(sched, A, Aop, b, iters=400, tol=1e-10, mode="relaxed", eta=0.1):
    """CG per right-hand side, one emulated block matvec A P per iteration. Ramp: ABSOLUTE
    eta ||r||_inf (monotone; the fixed-point forcing rule); relaxed: NORMWISE matvec tolerance
    min(1e-2, tol / ||r_k||) (decreasing precision, monotone=False; no iteration-count factor,
    so the attainable residual keeps the inexact-Krylov gap: the theory's form scales it,
    e.g. tol / (m_max ||r_k||)); mode "tighten": NORMWISE
    eta ||r_k|| / ||b|| (monotone tightening of the relative matvec accuracy); kind "fp64":
    the FP64 BLAS matvec (the attainable-accuracy baseline). True residual max_rhs ||b - A x|| / ||b|| in FP64
    every iteration, 80-bit long double at the end."""
    n, m = b.shape
    x, r = np.zeros_like(b), b.copy()
    p = r.copy()
    rr = np.sum(r * r, axis=0)
    bn = np.linalg.norm(b, axis=0)
    res, Nh, it_tol = [], [], {1e-7: None, 1e-10: None}
    for t in range(iters):
        Pop = Operand(p.T)
        rk = float((np.sqrt(rr) / bn).max())
        rel = min(1e-2, tol / max(rk, 1e-300)) if mode == "relaxed" else eta * rk
        N = sched.choose(0, (n, m, n), r_abs=float(np.abs(r).max()), nprod=norm_product(Aop, Pop),
                         rel_tol=rel) if sched.kind != "fp64" else 0
        Ap = A @ p if sched.kind == "fp64" else contract(Aop, Pop, N)   # n x m
        pAp = np.sum(p * Ap, axis=0)
        alpha = np.divide(rr, pAp, out=np.zeros_like(rr), where=pAp != 0)
        x += alpha * p
        r -= alpha * Ap
        rr_new = np.sum(r * r, axis=0)
        p = r + np.divide(rr_new, rr, out=np.zeros_like(rr), where=rr != 0) * p
        rr = rr_new
        true = float((np.linalg.norm(b - A @ x, axis=0) / bn).max())
        res.append(true)
        Nh.append([N])
        for tl in it_tol:
            if it_tol[tl] is None and true <= tl:
                it_tol[tl] = t + 1
        if true <= tol * 1e-2 or (t > 80 and min(res[-40:]) >= 0.9 * min(res[:-40])):
            break
    resl = b.astype(LD) - A.astype(LD) @ x.astype(LD)
    final = float(np.max(np.sqrt(np.sum(resl * resl, axis=0)) / bn))
    return dict(iters_run=len(res), it_1e7=it_tol[1e-7], it_1e10=it_tol[1e-10],
                attainable=min(res), final_residual=final, residuals=res, N=Nh,
                phases=phases(Nh))


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("system", choices=["jacobi", "diis", "ppl", "rho", "davidson", "cg"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--rho-impl", choices=["shipped", "local"], default="shipped",
                    help="rho: precision.rho_from_output (default) or the first runs' local recipe")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()
    t0 = time.process_time()
    ke.self_test(verbose=True)
    host, h = c2g.load_host()
    prov = c2g.provenance(h)
    perf = c2g.Perf(host, h)
    fout = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        fout = open(os.path.join(args.out, f"md1_{args.system}{args.tag}.jsonl"), "a")

    def log(rec):
        if fout:
            fout.write(json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist")
                                  else str(v)) + "\n")
            fout.flush()
        show = ("system", "schedule", "it_1e7", "it_1e10", "final_residual", "attainable",
                "eig_error", "miss_rate", "misses", "checked", "success", "drift", "phases")
        print(json.dumps({k: rec[k] for k in show if k in rec}, default=str)[:500], flush=True)

    s = args.system
    if s == "jacobi":
        fixed_point_study(Jacobi(), perf, args.iters or 100, log, prov)
    elif s == "diis":
        fixed_point_study(Jacobi(), perf, args.iters or 40, log, prov, diis=8)
    elif s == "ppl":
        fixed_point_study(PPL(), perf, args.iters or 100, log, prov)
    elif s == "rho":
        for eps in (1e-6, 1e-10):
            rho_miss(Jacobi(), perf, eps, args.iters or 40, log, prov, args.rho_impl)
            rho_miss(PPL(), perf, eps, args.iters or 40, log, prov, args.rho_impl)
    elif s == "davidson":
        H, d = davidson_setup()
        Hop = Operand(H)
        for rc in (False, True):
            for sc in (([Schedule(perf, "fixed", N=N) for N in (8, 14)] if not rc else []) +
                       [Schedule(perf, "ramp", eta=eta, levels=[6, 8, 10, 12, 14])
                        for eta in (0.1, 0.01)]):
                log(dict(prov, system="davidson", schedule=sc.name + ("_recompute" if rc else ""),
                         **run_davidson(sc, H, Hop, d, recompute=rc)))
    elif s == "cg":
        A, b = cg_setup()
        Aop = Operand(A)
        runs = [(Schedule(perf, "fixed", N=N), "relaxed") for N in (8, 10, 14)] + \
            [(Schedule(perf, "ramp", eta=0.1), "relaxed"),
             (Schedule(perf, "relaxed", monotone=False, levels=[6, 8, 10, 12, 14]), "relaxed"),
             (Schedule(perf, "relaxed", monotone=True, levels=[6, 8, 10, 12, 14]), "tighten"),
             (Schedule(perf, "fp64"), "relaxed")]
        for sc, mode in runs:
            name = "fp64_matvec" if sc.kind == "fp64" else \
                ("tighten_rel_eta0.1" if mode == "tighten" else sc.name)
            log(dict(prov, system="cg", schedule=name, **run_cg(sc, A, Aop, b, mode=mode)))
    print(f"done: {time.process_time() - t0:.1f} CPU-s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
