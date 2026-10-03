"""Shared helpers for the M5 accuracy study (bench/accuracy/).

* `Runner`: one dense contraction through the C API (tests/harness.py ctypes wrappers),
  plans cached per (labels, extents, N, rounding, mode).
* GEMM view of a dense einsum without batch labels (free A labels, contracted labels,
  free B labels, in the caller's label order).
* The DESIGN §3.8 bound per output element, with the exact shifts sigma/tau
  (§3.2 integer rule, the same arithmetic as spec/oracle.py scale_vector).
* An essentially exact reference for sampled outputs: every product split exactly
  (Dekker TwoProd), then math.fsum (correctly rounded) gives hi and the remainder lo,
  so |exact - (hi + lo)| <= 2^-53 |lo|.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "spec"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import harness as hz  # noqa: E402
import oracle as o  # noqa: E402

U = 2.0**-53
RND = {"RN": hz.RN, "TRUNC": hz.TRUNC}


def default_lib():
    return os.environ.get("IMMERDINK_LIB", hz.default_lib())


class Runner:
    """Dense tensors in, dense C out (labels are single characters)."""

    def __init__(self, lib_path=None):
        self.lib = hz.Library(lib_path or default_lib())
        self.cuda = hz.Cuda()
        self.plans = {}
        self.ws = None
        self.ws_bytes = 0

    def _plan(self, key, tA, tB, tC, N, rnd, mode):
        if key not in self.plans:
            st, ws = self.lib.workspace_size(tA, tB, tC, N, 0, RND[rnd], mode)
            if st != "SUCCESS":
                return st, None, 0
            st, plan = self.lib.plan_create(tA, tB, tC, N, 0, RND[rnd], mode)
            if st != "SUCCESS":
                return st, None, 0
            self.plans[key] = (plan, ws, (tA, tB, tC))
        plan, ws, _ = self.plans[key]
        return "SUCCESS", plan, ws

    def contract(self, labA, A, labB, B, labC, extC, N, rnd="RN", mode=hz.EMULATE):
        """C = einsum(labA,labB->labC)(A, B) via imd_contract (alpha 1, beta 0).
        Returns (status, C or None)."""
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        tA = hz.make_tensor(list(A.shape), hz.dense(list(A.shape)), [ord(c) for c in labA])
        tB = hz.make_tensor(list(B.shape), hz.dense(list(B.shape)), [ord(c) for c in labB])
        tC = hz.make_tensor(list(extC), hz.dense(list(extC)), [ord(c) for c in labC])
        key = (labA, A.shape, labB, B.shape, labC, tuple(extC), N, rnd, mode)
        st, plan, wsb = self._plan(key, tA, tB, tC, N, rnd, mode)
        if st != "SUCCESS":
            return st, None
        if wsb > self.ws_bytes:
            if self.ws is not None:
                self.cuda.free(self.ws)
            self.ws, self.ws_bytes = self.cuda.malloc(wsb), wsb
        C = np.zeros(extC, dtype=np.float64)
        dA, dB, dC = (self.cuda.malloc(x.nbytes) for x in (A, B, C))
        try:
            self.cuda.upload(dA, A)
            self.cuda.upload(dB, B)
            st = self.lib.contract(plan, 1.0, dA.value, dB.value, 0.0, dC.value,
                                   self.ws.value if self.ws is not None else None, wsb)
            self.cuda.sync()
            if st != "SUCCESS":
                return st, None
            self.cuda.download(C, dC)
        finally:
            for p in (dA, dB, dC):
                self.cuda.free(p)
        return "SUCCESS", C

    def close(self):
        for plan, _, _ in self.plans.values():
            self.lib.plan_destroy(plan)
        self.plans = {}
        if self.ws is not None:
            self.cuda.free(self.ws)
            self.ws = None


# ---------------------------------------------------------------------------------------
# GEMM view of a dense einsum (no batch labels)
# ---------------------------------------------------------------------------------------

class View:
    def __init__(self, labA, labB, labC, ext):
        """ext: dict label -> extent."""
        self.labA, self.labB, self.labC, self.ext = labA, labB, labC, ext
        assert not (set(labA) & set(labB) & set(labC)), "batch labels not supported here"
        self.I = [c for c in labA if c in labC]
        self.J = [c for c in labB if c in labC]
        self.K = [c for c in labA if c in labB]
        assert sorted(self.I + self.J) == sorted(labC)
        self.m = math.prod(ext[c] for c in self.I)
        self.n = math.prod(ext[c] for c in self.J)
        self.k = math.prod(ext[c] for c in self.K)
        self.shapeA = [ext[c] for c in labA]
        self.shapeB = [ext[c] for c in labB]
        self.shapeC = [ext[c] for c in labC]

    def a_from_mat(self, Am):
        """Tensor A (labA order) from the GEMM matrix A (m x k, rows I, cols K)."""
        src = self.I + self.K
        T = Am.reshape([self.ext[c] for c in src])
        return np.ascontiguousarray(T.transpose([src.index(c) for c in self.labA]))

    def b_from_mat(self, Bm):
        src = self.K + self.J
        T = Bm.reshape([self.ext[c] for c in src])
        return np.ascontiguousarray(T.transpose([src.index(c) for c in self.labB]))

    def c_to_mat(self, C):
        dst = self.I + self.J
        return np.ascontiguousarray(C.transpose([self.labC.index(c) for c in dst])).reshape(
            self.m, self.n)


# ---------------------------------------------------------------------------------------
# §3.2 shifts and the §3.8 bound
# ---------------------------------------------------------------------------------------

def s_gamma(vec):
    """(S, gamma) of one row/column (§3.2 steps 2-3); (0, None) for a zero row."""
    v = np.asarray(vec, dtype=np.float64)
    v = v[v != 0.0]
    if v.size == 0:
        return 0, None
    f, E = np.frexp(np.abs(v))
    mant = (f * 2.0**53).astype(np.int64)      # exact: |f| in [0.5, 1)
    e = E.astype(np.int64) - 53                # |v| = mant 2^e
    gamma = int(E.max()) - 1                   # ilogb(max |v|)
    F = o.F
    S = 0
    for mm, t in zip(mant.tolist(), (2 * (e - gamma) + F).tolist()):
        sq = mm * mm
        S += (sq << t) if t >= 0 else -((-sq) >> (-t))
    return S, gamma


def sigma_of(sg, plan):
    S, gamma = sg
    if S == 0:
        return 0
    return o.largest_u(plan.shift_lhs_mul * S, plan.shift_rhs) - gamma


def bound(N, k, rnd, na, nb, sig, tau, D):
    """DESIGN §3.8 bound (float; only the comparison near 1 would need exact arithmetic).
    na, nb: row/col 2-norms (arrays broadcastable); sig, tau: int shifts; D: computed."""
    plan = o.Plan(N, k, rnd)
    T, kap = plan.T, plan.kappa
    s = math.sqrt(1 + k / 2.0**o.F)
    ex = (-np.asarray(sig, dtype=np.float64) - np.asarray(tau, dtype=np.float64))
    if rnd == "RN":
        t1 = 2 * math.sqrt(k) * s / (T - kap / 2)
        t2 = k * np.exp2(ex - 2)
    else:
        t1 = 4 * math.sqrt(k) * s / T
        t2 = k * np.exp2(ex)
    return t1 * na * nb + t2 + U * np.abs(D)


# ---------------------------------------------------------------------------------------
# Reference for sampled outputs
# ---------------------------------------------------------------------------------------

_SPLIT = 134217729.0  # 2^27 + 1


def two_prod(a, b):
    p = a * b
    def split(x):
        c = _SPLIT * x
        hi = c - (c - x)
        return hi, x - hi
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def exact_ref(arow, bcol):
    """(hi, lo) with hi + lo == sum a_h b_h to within 2^-53 |lo| (no over/underflow)."""
    p, e = two_prod(arow, bcol)
    terms = p.tolist() + e.tolist()
    hi = math.fsum(terms)
    lo = math.fsum(terms + [-hi])
    return hi, lo


def norms(M, axis):
    """Row (axis=1) or column (axis=0) 2-norms, scaled to avoid over/underflow."""
    mx = np.max(np.abs(M), axis=axis)
    mx = np.where(mx == 0, 1.0, mx)
    sc = M / (mx[:, None] if axis == 1 else mx[None, :])
    return mx * np.sqrt(np.sum(sc * sc, axis=axis))
