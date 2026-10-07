"""Precision phases chosen from targets on the host (DESIGN v1.19 §6g.10-§6g.11; NOTES-C1 §4.2,
stage 1 of ROADMAP C1(b)).

An iterative method runs each contraction at the smallest N its current target needs and
switches N between calls (§6g: a plan and a pack per N). This module composes the existing
host queries per phase: `imd_err_min_moduli` (the cheapest N for a target), `imd_perf_rank`
(the configurations at the allowed levels), `imd_perf_predict_pack` (one packed role) and the
footprint queries, through a `planner.Planner`. It computes in Python (levels, the monotone
floor, share groups, per-role pack costs), so it lives here and not in planner.py, whose
queries compute nothing (§6e.4). It runs once per phase switch, off the hot path. No GPU, no
data, no policy: the residual -> target rule is the caller's (bench/accuracy/schedule.py is an
example), and immerdink never changes N on its own (ehrenfest D8).

    from immerdink import planner, precision
    from immerdink.host import Accuracy
    pp = precision.PrecisionPlan(planner.Planner(),
                                 {"ring": ("kjbc,kiac->abij", ext, 2),     # 2 calls / iteration
                                  "df": ("acQ,bdQ->acbd", ext2)},
                                 packs={"df": {"NONE", "B"}},              # B may be packed
                                 reuse={"df": {"B": "phase"}})             # B is invariant
    for name, by_level in pp.configs().items(): ...      # create every plan before the loop
    ph = pp.phase({"ring": Accuracy("ABSOLUTE", 1e-8, norm_product=3.0),
                   "df": Accuracy("NORMWISE", 1e-10)}, iterations=4)
    ph.N, ph.g, ph.packed, ph.status, ph.time_s, ph.tflops_eq
    rho, ok = precision.rho_from_output(C, row_ub, col_ub, N=8, rounding="RN", k=k)

rho_from_output: the rigorous rho of a previous output for ELEMENTWISE targets (§6g.11,
[cert: cert_rho_recipe], spec/params.py rho_recipe is its reference).
"""

import dataclasses
import math
import numbers
import threading

from . import host
from . import planner as _planner
from .host import ImmerdinkError, STATUS

__all__ = ["PrecisionPlan", "Phase", "Choice", "Footprints", "rho_from_output", "DIRECT",
           "PHASE_STATUSES"]

DIRECT = "DIRECT"        # the `levels` sentinel: native FP64 when no emulated level meets
PHASE_STATUSES = ("MET", "MISSED", "BUDGET", "DIRECT")
_PACK_MODES = ("NONE", "A", "B", "AB")
_ROLES = {"NONE": (), "A": ("A",), "B": ("B",), "AB": ("A", "B")}
_MODE = {(): "NONE", ("A",): "A", ("B",): "B", ("A", "B"): "AB"}
_OPERAND = {"A": 0, "B": 1}
_VEC_ORDER = {"GENERATE": 0, "RESIDENT": 1}
_REUSE_KINDS = ("call", "iteration", "phase")
_PERF_FIELDS = ("regime", "cache", "clock_mhz")
_DBL_MIN = 2.0**-1022


# ------------------------------------------------------------------------------------------
# Results

class Choice:
    """One contraction's configuration at one level: the rank candidate (a planner.Candidate
    view, `candidate`) re-keyed with the per-role pack cost (NOTES-C1 IR2):

        time_s = contract_s + sum over packed roles of pack_s[role] / reuse[role]

    (added in the order A, B). level: the `levels` entry (an int N, or "DIRECT");
    num_moduli, residue_group, vec_mode, mode, rounding: the plan options (options() gives them
    as Plan keywords); packed "NONE" / "A" / "B" / "AB"; pack_specs / pack_s / pack_bytes /
    pack_ws_bytes / reuse: per packed role; status: in a Phase "MET", "MISSED", "BUDGET" or
    "DIRECT", in configs() "OK" (within the budget) or "BUDGET"; feasible: the candidate's
    (status, budget and, in a Phase, accuracy); total_bytes, workspace_bytes,
    plan_device_bytes, err_bound, err_typical, err_status: the candidate's; tflops_eq: 2bmnk /
    time_s. time_lo_s / time_hi_s compose the band the same way (computed on first access, one
    prediction)."""
    __slots__ = ("name", "level", "candidate", "mode", "rounding", "status", "contract_s",
                 "pack_specs", "pack_s", "pack_bytes", "pack_ws_bytes", "reuse", "time_s",
                 "tflops_eq", "_ok", "_pp", "_band", "_beta", "_perf", "_query")

    def __init__(self, name, level, cand, mode, rounding, packs, reuse, flops_eq, query, beta,
                 perf):
        self.name, self.level, self.candidate = name, level, cand
        self.mode, self.rounding = mode, rounding
        self.status = None
        self.contract_s = cand.contract_s
        roles = _ROLES[cand.packed]
        self._pp = {r: packs[r] for r in roles}               # role -> (PackSpec, Prediction)
        self.pack_specs = {r: p[0] for r, p in self._pp.items()}
        self.pack_s = {r: p[1].time_s for r, p in self._pp.items()}
        self.pack_bytes = {r: p[1].pack_bytes for r, p in self._pp.items()}
        self.pack_ws_bytes = {r: p[1].pack_ws_bytes for r, p in self._pp.items()}
        self.reuse = {r: reuse[r] for r in roles}
        t = cand.contract_s
        for r in roles:
            t += self.pack_s[r] / self.reuse[r]
        self.time_s = t
        self.tflops_eq = flops_eq / t * 1e-12 if t > 0 else 0.0
        self._ok = cand.feasible        # what the choice rule ranks as feasible
        self._band = None
        self._query, self._beta, self._perf = query, beta, perf

    num_moduli = property(lambda s: s.candidate.num_moduli)
    residue_group = property(lambda s: s.candidate.residue_group)
    g = residue_group
    vec_mode = property(lambda s: s.candidate.vec_mode)
    packed = property(lambda s: s.candidate.packed)
    feasible = property(lambda s: s.candidate.feasible)
    total_bytes = property(lambda s: s.candidate.total_bytes)
    workspace_bytes = property(lambda s: s.candidate.workspace_bytes)
    plan_device_bytes = property(lambda s: s.candidate.plan_device_bytes)
    err_bound = property(lambda s: s.candidate.err_bound)
    err_typical = property(lambda s: s.candidate.err_typical)
    err_status = property(lambda s: s.candidate.err_status)

    def _key(self):
        """Rank's order after the composed time (NOTES-C1 §4.2): total_bytes, N, g, vec mode,
        packed."""
        c = self.candidate
        return (c.total_bytes, c.num_moduli, c.residue_group, _VEC_ORDER[c.vec_mode],
                _PACK_MODES.index(c.packed))

    def options(self):
        """The Plan / PlanCache keywords of this configuration (pass packs= for the packed
        roles: Pack objects of pack_specs)."""
        return dict(N=self.num_moduli, g=self.residue_group, rounding=self.rounding,
                    mode=self.mode, vec_mode=self.vec_mode)

    def _bands(self):
        if self._band is None:
            spec = (self.pack_specs.get("A"), self.pack_specs.get("B"))
            pr = self._query.predict(self.num_moduli, self.residue_group, self.vec_mode,
                                     rounding=self.rounding, mode=self.mode,
                                     packing=None if spec == (None, None) else spec,
                                     beta=self._beta, **self._perf)
            lo, hi = pr.time_lo_s, pr.time_hi_s
            for r, (_, p) in self._pp.items():
                lo += p.time_lo_s / self.reuse[r]
                hi += p.time_hi_s / self.reuse[r]
            self._band = (lo, hi)
        return self._band

    time_lo_s = property(lambda s: s._bands()[0])
    time_hi_s = property(lambda s: s._bands()[1])

    def __repr__(self):
        return (f"<Choice {self.name} level={self.level} g={self.residue_group} "
                f"{self.vec_mode} packed={self.packed} {self.status} "
                f"{self.tflops_eq:.2f} TFLOPS-eq {self.time_s * 1e3:.4f} ms>")


class Phase:
    """The configuration of every contraction for one phase (PrecisionPlan.phase): a view of
    its Choices. N[name] is the level (an int, or "DIRECT"); g, vec_mode, packed, status
    ("MET", "MISSED", "BUDGET", "DIRECT"), pack_bytes ({role: bytes} of the packed roles; a
    shared pack is repeated under every member that reads it, while footprints() counts it
    once, keyed by its group), reuse ({role: calls per pack}; a shared role's still counts a
    member at DIRECT in this phase, which reads no pack, so the other members' times are then
    optimistic [M], by at most pack_s x the DIRECT members' summed share of the calls per
    iteration: PrecisionPlan's `reuse`), assumed_reuse ({role: "iteration"} where
    'phase' fell back for want of `iterations`), err_bound / err_typical (in each target's
    metric) are dicts by name; time_s = the calls-weighted sum of the composed times (one
    iteration), tflops_eq = sum of calls x 2bmnk / time_s."""
    __slots__ = ("choices", "calls", "assumed", "_flops")

    def __init__(self, choices, calls, assumed, flops):
        self.choices, self.calls, self.assumed, self._flops = choices, calls, assumed, flops

    def _by(self, attr):
        return {n: getattr(c, attr) for n, c in self.choices.items()}

    N = property(lambda s: s._by("level"))
    g = property(lambda s: s._by("residue_group"))
    vec_mode = property(lambda s: s._by("vec_mode"))
    mode = property(lambda s: s._by("mode"))
    status = property(lambda s: s._by("status"))
    packed = property(lambda s: s._by("packed"))
    pack_bytes = property(lambda s: s._by("pack_bytes"))
    reuse = property(lambda s: s._by("reuse"))
    err_bound = property(lambda s: s._by("err_bound"))
    err_typical = property(lambda s: s._by("err_typical"))

    @property
    def assumed_reuse(self):
        return {n: {r: k for r, k in self.assumed.get(n, {}).items() if r in c.reuse}
                for n, c in self.choices.items()}

    @property
    def time_s(self):
        return sum(self.calls[n] * c.time_s for n, c in self.choices.items())

    @property
    def tflops_eq(self):
        t = self.time_s
        return sum(self.calls[n] * self._flops[n] for n in self.choices) / t * 1e-12 if t > 0 \
            else 0.0

    def __repr__(self):
        return (f"<Phase N={self.N} status={self.status} {self.tflops_eq:.2f} TFLOPS-eq "
                f"{self.time_s * 1e3:.4f} ms per iteration>")


@dataclasses.dataclass(frozen=True)
class Footprints:
    """Exact bytes for sizing once (DESIGN §6g.2 C-ws, C-pack) or per level, from the queries
    (never recomputed): the maximum over what a phase may choose at that level, i.e. the
    allowed pack modes x vec modes x every g that succeeds (no reliance on I2, a certificate
    and not ABI, for the plan device bytes). levels: the PrecisionPlan's. Per contraction
    name: workspace and plan_device_bytes {level: bytes, None where no configuration exists},
    workspace_max. Per packed operand (key (name, role), or the share group's name for a
    shared pack): pack_bytes and pack_ws_bytes (imd_pack's workspace) {level: bytes},
    pack_max, pack_ws_max. totals {level: bytes}: the sum of every contraction's workspace and
    plan device bytes and every packed operand's pack and pack workspace bytes (none shared:
    a joint budget is the caller's policy); total_max."""
    levels: tuple
    workspace: dict
    workspace_max: dict
    plan_device_bytes: dict
    pack_bytes: dict
    pack_max: dict
    pack_ws_bytes: dict
    pack_ws_max: dict
    totals: dict
    total_max: int


def _max(vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# ------------------------------------------------------------------------------------------
# PrecisionPlan

@dataclasses.dataclass
class _Contraction:
    name: str
    query: object          # planner.Query
    calls: int             # calls per iteration
    beta: bool             # C is read (rank / predict's beta flag)
    budget: int
    modes: tuple           # allowed pack modes, shared roles forced in
    shared: dict           # role -> share group name
    kinds: dict            # role -> reuse kind ('call' | 'iteration' | 'phase' | int)
    flops_eq: float        # 2 b m n k of the GEMM view


class PrecisionPlan:
    """Per-phase N for a set of contractions (DESIGN v1.19 §6g.10; NOTES-C1 §4.2).

    planner: a planner.Planner (its calibration and memo are used).
    contractions: {name: (spec, extents[, strides][, calls_per_iteration])} (an int third
        element is the calls; spec may also be raw descriptors with extents None), or {name:
        dict(spec=, extents=, strides=, calls=, beta=)}; beta=True: the calls read C.
    levels: increasing N in 2..20 (3-5 recommended); may end with "DIRECT" (native FP64 when no
        emulated level meets a target; not in the default: it runs at 1:64 on the 4090).
    rounding, mode: the plans' options. vec_modes: the vec modes ranked.
    share: [(group, members)], each member a name (equal N only) or (name, "A" | "B"): the
        role that reads the group's one packed operand. Members get one N, ranked jointly over
        every level >= the group floor (IR24); a shared role is always packed (the other roles
        follow `packs`), and its pack cost is divided by the calls of every member reading it.
        Groups that share a contraction are merged for N. A member at DIRECT in a phase reads
        no pack and leaves the group for that phase (the others keep one N).
    packs: {name: mode or set of modes} ("NONE", "A", "B", "AB"; default {"NONE"}): the
        allowed pack modes, ranked at every level, so the packing may differ by N (IR8).
    reuse: {name: {"A" | "B": "call" | "iteration" | "phase" | int}} per packed role (default
        "iteration": the pack is remade every iteration): calls per pack = 1, the calls per
        iteration, those x phase()'s `iterations` (without it: as "iteration", recorded in
        Phase.assumed_reuse), or the int. A shared role: the members' summed calls (an int:
        the total), the members' kinds must agree; fixed at construction, so it still counts
        a member at DIRECT in a phase (which reads no pack). The other members' pack term,
        and Phase.time_s, are then optimistic [M], by at most pack_s x the DIRECT members'
        summed share of the calls per iteration, and every phase's choice stays a configs()
        plan (NOTES-C1 §5.5 S1-10).
    budget: immerdink-owned device bytes per contraction (workspace + plan + packs + pack
        workspace), rank's meaning; an int for all or {name: int}; 0 = none. The caller
        subtracts its own resident tensors, the FP64 source of a packed operand included.
    monotone: True (default) for tightening iterations: a phase never lowers a contraction's
        N below its previous phase's final level until reset(); False for relaxed (inexact)
        Krylov methods, which loosen the matvec precision as the residual falls.
        Caller guidance at a switch (DESIGN §6g.12; NOTES-C1 §1.3, M-d1: CPU evidence on one
        fixture each, report-only): reset a DIIS history at each switch (no iteration lost,
        a lower final residual); for subspace methods (Davidson), collapse to the Ritz block
        and recompute its products at the new N (else the ramp stalls; even so it took +5
        iterations, so a ramp may not pay there); for Krylov methods (CG), start tight and
        relax (monotone=False): a ramp that starts low stalls, and a tolerance tol/||r_k||
        leaves a residual gap unless scaled by the expected iteration count, e.g.
        tol/(m_max ||r_k||).
    perf: {"regime", "cache", "clock_mhz"}: rank / predict_pack request fields (default: the
        library's DEFAULT regime, COLD).
    Thread-safe; phase() calls are serialized (they update the monotone floor)."""

    def __init__(self, planner, contractions, *, levels=(6, 8, 10, 12, 14), rounding="RN",
                 mode="EMULATE", share=(), packs=None, reuse=None, budget=0, monotone=True,
                 vec_modes=("GENERATE",), perf=None):
        if not isinstance(planner, _planner.Planner):
            raise TypeError("planner must be an immerdink.planner.Planner")
        self.planner = planner
        if rounding not in ("RN", "TRUNC") or mode not in ("EMULATE", "DIRECT"):
            raise ValueError("rounding RN | TRUNC, mode EMULATE | DIRECT")
        self.rounding, self.mode, self.monotone = rounding, mode, bool(monotone)
        self.levels = self._levels(levels)
        if DIRECT in self.levels and mode != "EMULATE":
            raise ValueError("a 'DIRECT' level needs mode EMULATE")
        self.emulated = tuple(x for x in self.levels if x != DIRECT)
        self.vec_modes = tuple(dict.fromkeys(vec_modes))
        if not self.vec_modes or any(v not in _VEC_ORDER for v in self.vec_modes):
            raise ValueError(f"vec_modes: a non-empty subset of {sorted(_VEC_ORDER)}")
        self.perf = dict(perf or {})
        bad = set(self.perf) - set(_PERF_FIELDS)
        if bad:
            raise ValueError(f"perf fields {sorted(bad)}: only {_PERF_FIELDS}")
        if not contractions:
            raise ValueError("no contractions")
        packs, reuse = dict(packs or {}), dict(reuse or {})
        budgets = budget if isinstance(budget, dict) else {}
        for what, d in (("packs", packs), ("reuse", reuse), ("budget", budgets)):
            unknown = set(d) - set(contractions)
            if unknown:
                raise ValueError(f"{what}: unknown contractions {sorted(unknown)}")
        self.share = self._share(share, contractions)
        self._c = {}
        for name, entry in contractions.items():
            q, calls, beta = self._entry(name, entry)
            shared = {r: g for g, members in self.share.items() for n, r in members
                      if n == name and r is not None}
            modes = self._modes(name, packs.get(name, "NONE"), shared)
            kinds = self._kinds(name, reuse.get(name, {}))
            b = budgets.get(name, 0) if isinstance(budget, dict) else budget
            if not isinstance(b, numbers.Integral) or b < 0:
                raise ValueError(f"{name}: budget must be an int >= 0")
            self._c[name] = _Contraction(name, q, calls, beta, int(b), modes, shared, kinds,
                                         self._flops(name, q))
        for g, members in self.share.items():     # the shared role's reuse kind is one
            ks = {repr(self._c[n].kinds[r]) for n, r in members if r is not None}
            if len(ks) > 1:
                raise ValueError(f"share group {g!r}: its members' reuse kinds differ ({ks})")
        self._lock = threading.Lock()
        self._packs = {}       # (name, role, N) -> (PackSpec, planner.Prediction)
        self._floor = {}       # name -> the previous phase's final level (monotone)

    # -- construction -----------------------------------------------------------------------

    @staticmethod
    def _levels(levels):
        lv = tuple(levels)
        emu = lv[:-1] if lv and lv[-1] == DIRECT else lv
        if not emu or any(not isinstance(x, numbers.Integral) or isinstance(x, bool) or
                          not 2 <= x <= 20 for x in emu) or \
                any(a >= b for a, b in zip(emu, emu[1:])):
            raise ValueError("levels: increasing N in 2..20, optionally ending with 'DIRECT'")
        return tuple(int(x) for x in emu) + lv[len(emu):]

    def _entry(self, name, e):
        if isinstance(e, dict):
            unknown = set(e) - {"spec", "extents", "strides", "calls", "beta"}
            if unknown or "spec" not in e:
                raise ValueError(f"{name}: dict(spec=, extents=, strides=, calls=, beta=)")
            spec, ext, st = e["spec"], e.get("extents"), e.get("strides")
            calls, beta = e.get("calls", 1), bool(e.get("beta", False))
        else:
            e = tuple(e)
            if not 2 <= len(e) <= 4:
                raise ValueError(f"{name}: (spec, extents[, strides][, calls_per_iteration])")
            spec, ext, rest = e[0], e[1], list(e[2:])
            calls = rest.pop() if rest and isinstance(rest[-1], numbers.Integral) else 1
            st = rest.pop() if rest else None
            if rest:
                raise ValueError(f"{name}: (spec, extents[, strides][, calls_per_iteration])")
            beta = False
        if not isinstance(calls, numbers.Integral) or isinstance(calls, bool) or calls < 1:
            raise ValueError(f"{name}: calls per iteration must be an int >= 1")
        return self.planner.query(spec, ext, strides=st), int(calls), beta

    @staticmethod
    def _share(share, contractions):
        out = {}
        for item in share:
            g, members = item
            if g in out:
                raise ValueError(f"share group {g!r} given twice")
            ms = []
            for m in members:
                n, r = (m, None) if isinstance(m, str) else tuple(m)
                if n not in contractions:
                    raise ValueError(f"share group {g!r}: unknown contraction {n!r}")
                if r not in (None, "A", "B"):
                    raise ValueError(f"share group {g!r}: role of {n!r} must be 'A' or 'B'")
                if any(n == x for x, _ in ms):
                    raise ValueError(f"share group {g!r}: {n!r} listed twice")
                ms.append((n, r))
            if not ms:
                raise ValueError(f"share group {g!r} is empty")
            out[g] = tuple(ms)
        seen = {}
        for g, ms in out.items():
            for n, r in ms:
                if r is not None and (n, r) in seen:
                    raise ValueError(f"{n!r} role {r} is in share groups {seen[(n, r)]!r} and "
                                     f"{g!r}: one role reads one pack")
                seen[(n, r)] = g
        return out

    @staticmethod
    def _modes(name, allowed, shared):
        allowed = (allowed,) if isinstance(allowed, str) else tuple(allowed)
        if not allowed or any(m not in _PACK_MODES for m in allowed):
            raise ValueError(f"{name}: pack modes must be a non-empty set of {_PACK_MODES}")
        forced = set(shared)
        modes = {_MODE[tuple(sorted(set(_ROLES[m]) | forced))] for m in allowed}
        return tuple(m for m in _PACK_MODES if m in modes)

    @staticmethod
    def _kinds(name, given):
        unknown = set(given) - {"A", "B"}
        if unknown:
            raise ValueError(f"{name}: reuse roles are 'A' and 'B'")
        out = {}
        for r in ("A", "B"):
            k = given.get(r, "iteration")
            if isinstance(k, bool) or not (k in _REUSE_KINDS or
                                           (isinstance(k, numbers.Integral) and k >= 1)):
                raise ValueError(f"{name} {r}: reuse is 'call', 'iteration', 'phase' or an "
                                 "int >= 1")
            out[r] = k if isinstance(k, str) else int(k)
        return out

    def _flops(self, name, q):
        """2 b m n k of the GEMM view, as the C library's flops_eq (the lowering's sizes). b,
        m, n, k do not depend on N, and a size past uint64 is closed upward in N (F1), so the
        lowest level lowers wherever any level does."""
        lw = q.lowering(self.emulated[0], 1, "GENERATE", rounding="TRUNC", mode=self.mode)
        if not lw.lowered:
            raise ImmerdinkError(STATUS.index(lw.status), f"{name}: imd_lowering")
        return 2.0 * float(lw.b) * float(lw.m) * float(lw.n) * float(lw.k)

    def _components(self, names):
        """The contractions of `names` that share groups tie to one N, as tuples in the
        construction order: a member outside `names` (a DIRECT one) links nothing."""
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for ms in self.share.values():
            ms = [n for n, _ in ms if n in parent]
            for n in ms[1:]:
                parent[find(n)] = find(ms[0])
        comps = {}
        for n in self._c:
            if n in parent:
                comps.setdefault(find(n), []).append(n)
        return [tuple(v) for v in comps.values()]

    # -- the composition (IR2) --------------------------------------------------------------

    def _reuse(self, iterations):
        """{name: {role: calls per pack}} and {name: {role: 'iteration'}} where 'phase' fell
        back."""
        out, assumed = {}, {}
        for name, c in self._c.items():
            r = out[name] = {}
            for role in ("A", "B"):
                kind = c.kinds[role]
                g = c.shared.get(role)
                calls = sum(self._c[n].calls for n, x in self.share[g] if x is not None) \
                    if g is not None else c.calls
                if kind == "call":
                    r[role] = 1
                elif kind == "iteration":
                    r[role] = calls
                elif kind == "phase":
                    if iterations is None:
                        r[role] = calls
                        assumed.setdefault(name, {})[role] = "iteration"
                    else:
                        r[role] = calls * iterations
                else:
                    r[role] = kind
        return out, assumed

    def _pack(self, c, role, N, g, vec):
        """(PackSpec, planner.Prediction) of `role` of contraction c at N, memoized per
        (contraction, role, N): imd_pack_desc_for does not depend on g or the vec mode."""
        key = (c.name, role, N)
        v = self._packs.get(key)
        if v is None:
            spec = c.query.pack_desc_for(_OPERAND[role], N, g, vec, rounding=self.rounding,
                                         mode=self.mode)
            v = self._packs[key] = (spec, c.query.predict_pack(spec, **self.perf))
        return v

    def _ranked(self, c, Ns, reuse, accuracy=None, mode=None, packs=None, gs=None,
                level=None, ignore_accuracy=False):
        """{N: [Choice]} of the SUCCESS candidates of one rank call (reuse 1, every candidate)
        at the N in Ns, re-keyed by the composed time; level: the Choices' level (None: N).
        ignore_accuracy: a target no level meets (MISSED) is ranked with its accuracy (the
        err fields in its metric) but chosen by status and budget only."""
        mode = mode or self.mode
        r = c.query.rank(tuple(Ns), gs, self.vec_modes, packs or c.modes, 1, c.budget, c.beta,
                         rounding=self.rounding, mode=mode, accuracy=accuracy, **self.perf)
        out = {N: [] for N in Ns}
        for cand in r:
            if cand.status != "SUCCESS":
                continue
            N, g, vec = cand.num_moduli, cand.residue_group, cand.vec_mode
            pk = {role: self._pack(c, role, N, g, vec) for role in _ROLES[cand.packed]}
            ch = Choice(c.name, N if level is None else level, cand, mode, self.rounding, pk,
                        reuse[c.name], c.flops_eq, c.query, c.beta, self.perf)
            if ignore_accuracy:
                ch._ok = not cand.infeasible & host.INFEASIBLE["BUDGET"]
            out[N].append(ch)
        return out

    @staticmethod
    def _best(choices):
        """The feasible choice with the least composed time (then rank's tie order), or
        None."""
        f = [x for x in choices if x._ok]
        return min(f, key=lambda x: (x.time_s,) + x._key()) if f else None

    @staticmethod
    def _least_bytes(choices):
        return min(choices, key=lambda x: (x.total_bytes, x.time_s) + x._key()[1:]) \
            if choices else None

    def _direct(self, c, reuse, accuracy, ignore_accuracy=False):
        """[Choice]: the native plans (mode DIRECT, in-line operands, g = 1)."""
        got = self._ranked(c, (self.emulated[-1],), reuse, accuracy, mode="DIRECT",
                           packs=("NONE",), gs=(1,), level=DIRECT,
                           ignore_accuracy=ignore_accuracy)
        return got[self.emulated[-1]]

    # -- public -----------------------------------------------------------------------------

    def configs(self, iterations=None):
        """{name: {level: Choice or None}}: the best configuration at every level (every g,
        the vec modes, the allowed pack modes, the composed reuse with `iterations` as in
        phase(), the budget; no accuracy: a fixed N's estimate is the same for every
        candidate). status "OK", or "BUDGET" (none fits: the least total_bytes); None: no
        plan exists at that level (e.g. RN with kappa >= T(N)). A phase's choice at a level is
        this one (for the same `iterations`), so creating these plans before the iteration
        loop covers every phase."""
        reuse, _ = self._reuse(iterations)
        out = {}
        for name, c in self._c.items():
            got = self._ranked(c, self.emulated, reuse)
            if DIRECT in self.levels:
                got[DIRECT] = self._direct(c, reuse, None)
            res = out[name] = {}
            for level in self.levels:
                ch = self._best(got[level])
                if ch is not None:
                    ch.status = "OK"
                else:
                    ch = self._least_bytes(got[level])
                    if ch is not None:
                        ch.status = "BUDGET"
                res[level] = ch
        return out

    def _at_least(self, a, b):
        return a if self.levels.index(a) >= self.levels.index(b) else b

    def _floor_of(self, c, acc):
        """(floor level, met): the smallest level whose estimate meets acc (NOTES-C1 §4.2
        steps 1-2); not met at any emulated level: DIRECT if listed and it meets, else the top
        emulated level with met False."""
        e = c.query.min_moduli(acc, rounding=self.rounding, mode=self.mode,
                               n_lo=self.emulated[0], n_hi=self.emulated[-1])
        if e.status == "SUCCESS":
            n = e.num_moduli
            return next(x for x in self.emulated if x >= n), True
        if not (e.status == "NOT_SUPPORTED" and e.err_status == "MISSED"):
            raise ImmerdinkError(STATUS.index(e.status), f"{c.name}: imd_err_min_moduli")
        if DIRECT in self.levels and self._direct_met(c, acc):
            return DIRECT, True
        return self.emulated[-1], False

    def _direct_met(self, c, acc):
        top = self.emulated[-1]
        e = c.query.min_moduli(acc, rounding=self.rounding, mode="DIRECT", n_lo=top, n_hi=top)
        if e.status == "SUCCESS":
            return True
        if not (e.status == "NOT_SUPPORTED" and e.err_status == "MISSED"):
            raise ImmerdinkError(STATUS.index(e.status), f"{c.name}: imd_err_min_moduli")
        return False

    def phase(self, targets, *, iterations=None, fail_budget=None):
        """The configuration of every contraction for the next phase (NOTES-C1 §4.2 rule):
          1. n_c = min_moduli(target, levels[0] .. top emulated level); none meets: DIRECT if
             listed and it meets (status DIRECT), else the top level (status MISSED);
          2. the floor f_c = the smallest level >= n_c;
          3. monotone: f_c = max(f_c, the previous phase's final level; DIRECT is the top);
          4. outside a share group: the best composed candidate over the levels >= f_c (one
             rank call with the accuracy, the pack-mode mask and the budget); between levels
             of equal composed time, the smaller N;
          5. a share group: a member at DIRECT reads no pack, so it leaves the group for the
             phase (settled alone, as outside a group); for the others, for each level N >=
             their max f_c, the calls-weighted sum of their best composed times; the least
             wins (ties: the smaller N), for every one of them;
          6. BUDGET: no level >= the floor fits: the floor level and its least-total_bytes
             SUCCESS candidate, status BUDGET if it meets the target (else MISSED); MISSED
             takes precedence over BUDGET, BUDGET over DIRECT;
          7. with `iterations` (the phase length) and `fail_budget` P, every TYPICAL target
             gets fail_prob P / (iterations x the calls per iteration of every contraction).
        targets: {name: host.Accuracy with a target > 0} for every contraction. Raises
        ImmerdinkError for a lowering error, ValueError for bad arguments."""
        if set(targets) != set(self._c):
            raise ValueError(f"targets must name every contraction: {sorted(self._c)}")
        if iterations is not None and (not isinstance(iterations, numbers.Integral) or
                                       isinstance(iterations, bool) or iterations < 1):
            raise ValueError("iterations must be an int >= 1")
        accs = {}
        for name, acc in targets.items():
            if not isinstance(acc, host.Accuracy) or not acc.target > 0:
                raise ValueError(f"{name}: a host.Accuracy with a target > 0")
            accs[name] = acc
        if fail_budget is not None:
            if iterations is None:
                raise ValueError("fail_budget needs iterations (p = P / (iterations x calls))")
            if not 0 < fail_budget < 1:
                raise ValueError("fail_budget must be in (0, 1)")
            p = fail_budget / (iterations * sum(c.calls for c in self._c.values()))
            accs = {n: dataclasses.replace(a, fail_prob=p) if a.basis == "TYPICAL" else a
                    for n, a in accs.items()}
        reuse, assumed = self._reuse(iterations)
        with self._lock:
            floors, met = {}, {}
            for name, c in self._c.items():
                f, ok = self._floor_of(c, accs[name])
                if self.monotone and name in self._floor:
                    f = self._at_least(f, self._floor[name])
                    if f == DIRECT and ok and self._floor[name] == DIRECT:
                        ok = self._direct_met(c, accs[name])
                floors[name], met[name] = f, ok
            choices = {}
            for name, c in self._c.items():    # DIRECT reads no pack: outside every group
                if floors[name] == DIRECT:
                    got = self._direct(c, reuse, accs[name], ignore_accuracy=not met[name])
                    choices[name] = self._settle(c, got, "DIRECT" if met[name] else "MISSED")
            for comp in self._components([n for n in self._c if floors[n] != DIRECT]):
                choices.update(self._choose(comp, floors, met, accs, reuse))
            choices = {n: choices[n] for n in self._c}
            if self.monotone:
                for name, ch in choices.items():
                    self._floor[name] = ch.level
        return Phase(choices, {n: c.calls for n, c in self._c.items()}, assumed,
                     {n: c.flops_eq for n, c in self._c.items()})

    def _choose(self, comp, floors, met, accs, reuse):
        """One N for the emulated members of a component: the least calls-weighted sum of
        their best composed times over the levels >= their maximum floor (ties: the smaller
        N); none fits every member: the floor."""
        floor = max(floors[n] for n in comp)
        Ns = tuple(x for x in self.emulated if x >= floor)
        got = {}
        for n in comp:
            got[n] = self._ranked(self._c[n], Ns, reuse, accs[n], ignore_accuracy=not met[n])
        best_N, best_t = None, None
        for N in Ns:
            bests = [self._best(got[n][N]) for n in comp]
            if any(b is None for b in bests):
                continue
            t = sum(self._c[n].calls * b.time_s for n, b in zip(comp, bests))
            if best_t is None or t < best_t:
                best_N, best_t = N, t
        N = best_N if best_N is not None else Ns[0]
        return {n: self._settle(self._c[n], got[n][N], "MET" if met[n] else "MISSED")
                for n in comp}

    def _settle(self, c, choices, status):
        """The best feasible choice with `status`; none: the least total_bytes, status BUDGET
        if it meets the target, else MISSED (MISSED takes precedence)."""
        ch = self._best(choices)
        if ch is None:
            ch = self._least_bytes(choices)
            if ch is None:
                raise ImmerdinkError(STATUS.index("NOT_SUPPORTED"),
                                     f"{c.name}: no plan at the chosen level")
            if status != "MISSED":
                status = "MISSED" if ch.candidate.infeasible & host.INFEASIBLE["ACCURACY"] \
                    else "BUDGET"
        ch.status = status
        return ch

    def reset(self):
        """Forget the monotone floor: a new solve, a new geometry, or a restart."""
        with self._lock:
            self._floor.clear()

    def footprints(self):
        """Footprints: exact bytes per level and their maxima (see Footprints)."""
        ws, pdev, packs, pws = {}, {}, {}, {}
        op_of = {}
        for name, c in self._c.items():
            for role in ("A", "B"):
                op_of[(name, role)] = c.shared.get(role, (name, role))
        for name, c in self._c.items():
            ws[name], pdev[name] = {}, {}
            for level in self.levels:
                if level == DIRECT:
                    cfgs = [(self.emulated[-1], vec, "NONE", "DIRECT") for vec in self.vec_modes]
                else:
                    cfgs = [(level, vec, m, self.mode) for m in c.modes for vec in self.vec_modes]
                w, d = [], []
                for cfg in cfgs:
                    for got in self._sizes(c, *cfg):
                        w.append(got[0])
                        d.append(got[1])
                        for role, lay in got[2].items():
                            op = op_of[(name, role)]
                            packs.setdefault(op, {}).setdefault(level, set()).add(
                                lay.pack_bytes)
                            pws.setdefault(op, {}).setdefault(level, set()).add(
                                lay.pack_ws_bytes)
                ws[name][level], pdev[name][level] = _max(w), _max(d)
        pb = {op: {lv: _max(v.get(lv, ())) for lv in self.levels} for op, v in packs.items()}
        pw = {op: {lv: _max(v.get(lv, ())) for lv in self.levels} for op, v in pws.items()}
        totals = {}
        for lv in self.levels:
            parts = [ws[n][lv] for n in ws] + [pdev[n][lv] for n in pdev]
            if any(x is None for x in parts):
                totals[lv] = None
                continue
            totals[lv] = sum(parts) + sum(v[lv] or 0 for v in pb.values()) + \
                sum(v[lv] or 0 for v in pw.values())
        return Footprints(self.levels, ws, {n: _max(v.values()) for n, v in ws.items()}, pdev,
                          pb, {op: _max(v.values()) for op, v in pb.items()}, pw,
                          {op: _max(v.values()) for op, v in pw.items()}, totals,
                          _max(totals.values()))

    def _sizes(self, c, N, vec, packed, mode):
        """[(workspace, plan device bytes, {role: PackLayout})] of every g that succeeds at
        (N, vec, packed, mode): footprints() takes the maximum over them, so it relies on
        neither F1's order in g nor I2 (a certificate, not ABI) for the plan device bytes."""
        q, roles, out = c.query, _ROLES[packed], []
        for g in range(1, N + 1):
            try:
                specs = {r: q.pack_desc_for(_OPERAND[r], N, g, vec, rounding=self.rounding,
                                            mode=mode) for r in roles}
                packing = (specs.get("A"), specs.get("B")) if roles else None
                w = q.workspace_size(N, g, vec, rounding=self.rounding, mode=mode,
                                     packing=packing)
            except ImmerdinkError:
                continue
            lw = q.lowering(N, g, vec, rounding=self.rounding, mode=mode, packing=packing)
            if lw.status != "SUCCESS":
                continue
            lays = {r: host.pack_layout(s, lib=self.planner.lib) for r, s in specs.items()}
            if any(l.status != "SUCCESS" for l in lays.values()):
                continue
            out.append((w, lw.plan_device_bytes, lays))
        return out


# ------------------------------------------------------------------------------------------
# The rigorous rho from a previous output (DESIGN v1.19 §6g.11; NOTES-C1 §4.2, IR7)

def _norm_bound(arg, x):
    """A norm bound as float64, converted exactly or refused: np.asarray(x, dtype=float64)
    rounds to nearest, which can fall below the caller's upper bound."""
    import numpy as np
    b = np.asarray(x)
    if b.dtype.kind == "f" and b.dtype.itemsize <= 8:      # float16/32/64: widen exactly
        return b.astype(np.float64, copy=False)
    if b.dtype.kind in "iu":
        if b.size and (b.max() > 2**53 or b.min() < -2**53):
            raise ValueError(f"{arg}: an integer bound above 2^53 in magnitude does not convert "
                             "to float64 exactly; pass it rounded up as a float")
        return b.astype(np.float64)
    raise TypeError(f"{arg} of dtype {b.dtype}: a norm bound is a float (at most 64-bit) or "
                    "integer array; converting it to float64 would round it to nearest, "
                    "possibly below the bound (round it up to float64 first)")


def rho_from_output(C, row_norm_ub, col_norm_ub, *, N, rounding, k, alpha=1.0, beta=0.0,
                    l1_ratio=None, mode="EMULATE", lib=None):
    """(rho_hat, certified): upper bounds rho_hat_ij >= rho_ij = ||a_i|| ||b_j|| / |x_ij| from
    the output C of a call at (N, rounding, k, mode) with alpha = 2^e and beta = 0, for an
    ELEMENTWISE target's `cond` (DESIGN §6f.1: the max over the outputs that must meet it).
    Rigorous for that call's data, a prediction for the next iteration's.

    C: a float64 host array (numpy; copy a device C or the part needed to the host first) in
    the GEMM view (..., m, n) (DESIGN §1.1): a tensor output whose modes interleave A's and
    B's free modes, e.g. abij of "kjbc,kiac->abij", is permuted to (batch modes, A's free
    modes, B's free modes) and reshaped first, its bounds in the same order; the helper pairs
    bounds with outputs by position only.
    row_norm_ub, col_norm_ub: upper bounds of the GEMM view's row 2-norms ||a_i|| and column
    2-norms ||b_j|| (§6f.1's obligation: really upper bounds, e.g. rounded up), as FP values:
    float arrays (float16 / float32 widen to float64 exactly) or integers of magnitude
    <= 2^53. A bound that float64 conversion would round to nearest, possibly below the
    caller's bound, is refused: an integer above 2^53 (ValueError); a wider float (long
    double), Fractions, Decimals, Python ints beyond 64 bits (object), complex or bool
    (TypeError). A Python list is converted by numpy before this check (an integer above 2^53
    mixed with floats is already rounded), so pass numpy float arrays. A row bound lies along
    axis -2 and a column bound along axis -1:
    - C of shape (m, n): a scalar; a 1-D row_norm_ub of length m and a 1-D col_norm_ub of
      length n, each on its own axis; a row bound (m, 1) and a column bound (1, n).
    - A batched C (..., m, n), C.ndim >= 3: a scalar, or a bound with exactly C.ndim dims, a
      row bound (..., m, 1) and a column bound (..., 1, n), each batch dim C's or 1 (shared
      over that batch axis), keepdims=True: np.linalg.norm(A, axis=-1, keepdims=True) per
      batch, (b, 1, 1) one value per batch, r[None, :, None] / c[None, None, :] one vector
      shared by every batch of a 3-D C. A 1-D or 2-D bound on a batched C is a ValueError:
      numpy's per-batch reductions without keepdims, (b,), (b, 1), (b, m) or (b, n), cannot
      be told apart from a row or column vector when b equals m or n, and trailing-axis
      broadcasting pairs them with C's rows or columns.
    Anything else is a ValueError too: a 1-D bound of another length (1 included), a
    transposed (1, m) row or (m, 1) column bound, an elementwise (m, n) one. For a 0-D / 1-D
    C the bounds broadcast elementwise.
    Real norms, not sigma-derived ones (a zero row's sigma bound is vacuous). l1_ratio: None
    (nu = sqrt k), one value for both, or (l1_ratio_a, l1_ratio_b), as host.Accuracy.
    U is the NORMWISE `bound` of imd_err_estimate at (N, rounding, k) for the call's dispatch
    (EMULATE; DIRECT when mode is DIRECT or k = 1). Per output, evaluated in FP64 exactly as
    spec/params.py rho_recipe (bit-identical; certified in exact rationals by
    cert_rho_recipe): D = ldexp(C, -e); n = nextafter(row x col, +inf);
    t = nextafter(U n, +inf); den = nextafter(|D| - t, 0); certified iff C and D are finite,
    D is normal (and |C| > 2^-1022 when e < 0: a C of exactly 2^-1022 may be a rounded
    subnormal), den > 0 and rho_hat = nextafter(n / den, +inf) is finite; uncertified
    entries are +inf. With true norm bounds rho_hat >= 1.
    ValueError: beta != 0 (for beta = 1 accumulations take rho from a probe call with
    beta = 0 into a scratch buffer, once per phase), alpha not a power of two (D cannot be
    recovered exactly from RN(alpha D)), a negative or NaN norm bound, an integer bound above
    2^53, a bound off its axis or of another ndim (above), shapes that do not broadcast to C.
    TypeError: C not float64, a norm bound of another dtype (above)."""
    import numpy as np
    if beta != 0:
        raise ValueError("rho_from_output needs beta = 0 (take rho from a beta = 0 probe call)")
    a = float(alpha)
    m, ex = math.frexp(a) if math.isfinite(a) else (0.0, 0)
    if not (a > 0 and m == 0.5):
        raise ValueError("rho_from_output needs alpha = 2^e (D is not recoverable otherwise)")
    e = ex - 1
    if mode not in ("EMULATE", "DIRECT") or rounding not in ("RN", "TRUNC"):
        raise ValueError("mode EMULATE | DIRECT, rounding RN | TRUNC")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 0:
        raise ValueError("k must be an int >= 0")
    if l1_ratio is None:
        la = lb = 0.0
    elif isinstance(l1_ratio, numbers.Real):
        la = lb = float(l1_ratio)
    else:
        la, lb = (float(x) for x in l1_ratio)
    C = np.asarray(C)
    if C.dtype != np.float64:
        raise TypeError(f"C must be float64, got {C.dtype}")
    row = _norm_bound("row_norm_ub", row_norm_ub)
    col = _norm_bound("col_norm_ub", col_norm_ub)
    if C.ndim >= 2:     # a row bound lies along axis -2, a column bound along axis -1
        lead = "None, " * (C.ndim - 2)
        for arg, b, own, along, what, form, shared in (
                ("row_norm_ub", row, -2, -1, "rows", "(..., m, 1)", f"r[{lead}:, None]"),
                ("col_norm_ub", col, -1, -2, "columns", "(..., 1, n)", f"c[{lead}None, :]")):
            if C.ndim >= 3 and b.ndim not in (0, C.ndim):
                raise ValueError(
                    f"{arg} of shape {b.shape} on a batched C {C.shape}: a bound is a scalar "
                    f"or has C's {C.ndim} dims, {form} with each batch dim C's or 1 "
                    f"(keepdims=True; {shared} for one shared by every batch); a 1-D or 2-D "
                    "bound cannot be told apart from a per-batch one where a batch extent "
                    "equals m or n")
            if b.ndim == 1 and b.shape[0] != C.shape[own]:
                raise ValueError(f"a 1-D {arg} has length {b.shape[0]}, C has "
                                 f"{C.shape[own]} {what} (axis {own})")
            if b.ndim >= 2 and b.shape[along] != 1:
                raise ValueError(f"{arg} of shape {b.shape} on C {C.shape}: a bound of the "
                                 f"{what} has shape[{along}] == 1, {form} (keepdims=True)")
        if row.ndim == 1:
            row = row[:, None]
    try:
        shape = np.broadcast_shapes(row.shape, col.shape, C.shape)
    except ValueError:
        shape = None
    if shape != C.shape:
        raise ValueError(f"norm bounds of shapes {row.shape}, {col.shape} do not broadcast "
                         f"to C's {C.shape}")
    if np.isnan(row).any() or np.isnan(col).any() or (row < 0).any() or (col < 0).any():
        raise ValueError("norm bounds must be >= 0 (+inf allowed)")
    dispatch = "SCALE" if k == 0 else "DIRECT" if (mode == "DIRECT" or k == 1) else "EMULATE"
    U = host.err_estimate(N, k, rounding=rounding, dispatch=dispatch, outputs=1,
                          accuracy=host.Accuracy("NORMWISE", l1_ratio_a=la, l1_ratio_b=lb),
                          lib=lib).bound
    with np.errstate(all="ignore"):
        D = np.ldexp(C, -e)
        aD = np.abs(D)
        ok = np.isfinite(C) & np.isfinite(D) & (aD >= _DBL_MIN)
        if e < 0:
            ok &= np.abs(C) > _DBL_MIN
        n = np.nextafter(row * col, np.inf)
        t = np.nextafter(U * n, np.inf)
        den = np.nextafter(aD - t, 0.0)
        ok &= den > 0
        rho = np.nextafter(n / den, np.inf)
        ok &= np.isfinite(rho)
    return np.where(ok, rho, np.inf), ok
