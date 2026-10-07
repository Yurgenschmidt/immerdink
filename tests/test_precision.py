#!/usr/bin/env python3
"""python/immerdink/precision.py (DESIGN v1.19 §6g.10-§6g.11; NOTES-C1 §4.2, stage 1) against
a brute force from host.py's queries and against spec/params.py. CPU only.

    python3 tests/test_precision.py --host-lib build/libimmerdink_host.so [--configs 150] [-v]

  - PrecisionPlan.phase == a brute-force composition (NOTES-C1 IR2): every level, g, vec mode
    and allowed pack mode enumerated with host.pack_desc_for / predict / predict_pack /
    workspace_size / lowering / pack_layout, the targets checked with host.err_estimate at
    the lowering's (N, k, dispatch, b m n), each candidate re-keyed by contract_s +
    sum pack_s(role) / reuse(role), the rule of §4.2 applied from first principles (monotone
    floor, share groups ranked jointly over every level >= the group floor, BUDGET, MISSED,
    the DIRECT sentinel, TYPICAL p = P / C); the chosen configuration, its status and its
    composed time bit for bit, on random plans (levels, masks, reuse, budgets, vec modes,
    groups, 3-phase target sequences); each choice == the configs() lookup; the C rank's
    pack_s == the per-role predict_pack sum (the re-keying is exact); the composed band;
  - scenarios: per-role reuse ('call', 'iteration', 'phase' with and without iterations ->
    assumed_reuse, int), a shared role divided by the members' summed calls; a level above
    the floor chosen where the model predicts it faster (a shape found at run time), alone and
    in a share group; N monotone over phases until reset(), relaxed with monotone=False;
    DIRECT only when no emulated level meets and only if listed; a share-group member at
    DIRECT leaves the group for the phase (both member forms; the others keep one N; it links
    no groups); BUDGET (the floor and the least total_bytes) and MISSED's precedence, BUDGET
    never on a missed target; TYPICAL p = P / (iterations x calls); RN kappa >= T levels;
    argument errors;
  - footprints() == the maximum of the C queries over the allowed pack modes x vec modes x
    every g, per level, and their totals;
  - rho_from_output == spec/params.py rho_recipe bit for bit (random and boundary values,
    every alpha = 2^e class), U == imd_err_estimate's NORMWISE bound for the call's dispatch
    and >= params' exact bound (the upper end of its enclosure) at N = 2..20, both dispatches;
    a 1-D row bound is axis -2 and a 1-D column bound axis -1 for any C.ndim >= 2 (batched,
    m == n, mixed with 2-D bounds: == the explicit (m, 1) x (1, n) result), rho_hat >= rho in
    exact rationals on emulated outputs (params.emulate_element), the ValueError / TypeError
    cases (beta != 0, alpha != 2^e, bad norms, 1-D lengths, shapes, dtype).
"""

import argparse
import dataclasses
import math
import os
import random
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "spec"))
import numpy as np  # noqa: E402
import params as P  # noqa: E402
from immerdink import host, planner, precision  # noqa: E402
from immerdink.host import Accuracy  # noqa: E402

DIRECT = precision.DIRECT
MODES = ("NONE", "A", "B", "AB")
ROLES = {"NONE": (), "A": ("A",), "B": ("B",), "AB": ("A", "B")}
OP = {"A": 0, "B": 1}
VEC = {"GENERATE": 0, "RESIDENT": 1}
LIB = None
STATS = {"phases": 0, "choices": 0, "candidates": 0, "configs": 0, "footprints": 0, "rho": 0}
STATUSES = {}   # how often each status / feature occurred in the random suite


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


@dataclasses.dataclass(frozen=True)
class Case:
    spec: str
    ext: tuple            # sorted (label, extent) pairs
    calls: int = 1
    beta: bool = False

    @property
    def extents(self):
        return dict(self.ext)

    def entry(self):
        return dict(spec=self.spec, extents=self.extents, calls=self.calls, beta=self.beta)


def case(spec, calls=1, beta=False, **ext):
    return Case(spec, tuple(sorted(ext.items())), calls, beta)


CASES = {
    "kf": case("ik,kj->ij", i=64, k=300, j=48),
    "kl": case("lik,lkj->lij", l=3, i=33, k=1100, j=17, calls=2, beta=True),
    "ring": case("kjbc,kiac->abij", a=12, b=12, c=12, i=6, j=6, k=6, calls=3),
    "big": case("ik,kj->ij", i=4096, k=4096, j=4096),
    "df": case("acQ,bdQ->acbd", a=16, b=16, c=16, d=16, Q=508),
}


# ------------------------------------------------------------------------------------------
# The brute force (host.py only)

_cands = {}


def candidate(cs, N, g, vec, m, mode="EMULATE", rounding="RN"):
    """One configuration's figures from host.py (None if it has no plan): the contract's
    prediction, each packed role's pack descriptor and pack prediction, the footprint
    queries."""
    key = (cs, N, g, vec, m, mode, rounding)
    if key in _cands:
        return _cands[key]
    roles, ext = ROLES[m], cs.extents
    out = None
    try:
        specs = {r: host.pack_desc_for(cs.spec, ext, operand=OP[r], N=N, g=g, rounding=rounding,
                                       mode=mode, vec_mode=vec, lib=LIB) for r in roles}
    except host.ImmerdinkError:
        specs = None
    if specs is not None:
        packing = (specs.get("A"), specs.get("B")) if roles else None
        p = host.predict(cs.spec, ext, N=N, g=g, rounding=rounding, mode=mode, vec_mode=vec,
                         packing=packing, beta=cs.beta, lib=LIB)
        pps = {r: host.predict_pack(specs[r], lib=LIB) for r in roles}
        if p.status == "SUCCESS" and all(x.status == "SUCCESS" for x in pps.values()):
            ws = host.workspace_size(cs.spec, ext, N=N, g=g, rounding=rounding, mode=mode,
                                     vec_mode=vec, packing=packing, lib=LIB)
            lw = host.lowering(cs.spec, ext, N=N, g=g, rounding=rounding, mode=mode,
                               vec_mode=vec, packing=packing, lib=LIB)
            lays = {r: host.pack_layout(specs[r], lib=LIB) for r in roles}
            need(ws == p.workspace_bytes and lw.plan_device_bytes == p.plan_device_bytes,
                 f"{cs.spec} N={N} g={g} {vec} {m}: predict's footprint != the queries'")
            for r in roles:
                need(lays[r].pack_bytes == pps[r].pack_bytes and
                     lays[r].pack_ws_bytes == pps[r].pack_ws_bytes, "pack layout vs predict")
            total = ws + lw.plan_device_bytes + sum(x.pack_bytes for x in lays.values()) + \
                sum(x.pack_ws_bytes for x in lays.values())
            out = dict(N=N, g=g, vec=vec, packed=m, mode=mode, contract_s=p.time_s,
                       pack_s={r: pps[r].time_s for r in roles}, pp=pps, specs=specs,
                       total=total, ws=ws, dev=lw.plan_device_bytes,
                       pack_bytes={r: lays[r].pack_bytes for r in roles},
                       pack_ws={r: lays[r].pack_ws_bytes for r in roles})
            STATS["candidates"] += 1
    _cands[key] = out
    return out


_met = {}


def met_at(cs, N, acc, mode="EMULATE", rounding="RN"):
    """Whether acc's estimate (by its basis) meets its target at N: host.err_estimate at the
    lowering's (k, dispatch, b m n)."""
    key = (cs, N, acc, mode, rounding)
    if key not in _met:
        lw = host.lowering(cs.spec, cs.extents, N=N, g=1, rounding="TRUNC", mode=mode, lib=LIB)
        e = host.err_estimate(N, lw.k, rounding=rounding, dispatch=lw.dispatch,
                              outputs=lw.b * lw.m * lw.n, accuracy=acc, lib=LIB)
        rn_ok = rounding == "TRUNC" or mode == "DIRECT" or P.kappa(lw.k) < P.constants(N)["T"]
        _met[key] = e.err_status == "MET" and rn_ok
    return _met[key]


def compose(c, reuse):
    t = c["contract_s"]
    for r in ("A", "B"):
        if r in ROLES[c["packed"]]:
            t += c["pack_s"][r] / reuse[r]
    return t


def tie(c):
    return (c["total"], c["N"], c["g"], VEC[c["vec"]], MODES.index(c["packed"]))


def level_cands(cs, N, cfg, modes, mode="EMULATE", gs=None):
    out = []
    for g in (gs or range(1, N + 1)):
        for vec in cfg["vec_modes"]:
            for m in modes:
                c = candidate(cs, N, g, vec, m, mode)
                if c is not None:
                    out.append(c)
    return out


def best(cands, reuse, budget):
    f = [c for c in cands if budget == 0 or c["total"] <= budget]
    return min(f, key=lambda c: (compose(c, reuse),) + tie(c)) if f else None


def least_bytes(cands, reuse):
    return min(cands, key=lambda c: (c["total"], compose(c, reuse)) + tie(c)[1:]) \
        if cands else None


def expected_modes(cfg, name):
    forced = set(cfg["shared"].get(name, {}))
    allowed = cfg["packs"].get(name, {"NONE"})
    return {m for m in MODES
            if set(ROLES[m]) in [set(ROLES[a]) | forced for a in allowed]}


def expected_reuse(cfg, iterations):
    out, assumed = {}, {}
    for name in cfg["names"]:
        out[name] = {}
        for r in ("A", "B"):
            kind = cfg["reuse"].get(name, {}).get(r, "iteration")
            g = cfg["shared"].get(name, {}).get(r)
            calls = sum(CASES[n].calls for n, x in cfg["share"][g] if x is not None) \
                if g is not None else CASES[name].calls
            if kind == "call":
                v = 1
            elif kind == "iteration":
                v = calls
            elif kind == "phase":
                v = calls if iterations is None else calls * iterations
                if iterations is None:
                    assumed.setdefault(name, {})[r] = "iteration"
            else:
                v = kind
            out[name][r] = v
    return out, assumed


def components(cfg, names):
    """The contractions of `names` that share groups tie to one N (a DIRECT member, outside
    `names`, reads no pack and links nothing)."""
    comp = {n: {n} for n in names}
    for ms in cfg["share"].values():
        ms = [n for n, _ in ms if n in comp]
        merged = set().union(*(comp[n] for n in ms))
        for n in merged:
            comp[n] = merged
    seen, out = set(), []
    for n in cfg["names"]:
        if n in comp and n not in seen:
            out.append(sorted(comp[n], key=cfg["names"].index))
            seen |= comp[n]
    return out


def brute_phase(cfg, accs, prev, iterations):
    """{name: (level, status, candidate dict)} by the §4.2 rule from first principles."""
    levels = cfg["levels"]
    emu = [x for x in levels if x != DIRECT]
    reuse, _ = expected_reuse(cfg, iterations)
    floors, mets = {}, {}
    for name in cfg["names"]:
        cs, acc = CASES[name], accs[name]
        pf = prev.get(name) if cfg["monotone"] else None
        direct_met = DIRECT in levels and met_at(cs, emu[-1], acc, mode="DIRECT")
        if pf == DIRECT:
            floors[name], mets[name] = DIRECT, direct_met
            continue
        ok = [N for N in emu if (pf is None or N >= pf) and met_at(cs, N, acc)]
        if ok:
            floors[name], mets[name] = ok[0], True
        elif direct_met:
            floors[name], mets[name] = DIRECT, True
        else:
            floors[name], mets[name] = emu[-1], False
    out = {}
    for n in cfg["names"]:       # DIRECT: alone, in-line, g = 1, whatever its share groups
        if floors[n] != DIRECT:
            continue
        cands = level_cands(CASES[n], emu[-1], cfg, ("NONE",), mode="DIRECT", gs=(1,))
        b = best(cands, reuse[n], cfg["budget"].get(n, 0))
        st = "DIRECT" if mets[n] else "MISSED"
        if b is None:
            b = least_bytes(cands, reuse[n])
            if mets[n]:
                st = "BUDGET" if met_at(CASES[n], emu[-1], accs[n], mode="DIRECT") else "MISSED"
        out[n] = (DIRECT, st, b)
    for comp in components(cfg, [n for n in cfg["names"] if floors[n] != DIRECT]):
        floor = max(floors[n] for n in comp)
        Ns = [x for x in emu if x >= floor]
        per = {n: {N: level_cands(CASES[n], N, cfg, expected_modes(cfg, n)) for N in Ns}
               for n in comp}
        bestN, bestT = None, None
        for N in Ns:
            bs = [best(per[n][N], reuse[n], cfg["budget"].get(n, 0)) for n in comp]
            if any(b is None for b in bs):
                continue
            t = sum(CASES[n].calls * compose(b, reuse[n]) for n, b in zip(comp, bs))
            if bestT is None or t < bestT:
                bestN, bestT = N, t
        N = Ns[0] if bestN is None else bestN
        for n in comp:
            b = best(per[n][N], reuse[n], cfg["budget"].get(n, 0))
            st = "MET" if mets[n] else "MISSED"
            if b is None:
                b = least_bytes(per[n][N], reuse[n])
                if mets[n]:
                    st = "BUDGET" if met_at(CASES[n], N, accs[n]) else "MISSED"
            out[n] = (N, st, b)
    return out, reuse


# ------------------------------------------------------------------------------------------
# Random plans and targets

def random_target(rng):
    t = 10.0 ** -rng.uniform(2, 17)
    basis = rng.choice(("BOUND", "BOUND", "TYPICAL"))
    kw = {}
    if rng.random() < 0.2:
        kw = dict(l1_ratio_a=rng.uniform(1, 5), l1_ratio_b=rng.uniform(1, 30))
    metric = rng.choice(("NORMWISE", "ABSOLUTE", "ELEMENTWISE"))
    if metric == "NORMWISE":
        return Accuracy("NORMWISE", t, basis=basis, **kw)
    if metric == "ABSOLUTE":
        npd = 10.0 ** rng.uniform(-3, 3)
        return Accuracy("ABSOLUTE", t * npd, norm_product=npd, basis=basis, **kw)
    return Accuracy("ELEMENTWISE", t, cond=10.0 ** rng.uniform(0, 5), basis=basis, **kw)


def random_cfg(rng):
    names = rng.sample(sorted(CASES), rng.randint(1, 3))
    levels = sorted(rng.sample(range(3, 18), rng.randint(2, 5)))
    if rng.random() < 0.25:
        levels.append(DIRECT)
    cfg = dict(names=names, levels=tuple(levels), monotone=rng.random() < 0.7,
               vec_modes=rng.choice((("GENERATE",), ("RESIDENT",), ("GENERATE", "RESIDENT"))),
               packs={}, reuse={}, budget={}, share={}, shared={})
    for n in names:
        if rng.random() < 0.8:
            cfg["packs"][n] = set(rng.sample(MODES, rng.randint(1, 4)))
        cfg["reuse"][n] = {r: rng.choice(("call", "iteration", "phase", rng.randint(1, 40)))
                           for r in "AB" if rng.random() < 0.7}
        if rng.random() < 0.35:     # a budget between the smallest and largest footprints
            mid = [x for x in levels if x != DIRECT][len(levels) // 2 - 1]
            tot = [c["total"] for c in level_cands(CASES[n], mid, cfg, MODES)]
            if tot:
                cfg["budget"][n] = int(rng.uniform(min(tot) * 0.9, max(tot) * 1.1))
    if len(names) >= 2 and rng.random() < 0.5:
        members = [(n, rng.choice((None, "A", "B"))) for n in rng.sample(names, 2)]
        if all(r is None for _, r in members):
            members[0] = (members[0][0], "A")
        cfg["share"]["grp"] = tuple(members)
        kind = rng.choice(("call", "iteration", "phase", 7))
        for n, r in members:
            if r is not None:
                cfg["shared"].setdefault(n, {})[r] = "grp"
                cfg["reuse"].setdefault(n, {})[r] = kind
    return cfg


def make_plan(pl, cfg):
    return precision.PrecisionPlan(
        pl, {n: CASES[n].entry() for n in cfg["names"]}, levels=cfg["levels"],
        packs={n: m for n, m in cfg["packs"].items()}, reuse=cfg["reuse"],
        budget=dict(cfg["budget"]), share=[(g, list(ms)) for g, ms in cfg["share"].items()],
        monotone=cfg["monotone"], vec_modes=cfg["vec_modes"])


def check_choice(ch, want, reuse_r, what):
    level, st, c = want
    need(ch.level == level, f"{what}: level {ch.level} != {level}")
    need(ch.status == st, f"{what}: status {ch.status} != {st}")
    need((ch.num_moduli, ch.residue_group, ch.vec_mode, ch.packed) ==
         (c["N"], c["g"], c["vec"], c["packed"]),
         f"{what}: (N, g, vec, packed) {(ch.num_moduli, ch.residue_group, ch.vec_mode, ch.packed)}"
         f" != {(c['N'], c['g'], c['vec'], c['packed'])}")
    t = compose(c, reuse_r)
    need(ch.time_s == t, f"{what}: composed time {ch.time_s!r} != {t!r}")
    need(ch.contract_s == c["contract_s"], f"{what}: contract_s != host.predict's")
    need(ch.pack_s == c["pack_s"] and ch.pack_bytes == c["pack_bytes"] and
         ch.pack_ws_bytes == c["pack_ws"], f"{what}: per-role pack figures")
    need(ch.total_bytes == c["total"] and ch.workspace_bytes == c["ws"], f"{what}: footprint")
    need(ch.pack_specs == c["specs"], f"{what}: pack descriptors")
    need(ch.reuse == {r: reuse_r[r] for r in ROLES[c["packed"]]}, f"{what}: reuse")
    # the C rank's pack_s is the sum of the per-role predictions (0 + A + B): exact re-keying
    sp = 0.0
    for r in ("A", "B"):
        if r in c["pack_s"]:
            sp += c["pack_s"][r]
    need(ch.candidate.pack_s == sp, f"{what}: rank's pack_s != the per-role sum")
    STATS["choices"] += 1


def suite_random(pl, rng, count, verbose):
    t0 = time.time()
    for i in range(count):
        cfg = random_cfg(rng)
        pp = make_plan(pl, cfg)
        prev = {}
        for ph_i in range(3):
            iterations = rng.choice((None, 1, 5, 30))
            fb = rng.choice((None, 1e-3)) if iterations is not None else None
            accs = {n: random_target(rng) for n in cfg["names"]}
            adj = accs
            if fb is not None:
                p = fb / (iterations * sum(CASES[n].calls for n in cfg["names"]))
                adj = {n: dataclasses.replace(a, fail_prob=p) if a.basis == "TYPICAL" else a
                       for n, a in accs.items()}
            what = f"config {i} phase {ph_i} ({cfg['levels']}, monotone {cfg['monotone']})"
            want, reuse = brute_phase(cfg, adj, prev, iterations)
            ph = pp.phase(accs, iterations=iterations, fail_budget=fb)
            STATS["phases"] += 1
            for n in cfg["names"]:
                check_choice(ph.choices[n], want[n], reuse[n], f"{what} {n}")
                ch = ph.choices[n]
                STATUSES[ch.status] = STATUSES.get(ch.status, 0) + 1
                if ch.packed != "NONE":
                    STATUSES["packed"] = STATUSES.get("packed", 0) + 1
                if cfg["share"]:
                    STATUSES["shared"] = STATUSES.get("shared", 0) + 1
                # the err fields are the chosen configuration's under its target
                spec = (ch.pack_specs.get("A"), ch.pack_specs.get("B"))
                pr = host.predict(CASES[n].spec, CASES[n].extents, N=ch.num_moduli,
                                  g=ch.residue_group, vec_mode=ch.vec_mode, mode=ch.mode,
                                  packing=None if spec == (None, None) else spec,
                                  beta=CASES[n].beta, accuracy=adj[n], lib=LIB)
                need((pr.err_bound, pr.err_typical) == (ch.err_bound, ch.err_typical),
                     f"{what} {n}: err fields")
            need(ph.time_s == sum(CASES[n].calls * ph.choices[n].time_s for n in ph.choices),
                 f"{what}: Phase.time_s")
            # the same choice is configs()'s at that level
            cf = pp.configs(iterations)
            for n, ch in ph.choices.items():
                c2 = cf[n][ch.level]
                need(c2 is not None and (c2.residue_group, c2.vec_mode, c2.packed, c2.time_s) ==
                     (ch.residue_group, ch.vec_mode, ch.packed, ch.time_s),
                     f"{what} {n}: != configs()[{n}][{ch.level}]")
                need(c2.status == ("BUDGET" if ch.status == "BUDGET" or
                                   (ch.status == "MISSED" and not c2.feasible) else "OK"),
                     f"{what} {n}: configs status {c2.status} for a {ch.status} choice")
                STATS["configs"] += 1
            for grp, ms in cfg["share"].items():    # one N, but a DIRECT member leaves
                lv = {ph.N[m] for m, _ in ms if ph.N[m] != DIRECT}
                need(len(lv) <= 1, f"{what}: share group members at {lv}")
                if any(ph.N[m] == DIRECT for m, _ in ms):
                    STATUSES["shared DIRECT"] = STATUSES.get("shared DIRECT", 0) + 1
            if cfg["monotone"]:
                for n in cfg["names"]:
                    if n in prev:
                        need(cfg["levels"].index(ph.N[n]) >= cfg["levels"].index(prev[n]),
                             f"{what} {n}: N fell from {prev[n]} to {ph.N[n]}")
                prev = dict(ph.N)
    if verbose:
        print(f"  ok  random: {count} plans, {STATS['phases']} phases, {STATS['choices']} "
              f"choices == the brute force ({STATS['candidates']} host.py configurations), "
              f"== configs() ({time.time() - t0:.1f}s); {dict(sorted(STATUSES.items()))}",
              flush=True)
    for st in ("MET", "MISSED", "BUDGET", "DIRECT", "packed", "shared"):
        need(count < 20 or STATUSES.get(st, 0) > 0, f"random suite never produced {st}")


# ------------------------------------------------------------------------------------------
# Scenarios

def raises(exc, fn, what):
    try:
        fn()
    except exc:
        return
    raise Fail(f"{what}: no {exc.__name__}")


def suite_scenarios(pl, verbose):
    kf, big = CASES["kf"], CASES["big"]
    acc = lambda t: Accuracy("NORMWISE", t)   # noqa: E731
    # per-role reuse: one invariant packed B of a 4096^3 call
    for kind, its, want, assumed in (("call", None, 1, {}), ("iteration", None, 4, {}),
                                     ("phase", None, 4, {"B": "iteration"}),
                                     ("phase", 10, 40, {}), (25, None, 25, {})):
        pp = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", big.extents, 4)}, levels=(8, 14),
                                     packs={"m": {"B"}}, reuse={"m": {"B": kind}})
        ph = pp.phase({"m": acc(1e-6)}, iterations=its)
        need(ph.packed["m"] == "B" and ph.reuse["m"] == {"B": want},
             f"reuse {kind!r} iterations {its}: {ph.reuse['m']}, want {want}")
        need(ph.assumed_reuse["m"] == assumed, f"reuse {kind!r}: assumed {ph.assumed_reuse}")
        ch = ph.choices["m"]
        need(ch.time_s == ch.contract_s + ch.pack_s["B"] / want, f"reuse {kind!r}: composition")
        lo = host.predict("ik,kj->ij", big.extents, N=ch.num_moduli, g=ch.residue_group,
                          packing=(None, ch.pack_specs["B"]), lib=LIB)
        pb = host.predict_pack(ch.pack_specs["B"], lib=LIB)
        need(ch.time_lo_s == lo.time_lo_s + pb.time_lo_s / want and
             ch.time_hi_s == lo.time_hi_s + pb.time_hi_s / want, f"reuse {kind!r}: band")
    # a shared role: its cost divided by every reading member's calls
    ext = dict(i=2048, k=2048, j=2048, l=512)
    pp = precision.PrecisionPlan(
        pl, {"x": ("ik,kj->ij", ext, 2), "y": ("ik,kl->il", ext, 3), "z": ("ik,kj->ij", ext)},
        levels=(8, 10, 14), share=[("X", [("x", "A"), ("y", "A"), "z"])])
    ph = pp.phase({"x": acc(1e-6), "y": acc(1e-9), "z": acc(1e-3)})
    need(len(set(ph.N.values())) == 1, f"share group: N {ph.N}")
    need(ph.packed["x"] in ("A", "AB") and ph.packed["y"] in ("A", "AB"),
         f"share group: the shared role is packed: {ph.packed}")
    need(ph.reuse["x"]["A"] == 5 and ph.reuse["y"]["A"] == 5, f"shared reuse {ph.reuse}")
    floor = max(min(lv for lv in (8, 10, 14) if lv >= host.min_moduli(
        s, ext, accuracy=acc(t), n_lo=8, n_hi=14, lib=LIB).num_moduli)
        for s, t in (("ik,kj->ij", 1e-6), ("ik,kl->il", 1e-9), ("ik,kj->ij", 1e-3)))
    need(ph.N["x"] >= floor, f"group floor {floor}: {ph.N}")
    raises(ValueError, lambda: precision.PrecisionPlan(
        pl, {"x": ("ik,kj->ij", ext), "y": ("ik,kl->il", ext)},
        share=[("X", [("x", "A"), ("y", "A")])], reuse={"x": {"A": "call"}}),
        "share group with differing reuse kinds")
    # a level above the floor wins where the model's time is not monotone in N (NOTES-C1
    # §4.2 step 4 / IR24): found at run time on a few shapes, so a recalibration that moves
    # the pair does not void the scenario
    vecs = ("GENERATE", "RESIDENT")
    pair = None
    for m, n, k in ((200, 2048, 300), (200, 2048, 32), (512, 512, 32), (16, 2048, 32),
                    (512, 100, 32), (200, 512, 32)):
        ext = dict(i=m, k=k, j=n)
        t = {}
        for N in range(3, 21):
            r = host.rank("ik,kj->ij", ext, N=N, vec_modes=vecs, lib=LIB)
            if r and r[0].feasible:
                t[N] = r[0].time_s
        pair = next(((ext, N) for N in range(3, 20) if N in t and N + 1 in t and
                     t[N + 1] < t[N]), None)
        if pair:
            break
    need(pair is not None, "no shape with a faster higher N: the joint-search scenario needs one")
    ext, N = pair
    N_up = N
    pp = precision.PrecisionPlan(pl, {"p": ("ik,kj->ij", ext), "q": ("ik,kj->ij", ext, 2)},
                                 levels=(N, N + 1), vec_modes=vecs)
    ph = pp.phase({"p": acc(0.5), "q": acc(0.5)})
    need(ph.N == {"p": N + 1, "q": N + 1} and set(ph.status.values()) == {"MET"},
         f"faster higher level {N + 1} not chosen: {ph.N}")
    pp = precision.PrecisionPlan(pl, {"p": ("ik,kj->ij", ext), "q": ("ik,kj->ij", ext, 2)},
                                 levels=(N, N + 1), vec_modes=vecs, share=[("G", ["p", "q"])])
    need(pp.phase({"p": acc(0.5), "q": acc(0.5)}).N == {"p": N + 1, "q": N + 1},
         "share group: the faster higher level not chosen")
    # monotone floor, reset, relaxed
    pp = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", kf.extents)}, levels=(4, 6, 8, 10, 12, 14))
    seq = [pp.phase({"m": acc(t)}).N["m"] for t in (1e-3, 1e-7, 1e-12, 1e-4)]
    need(seq == sorted(seq) and seq[-1] == seq[-2] and seq[0] < seq[2],
         f"monotone: {seq}")
    pp.reset()
    need(pp.phase({"m": acc(1e-4)}).N["m"] < seq[-1], "reset() kept the floor")
    pr = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", kf.extents)}, levels=(4, 6, 8, 10, 12, 14),
                                 monotone=False)
    seq2 = [pr.phase({"m": acc(t)}).N["m"] for t in (1e-12, 1e-7, 1e-3)]
    need(seq2 == sorted(seq2, reverse=True) and seq2[0] > seq2[-1], f"relaxed: {seq2}")
    # DIRECT only if listed and only when no emulated level meets
    n4 = host.min_moduli("ik,kj->ij", kf.extents, accuracy=acc(1e-4), n_lo=4, n_hi=8,
                         lib=LIB).num_moduli
    met4 = min(lv for lv in (4, 6, 8) if lv >= n4)
    for levels, t, want, st in (((4, 6, 8), 1e-13, 8, "MISSED"),
                                ((4, 6, 8, DIRECT), 1e-13, DIRECT, "DIRECT"),
                                ((4, 6, 8, DIRECT), 1e-4, met4, "MET"),
                                ((4, 6, 8, DIRECT), 1e-17, 8, "MISSED")):
        pp = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", kf.extents)}, levels=levels)
        ph = pp.phase({"m": acc(t)})
        need((ph.N["m"], ph.status["m"]) == (want, st),
             f"levels {levels} target {t}: {ph.N['m']} {ph.status['m']}, want {want} {st}")
        if want == DIRECT:
            need(ph.mode["m"] == "DIRECT" and ph.packed["m"] == "NONE", "DIRECT choice")
            need(pp.phase({"m": acc(1e-3)}).N["m"] == DIRECT, "DIRECT is the monotone top")
    # a share-group member at DIRECT reads no pack, so it leaves the group for the phase and
    # does not drag the others along: y's TYPICAL target is met at N = 12 and missed by
    # DIRECT's gamma_k, x needs DIRECT, z (loose) stays coupled to y only through a group
    # without x
    sq, lv3 = ("ik,kj->ij", dict(i=64, k=4096, j=64)), (8, 10, 12)
    g_k = host.err_estimate(12, 4096, dispatch="DIRECT", outputs=64 * 64,
                            accuracy=Accuracy("NORMWISE"), lib=LIB).bound
    e12 = host.err_estimate(12, 4096, outputs=64 * 64,
                            accuracy=Accuracy("NORMWISE", basis="TYPICAL"), lib=LIB)
    need(e12.typical < g_k < e12.bound, "DIRECT-member scenario: gamma_k not inside N = 12's "
         "(typical, bound)")
    tg = {"x": acc((g_k + e12.bound) / 2), "z": acc(1e-3),
          "y": Accuracy("NORMWISE", (e12.typical + g_k) / 2, basis="TYPICAL")}
    zf = min(x for x in lv3 if x >= host.min_moduli(*sq, accuracy=tg["z"], n_lo=8, n_hi=12,
                                                    lib=LIB).num_moduli)
    for share, coupled in (([("G", [("x", "B"), ("y", "B")])], False),
                           ([("G", ["x", "y"])], False), ([("G", ["x", "y", "z"])], True),
                           ([("G", [("x", "A"), ("y", "A")]), ("H", [("x", "B"), ("z", "B")])],
                            False)):
        pp = precision.PrecisionPlan(pl, {"x": sq, "y": sq, "z": sq}, levels=lv3 + (DIRECT,),
                                     share=share)
        cf = pp.configs()
        ph = pp.phase(tg)
        what = f"DIRECT member of {share}"
        need((ph.N["x"], ph.status["x"], ph.mode["x"]) == (DIRECT, "DIRECT", "DIRECT"),
             f"{what}: x {ph.N['x']} {ph.status['x']}")
        need((ph.N["y"], ph.status["y"]) == (12, "MET") and ph.choices["y"].feasible and
             ph.err_typical["y"] <= tg["y"].target, f"{what}: y {ph.N['y']} {ph.status['y']}")
        zt = {N: cf["z"][N].time_s for N in lv3 if N >= zf}
        zbest = min(zt, key=lambda N: (zt[N], N))
        need(ph.N["z"] == (12 if coupled else zbest) and ph.status["z"] == "MET",
             f"{what}: z at {ph.N['z']}, want {12 if coupled else zbest}")
        if not coupled:
            need(zbest != 12, f"{what}: z's best level is 12, the scenario cannot tell")
        for n in "yz":      # the choice is configs()'s; a shared role keeps its reuse
            c2 = cf[n][ph.N[n]]
            need((c2.residue_group, c2.vec_mode, c2.packed, c2.time_s, c2.reuse) ==
                 (ph.choices[n].residue_group, ph.choices[n].vec_mode, ph.packed[n],
                  ph.choices[n].time_s, ph.reuse[n]), f"{what}: {n} != configs()")
        if share[0][1][0] == ("x", "B"):
            need(ph.packed["y"] == "B" and ph.reuse["y"] == {"B": 2},
                 f"{what}: the shared role's reuse {ph.reuse['y']}")
    # DIRECT with a budget nothing fits: BUDGET where DIRECT meets the target, MISSED where it
    # does not (the monotone top); and _settle's guard on its own: a least-bytes fallback
    # whose candidate misses the target is never BUDGET (the rule makes this unreachable)
    pp = precision.PrecisionPlan(pl, {"x": sq}, levels=lv3 + (DIRECT,), budget=1)
    ph = pp.phase({"x": tg["x"]})
    need((ph.N["x"], ph.status["x"]) == (DIRECT, "BUDGET"), f"DIRECT BUDGET: {ph.status}")
    ph = pp.phase({"x": tg["y"]})
    need((ph.N["x"], ph.status["x"]) == (DIRECT, "MISSED"), f"DIRECT MISSED: {ph.status}")
    reuse, _ = pp._reuse(None)
    got = pp._ranked(pp._c["x"], (12,), reuse, tg["x"])[12]
    need(got and all(ch.candidate.infeasible & host.INFEASIBLE["ACCURACY"] for ch in got),
         "guard: N = 12 should miss x's target")
    need(pp._settle(pp._c["x"], got, "MET").status == "MISSED", "guard: BUDGET on a miss")
    # BUDGET: the floor level and its least total_bytes; MISSED takes precedence
    pp = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", big.extents)}, levels=(6, 8, 10),
                                 packs={"m": set(MODES)}, budget=1 << 20)
    ph = pp.phase({"m": acc(1e-6)})
    n6 = host.min_moduli("ik,kj->ij", big.extents, accuracy=acc(1e-6), n_lo=6, n_hi=10,
                         lib=LIB).num_moduli
    lv = min(x for x in (6, 8, 10) if x >= n6)
    cands = [c for g in range(1, lv + 1) for m in MODES
             for c in [candidate(big, lv, g, "GENERATE", m)] if c is not None]
    lb = min(c["total"] for c in cands)
    need((ph.N["m"], ph.status["m"]) == (lv, "BUDGET") and ph.choices["m"].total_bytes == lb,
         f"BUDGET: {ph.N} {ph.status} {ph.choices['m'].total_bytes} vs least {lb} at {lv}")
    ph = pp.phase({"m": acc(1e-30)})
    need((ph.N["m"], ph.status["m"]) == (10, "MISSED"), f"MISSED over BUDGET: {ph.status}")
    # TYPICAL p = P / (iterations x calls)
    pp = precision.PrecisionPlan(pl, {"a": ("ik,kj->ij", kf.extents, 3),
                                      "b": ("ik,kj->ij", big.extents, 2)},
                                 levels=tuple(range(4, 21)))
    ta = Accuracy("NORMWISE", 1e-11, basis="TYPICAL")
    ph = pp.phase({"a": ta, "b": Accuracy("NORMWISE", 1e-11)}, iterations=8, fail_budget=1e-2)
    pa = dataclasses.replace(ta, fail_prob=1e-2 / (8 * 5))
    want = host.min_moduli("ik,kj->ij", kf.extents, accuracy=pa, n_lo=4, lib=LIB).num_moduli
    need(ph.N["a"] == want, f"TYPICAL p: N {ph.N['a']}, want {want}")
    e = host.err_estimate(want, 300, outputs=64 * 48, accuracy=pa, lib=LIB)
    need(ph.err_typical["a"] == e.typical, "TYPICAL p: err_typical at P / C")
    e0 = host.err_estimate(want, 300, outputs=64 * 48, accuracy=ta, lib=LIB)
    need(e0.typical != e.typical, "TYPICAL p: P / C had no effect")
    ph = pp.phase({"a": ta, "b": Accuracy("NORMWISE", 1e-11)})     # default p: unchanged
    need(ph.err_typical["a"] == host.err_estimate(ph.N["a"], 300, outputs=64 * 48,
                                                  accuracy=ta, lib=LIB).typical,
         "TYPICAL without fail_budget: the Accuracy's own fail_prob")
    # RN kappa >= T: no plan at N = 2 for k = 40000 (rn_feasible_kmax(2) = 32041)
    ext = dict(i=8, k=40000, j=8)
    pp = precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", ext)}, levels=(2, 3, 4))
    need(pp.configs()["m"][2] is None and pp.configs()["m"][3] is not None, "kappa >= T configs")
    need(pp.phase({"m": acc(0.5)}).N["m"] == 3, "kappa >= T level skipped")
    need(pp.footprints().workspace["m"][2] is None, "kappa >= T footprint")
    # argument errors
    one = {"m": ("ik,kj->ij", kf.extents)}
    for kw in (dict(levels=(8, 6)), dict(levels=(1, 8)), dict(levels=(8, DIRECT, 10)),
               dict(levels=()), dict(packs={"m": "X"}), dict(packs={"q": "A"}),
               dict(reuse={"m": {"A": "sometimes"}}), dict(reuse={"m": {"C": 1}}),
               dict(reuse={"m": {"A": 0}}), dict(budget=-1), dict(vec_modes=("FAST",)),
               dict(perf={"order": 1}), dict(share=[("g", ["m", "m"])]),
               dict(share=[("g", [("m", "C")])]), dict(rounding="RZ"),
               dict(mode="DIRECT", levels=(8, DIRECT))):
        raises(ValueError, lambda: precision.PrecisionPlan(pl, one, **kw), f"PrecisionPlan {kw}")
    raises(TypeError, lambda: precision.PrecisionPlan(host, one), "not a Planner")
    raises(ValueError, lambda: precision.PrecisionPlan(pl, {"m": ("ik,kj->ij", kf.extents, 0)}),
           "calls 0")
    pp = precision.PrecisionPlan(pl, one)
    raises(ValueError, lambda: pp.phase({}), "phase without targets")
    raises(ValueError, lambda: pp.phase({"m": Accuracy()}), "phase with target 0")
    raises(ValueError, lambda: pp.phase({"m": acc(1e-6)}, fail_budget=1e-3), "P without iterations")
    raises(ValueError, lambda: pp.phase({"m": acc(1e-6)}, iterations=0), "iterations 0")
    raises(host.ImmerdinkError, lambda: precision.PrecisionPlan(
        pl, {"m": ("ik,kj->ii", dict(i=4, k=4, j=4))}).phase({"m": acc(1e-6)}),
        "a lowering error (repeated C label)")
    if verbose:
        print(f"  ok  scenarios: per-role reuse and its band, shared role, a faster higher "
              f"level (N = {N_up} -> {N_up + 1}, alone and in a share group), monotone / reset / "
              f"relaxed, DIRECT sentinel, a DIRECT member leaves its share group, BUDGET and "
              f"MISSED, TYPICAL p = P/C, kappa >= T, argument errors", flush=True)


# ------------------------------------------------------------------------------------------
# footprints

def suite_footprints(pl, rng, count, verbose):
    for i in range(count):
        cfg = random_cfg(rng)
        pp = make_plan(pl, cfg)
        fp = pp.footprints()
        emu = [x for x in cfg["levels"] if x != DIRECT]
        packs = {}
        for n in cfg["names"]:
            for lv in cfg["levels"]:
                if lv == DIRECT:
                    cs = level_cands(CASES[n], emu[-1], cfg, ("NONE",), mode="DIRECT")
                else:
                    cs = level_cands(CASES[n], lv, cfg, expected_modes(cfg, n))
                w = max((c["ws"] for c in cs), default=None)
                d = max((c["dev"] for c in cs), default=None)
                need(fp.workspace[n][lv] == w and fp.plan_device_bytes[n][lv] == d,
                     f"footprints {i} {n} level {lv}: {fp.workspace[n][lv]} != {w}")
                for c in cs:
                    for r in ROLES[c["packed"]]:
                        op = cfg["shared"].get(n, {}).get(r, (n, r))
                        s = packs.setdefault(op, {}).setdefault(lv, [set(), set()])
                        s[0].add(c["pack_bytes"][r])
                        s[1].add(c["pack_ws"][r])
            need(fp.workspace_max[n] == max((v for v in fp.workspace[n].values()
                                             if v is not None), default=None), "workspace_max")
        need(set(fp.pack_bytes) == set(packs), f"footprints {i}: packed operands")
        for op, by in packs.items():
            for lv in cfg["levels"]:
                b = by.get(lv)
                need(fp.pack_bytes[op][lv] == (max(b[0]) if b else None) and
                     fp.pack_ws_bytes[op][lv] == (max(b[1]) if b else None),
                     f"footprints {i} {op} level {lv}")
        for lv in cfg["levels"]:
            parts = [fp.workspace[n][lv] for n in cfg["names"]] + \
                [fp.plan_device_bytes[n][lv] for n in cfg["names"]]
            want = None if any(x is None for x in parts) else sum(parts) + sum(
                (v[lv] or 0) for v in fp.pack_bytes.values()) + sum(
                (v[lv] or 0) for v in fp.pack_ws_bytes.values())
            need(fp.totals[lv] == want, f"footprints {i} total at {lv}")
        STATS["footprints"] += 1
    if verbose:
        print(f"  ok  footprints: {count} plans == the max of the C queries over the allowed "
              "pack modes x vec modes x every g", flush=True)


# ------------------------------------------------------------------------------------------
# rho_from_output

def U_of(N, k, rnd, dispatch, la=0.0, lb=0.0):
    return host.err_estimate(N, k, rounding=rnd, dispatch=dispatch,
                             accuracy=Accuracy("NORMWISE", l1_ratio_a=la, l1_ratio_b=lb),
                             lib=LIB).bound


def check_against_params(C, row, col, U, e, rho, ok, what):
    m, n = C.shape
    for i in range(m):
        for j in range(n):
            ref = P.rho_recipe(float(C[i, j]), float(row[i]), float(col[j]), U, e)
            need(bool(ok[i, j]) == (ref is not None),
                 f"{what} ({i},{j}): C={C[i, j]!r} certified {bool(ok[i, j])}, params "
                 f"{ref is not None}")
            if ref is not None:
                need(float(rho[i, j]).hex() == ref.hex(), f"{what} ({i},{j}): rho bits")
            else:
                need(rho[i, j] == math.inf, f"{what} ({i},{j}): uncertified rho not +inf")
            STATS["rho"] += 1


def suite_rho(rng, verbose):
    specials = [0.0, -0.0, 5e-324, -2.0**-1030, 2.0**-1022, -2.0**-1022,
                math.nextafter(2.0**-1022, 1.0), math.inf, -math.inf, math.nan, 2.0**1023]
    for trial in range(60):
        N, rnd = rng.randint(2, 20), rng.choice(("RN", "TRUNC"))
        k = rng.choice((0, 1, 2, 7, 300, 5000))
        mode = rng.choice(("EMULATE", "EMULATE", "DIRECT"))
        e = rng.choice((0, 0, 3, -4, 60, -60, 900, -900))
        l1 = rng.choice((None, 1.5, (2.0, 7.0)))
        m, n = rng.randint(1, 7), rng.randint(1, 7)
        dispatch = "SCALE" if k == 0 else "DIRECT" if mode == "DIRECT" or k == 1 else "EMULATE"
        la, lb = (0.0, 0.0) if l1 is None else (l1, l1) if isinstance(l1, float) else l1
        U = U_of(N, k, rnd, dispatch, la, lb)
        row = np.array([rng.choice((0.0, math.inf, 1e-300, 1e300)) if rng.random() < 0.1
                        else math.ldexp(1 + rng.random(), rng.randint(-40, 40))
                        for _ in range(m)])
        col = np.array([math.ldexp(1 + rng.random(), rng.randint(-40, 40)) for _ in range(n)])
        C = np.empty((m, n))
        for i in range(m):
            for j in range(n):
                r = rng.random()
                if r < 0.15:
                    C[i, j] = rng.choice(specials)
                elif r < 0.55 and math.isfinite(U):   # near U n (scaled by 2^e)
                    nn = math.nextafter(float(row[i]) * float(col[j]), math.inf)
                    base = U * nn
                    if 0 < base < 1e300:
                        D = base
                        for _ in range(rng.randint(-3, 6) + 3):
                            D = math.nextafter(D, math.inf)
                        for _ in range(3):
                            D = math.nextafter(D, 0.0)
                        try:
                            C[i, j] = float(Fraction(D) * Fraction(2)**e) * rng.choice((1, -1))
                        except OverflowError:
                            C[i, j] = math.inf
                    else:
                        C[i, j] = 1.0
                else:
                    C[i, j] = math.ldexp(rng.choice((1, -1)) * (1 + rng.random()),
                                         rng.randint(-60, 60) + e)
        with np.errstate(all="ignore"):
            rho, ok = precision.rho_from_output(C, row, col, N=N, rounding=rnd, k=k,
                                                alpha=math.ldexp(1.0, e), l1_ratio=l1,
                                                mode=mode, lib=LIB)
        need(rho.shape == C.shape and ok.shape == C.shape, "rho shapes")
        check_against_params(C, row, col, U, e, rho, ok,
                             f"rho trial {trial} (N={N} {rnd} k={k} {mode} e={e})")
    # broadcasting forms: (m, 1) x (1, n), a batch (b, m, n)
    C = np.array([[3.0, -2.0], [1e-3, 5.0]])
    r1, o1 = precision.rho_from_output(C, [1.0, 2.0], [3.0, 4.0], N=8, rounding="RN", k=10,
                                       lib=LIB)
    r2, o2 = precision.rho_from_output(C, np.array([[1.0], [2.0]]), np.array([[3.0, 4.0]]),
                                       N=8, rounding="RN", k=10, lib=LIB)
    need((r1 == r2).all() and (o1 == o2).all(), "1-D and broadcast norm forms differ")
    Cb = np.stack([C, -C])
    rb, ob = precision.rho_from_output(Cb, np.ones((2, 2, 1)), np.ones((1, 1, 2)) * 4.0, N=8,
                                       rounding="RN", k=10, lib=LIB)
    need(rb.shape == (2, 2, 2) and ob.all(), "batched C")
    # a 1-D row bound is always axis -2 and a 1-D column bound axis -1, each on its own, for
    # any C.ndim >= 2, also where m == n (numpy alone would pair row[j] with column j)
    row, col = np.array([1.0, 1.0, 100.0]), np.array([1.0, 2.0, 3.0])
    for C in (np.ones((3, 3)), np.ones((2, 3, 3)), np.ones((2, 1, 3, 3)),
              np.arange(1.0, 25.0).reshape(2, 3, 4)):
        m, n = C.shape[-2:]
        r, c = np.resize(row, m), np.resize(col, n)
        ref, rok = precision.rho_from_output(C, r[:, None], c[None, :], N=14, rounding="RN",
                                             k=10, lib=LIB)
        need(rok.all() and (ref >= np.multiply.outer(r, c) / np.abs(C)).all(),
             f"explicit shapes on C {C.shape}: rho_hat < row_i col_j / |C_ij|")
        for rr, cc in ((r, c), (r, c[None, :]), (r[:, None], c)):
            got, ok = precision.rho_from_output(C, rr, cc, N=14, rounding="RN", k=10, lib=LIB)
            need(got.shape == C.shape and (got == ref).all() and (ok == rok).all(),
                 f"1-D norms on C {C.shape} (row {np.shape(rr)}, col {np.shape(cc)}) != the "
                 "explicit (m, 1) x (1, n) result")
        for rr, cc in ((np.resize(r, m + 1), c), (r, np.resize(c, n + 1)), (r[:1], c),
                       (r, c[:1])):     # length 1 too: numpy would broadcast it silently
            raises(ValueError, lambda: precision.rho_from_output(C, rr, cc, N=14, rounding="RN",
                                                                 k=10, lib=LIB),
                   f"1-D norm of the wrong length on C {C.shape}")
    # rho_hat >= rho in exact rationals on emulated outputs (params.emulate_element)
    cert = 0
    for trial in range(10):
        N, rnd, k = rng.choice((3, 5, 8)), rng.choice(("RN", "TRUNC")), rng.choice((6, 25))
        e = rng.choice((0, 4, -3))
        A = [[math.ldexp(rng.uniform(-1, 1), rng.randint(-4, 4)) for _ in range(k)]
             for _ in range(3)]
        B = [[math.ldexp(rng.uniform(-1, 1), rng.randint(-4, 4)) for _ in range(3)]
             for _ in range(k)]
        for j in range(3):        # one cancelling column per trial: |x| near U ||a|| ||b||
            if j == 2:
                part = sum(Fraction(A[0][h]) * Fraction(B[h][j]) for h in range(k - 1))
                B[k - 1][j] = float(-part / Fraction(A[0][k - 1])) * (1 + 1e-3 * rng.random())
        D = np.empty((3, 3))
        X = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                col_j = [B[h][j] for h in range(k)]
                D[i, j] = P.emulate_element(A[i], col_j, N, rnd, chunk=7)["D"]
                X[i][j] = sum(Fraction(p) * Fraction(q) for p, q in zip(A[i], col_j))
        na2 = [sum(Fraction(v)**2 for v in A[i]) for i in range(3)]
        nb2 = [sum(Fraction(B[h][j])**2 for h in range(k)) for j in range(3)]
        up = lambda q: math.nextafter(float(P.sqrt_enclosure(q)[1]), math.inf)  # noqa: E731
        Cs = np.array([[float(Fraction(D[i, j]) * Fraction(2)**e) for j in range(3)]
                       for i in range(3)])
        rho, ok = precision.rho_from_output(Cs, [up(q) for q in na2], [up(q) for q in nb2],
                                            N=N, rounding=rnd, k=k, alpha=2.0**e, lib=LIB)
        for i in range(3):
            for j in range(3):
                if ok[i, j]:
                    cert += 1
                    need(X[i][j] != 0 and Fraction(float(rho[i, j]))**2 * X[i][j]**2 >=
                         na2[i] * nb2[j], f"emulated rho_hat < rho (N={N} {rnd} k={k})")
    need(cert > 0, "no emulated output certified")
    # U is the NORMWISE bound of the call's dispatch, never below params' exact bound (the
    # upper end of its enclosure)
    nU = 0
    for N in range(2, 21):
        for k in (1, 2, 7, 300, 5000, 40000):
            for rnd in ("RN", "TRUNC"):
                for dispatch in ("EMULATE", "DIRECT"):
                    U = U_of(N, k, rnd, dispatch)
                    ref = P.error_estimate(N, k, rnd, dispatch)["bound"]
                    need(U == math.inf if ref is None else Fraction(U) >= ref[1],
                         f"U below the exact bound (N={N} k={k} {rnd} {dispatch})")
                    nU += 1
    # argument errors
    C = np.ones((2, 2))
    kw = dict(N=8, rounding="RN", k=10, lib=LIB)
    for bad in (dict(beta=1.0), dict(beta=0.5), dict(alpha=3.0), dict(alpha=-1.0),
                dict(alpha=0.0), dict(alpha=math.inf), dict(alpha=math.nan),
                dict(mode="FAST"), dict(rounding="RZ"), dict(k=-1)):
        raises(ValueError, lambda: precision.rho_from_output(C, [1.0, 1.0], [1.0, 1.0],
                                                             **{**kw, **bad}), f"rho {bad}")
    for row, col in (([1.0, -1.0], [1.0, 1.0]), ([1.0, math.nan], [1.0, 1.0]),
                     ([1.0, 1.0, 1.0], [1.0, 1.0])):
        raises(ValueError, lambda: precision.rho_from_output(C, row, col, **kw),
               f"rho norms {row} {col}")
    raises(TypeError, lambda: precision.rho_from_output(C.astype(np.float32), [1.0, 1.0],
                                                        [1.0, 1.0], **kw), "rho float32 C")
    if verbose:
        print(f"  ok  rho_from_output: {STATS['rho']} outputs == params.rho_recipe bit for "
              f"bit, {cert} emulated outputs with rho_hat >= rho exactly, U >= params' bound "
              f"at {nU} points, 1-D norms on batched and mixed forms, argument errors",
              flush=True)


def main():
    global LIB
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", default=None)
    ap.add_argument("--configs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    LIB = host.load(a.host_lib)
    need(LIB.version >= 13, f"host API {LIB.version} < 13")
    rng = random.Random(a.seed)
    pl = planner.Planner(lib=LIB)
    t0 = time.time()
    try:
        suite_scenarios(pl, a.verbose)
        suite_random(pl, rng, a.configs, a.verbose)
        suite_footprints(pl, rng, max(a.configs // 2, 10), a.verbose)
        suite_rho(rng, a.verbose)
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  {STATS['phases']} phases ({STATS['choices']} choices) == the brute force and "
          f"configs(), {STATS['footprints']} footprints, {STATS['rho']} rho outputs == params, "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
