#!/usr/bin/env python3
"""A precision schedule from the residual: the forcing rule as caller code (NOTES-C1 §4.2,
DESIGN v1.19 §6g.10). An EXAMPLE of caller policy, not library code: immerdink never changes
N on its own (ehrenfest D8), and the residual -> target rule belongs to the caller.

    OMP_NUM_THREADS=1 tools/memcap.sh python3 bench/accuracy/schedule.py [--relaxed]
        [--eta 0.1] [--iterations 16] [--host-lib PATH]

CPU only: libimmerdink_host (the §6f bounds and the §6c performance model); no GPU, no data.
The residual history is ASSUMED (geometric, r_t = r_0 q^t), not computed: the example shows
how the targets, PrecisionPlan.phase and the monotone floor interact, and what the model
predicts for it. NOTES-C1 M-d1 measures real convergence (bench/accuracy/md1_converge.py)
and M-d2 the time on the GPU.

The rule (NOTES-C1 §4.2):
  - after iteration t, contraction c gets an ABSOLUTE target eta r_t w_c (sum w_c = 1), with
    norm_product_c an upper bound of max ||a_i|| ||b_j|| of its current operands (the
    caller's, refreshed per phase; here a constant);
  - N changes only at an iteration boundary (phase() is called there) and, with
    monotone=True, never falls until reset();
  - BOUND for the final phase, TYPICAL for the intermediate ones if the caller accepts a
    failure probability P over the run: phase() gives every call of a phase p = fail_budget /
    (iterations x calls per iteration), so phase(..., iterations=1, fail_budget=P / T_max) at
    every iteration is a union bound over the whole run (T_max: the iteration bound);
  - relaxed (inexact) Krylov methods (--relaxed): monotone=False and a matvec tolerance
    proportional to tol / ||r_k||, so N falls as the residual does (NOTES-C1 §1.3).

Caller guidance from M-d1 (DESIGN §6g.12, NOTES-C1 §1.3; CPU evidence on one fixture each,
report-only, not a C1 gate; immerdink-eval findings/c1-evidence.md):
  - fixed-point iterations (Jacobi, DIIS, PPL + ring): the forcing-rule ramps above match
    fixed N = 14 (no more iterations, the final residual within 2x). With DIIS, reset its
    history at each precision switch: no iteration lost, final residual 1.5e-16 against
    2.1e-15. The gain comes from restarting a DIIS that stalls: plain DIIS stalls near 1-2e-15
    with or without switches (the exact-arithmetic reference also ends at 2.09e-15);
  - subspace methods (block Davidson): at each switch, collapse to the Ritz block and
    recompute its sigma = HV at the new N, or the ramp stalls. Even so the ramp took 16
    iterations to 1e-10 against 11 at fixed N = 14 (+5), so it pays only if the per-iteration
    saving outweighs that;
  - Krylov methods (CG): start tight and relax. Ramps that start low stall CG at N = 5-6.
    The relaxed min(1e-2, tol / ||r_k||) converges but leaves a residual gap (7.1e-8 against
    3.3e-10 at fixed N = 14), so scale the relaxation by the expected iteration count, e.g.
    tol / (m_max ||r_k||) (not measured in M-d1).

Every time and TFLOPS-eq figure printed is a performance-model prediction [M] (shipped
calibration, DEFAULT regime, COLD). Levels outside N in {8, 10, 14, 16} are outside the
model's hold-out until NOTES-C1 M-c passes [M, model-only, outside hold-out].
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "python"))
from immerdink import host, planner, precision  # noqa: E402

# Two contractions of NOTES-C1 §2.3 / §2.4: the Jacobi view of bench/accuracy/iterate.py
# (y[ab,ij] = sum_cd W[ab,cd] x[cd,ij], v = 40, o = 10) and a DF build (acQ,bdQ->acbd) whose
# B (the bdQ integrals) is invariant, so it may be packed. phase() runs every iteration here,
# so its B is charged one pack per iteration (reuse 'iteration'); a caller that holds N for a
# known number of iterations passes reuse 'phase' and that number as `iterations`.
CONTRACTIONS = {
    "jacobi": ("ik,kj->ij", dict(i=1600, k=1600, j=100), 1),
    "df": ("acQ,bdQ->acbd", dict(a=64, b=64, c=64, d=64, Q=2032), 2),
}
NORM_PRODUCT = {"jacobi": 40.0, "df": 2.0e3}     # assumed max ||a_i|| ||b_j||
WEIGHTS = {"jacobi": 0.5, "df": 0.5}


def forcing_targets(residual, *, eta, norm_products=NORM_PRODUCT, weights=WEIGHTS,
                    basis="BOUND"):
    """{name: Accuracy}: ABSOLUTE target eta r w_c with the caller's norm_product_c."""
    return {c: host.Accuracy("ABSOLUTE", eta * residual * weights[c],
                             norm_product=norm_products[c], basis=basis)
            for c in norm_products}


def run(pp, targets_of, iterations, P=None):
    rows, total = [], 0.0
    for t in range(iterations):
        final = t == iterations - 1
        acc = targets_of(t, final)
        ph = pp.phase(acc, iterations=1,
                      fail_budget=None if final or P is None else P / iterations)
        rows.append((t, ph))
        total += ph.time_s
    return rows, total


def show(rows, total, flops, label):
    print(f"\n{label}")
    print(f"{'t':>3} {'N jacobi':>9} {'N df':>5} {'df packed':>9} {'status':>14} "
          f"{'TFLOPS-eq':>10} {'ms / iter':>10}")
    for t, ph in rows:
        print(f"{t:>3} {ph.N['jacobi']!s:>9} {ph.N['df']!s:>5} {ph.packed['df']:>9} "
              f"{'/'.join(sorted(set(ph.status.values()))):>14} {ph.tflops_eq:>10.2f} "
              f"{ph.time_s * 1e3:>10.4f}")
    print(f"run: {flops * len(rows) / total * 1e-12:.2f} TFLOPS-eq, {total * 1e3:.3f} ms [M]")


def gemm_size(spec, ext):
    """b m n k of the GEMM view: the product of every distinct label's extent (each label of
    these specs is in exactly two of A, B, C)."""
    out = 1
    for ch in dict.fromkeys(spec.replace(",", "").replace("->", "")):
        out *= ext[ch]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", default=None)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--r0", type=float, default=1.0, help="initial residual (assumed)")
    ap.add_argument("--q", type=float, default=0.25, help="residual ratio per iteration")
    ap.add_argument("--tol", type=float, default=1e-10, help="relaxed: the solve tolerance")
    ap.add_argument("--relaxed", action="store_true", help="relaxed Krylov: monotone=False")
    a = ap.parse_args()
    pl = planner.Planner(lib=host.load(a.host_lib))
    kw = dict(levels=(6, 8, 10, 12, 14, 16), packs={"df": {"NONE", "B"}},
              reuse={"df": {"B": "iteration"}})
    flops = sum(c * 2.0 * gemm_size(s, e) for s, e, c in CONTRACTIONS.values())
    r = [a.r0 * a.q**t for t in range(a.iterations)]
    if not a.relaxed:
        pp = precision.PrecisionPlan(pl, CONTRACTIONS, **kw)
        rows, total = run(pp, lambda t, final: forcing_targets(
            r[t], eta=a.eta, basis="BOUND" if final else "TYPICAL"), a.iterations, P=1e-3)
        show(rows, total, flops, f"tightening (monotone): eta {a.eta}, r_t = {a.r0} x "
                                 f"{a.q}^t [assumed], TYPICAL (P = 1e-3 per run) then BOUND")
    else:
        # the relative matvec error allowed at iteration t: eta tol / r_t (at most 1e-2)
        pp = precision.PrecisionPlan(pl, CONTRACTIONS, monotone=False, **kw)
        rows, total = run(pp, lambda t, final: {
            c: host.Accuracy("NORMWISE", min(a.eta * a.tol / r[t], 1e-2)) for c in CONTRACTIONS},
            a.iterations)
        show(rows, total, flops, f"relaxed (monotone=False): NORMWISE matvec target "
                                 f"eta tol / r_t, tol {a.tol}, r_t = {a.r0} x {a.q}^t [assumed]")
    n_fix = max(n for _, ph in rows for n in ph.N.values())
    fixed = precision.PrecisionPlan(pl, CONTRACTIONS, levels=(n_fix,), packs=kw["packs"],
                                    reuse=kw["reuse"])
    ph = fixed.phase({c: host.Accuracy("NORMWISE", 1.0) for c in CONTRACTIONS})
    print(f"fixed N = {n_fix} at every iteration: {ph.tflops_eq:.2f} TFLOPS-eq, "
          f"{ph.time_s * len(rows) * 1e3:.3f} ms [M]")


if __name__ == "__main__":
    main()
