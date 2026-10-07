# NOTES-C1 — Adaptive precision (ROADMAP C1) and partner-aware scaling (C2): design draft for DESIGN v1.19

## Status

**Revised after two independent reviews, 2026-10-07** (branch
`claude/ozaki-kernels-immerdink-1089fd`). This is a design phase: nothing has been built, no
test or TLA+ model has been run, and no GPU work has been done.

The draft merges four design proposals and three earlier reviews:
- **P1**: Ozaki-II N-range plans, K5 statistics, K-equilibration.
- **P2**: Ozaki-I slice mode.
- **P3**: a never-worse partner-envelope accurate mode.
- **P4**: precision phases on today's API, plus K-segmentation.
- **Earlier reviews**: numerical soundness; performance payoff; API, integration and
  maintenance.

It was then reviewed by two independent reviewers. Both returned **approve-with-fixes**:
- **R-num**: numerics, proofs and spec;
- **R-api**: API, pipeline, performance and integration (adversarial, read-only).

Every MUST FIX of both is applied below. Each design question's resolution (decision, reviewer,
rationale) is in §5, and so are the review decisions outside the questions (§5.2, R1–R9). Where
the reviewers disagreed, §5 decides on the merits and says so. Claims found wrong in any review
are listed in Appendix A, and nothing below relies on them.

**To report to Eric** (neither item blocks stages 0–3):
- **Q6** changes the "Your decision" column of ROADMAP C1 to: C1 = (a) per-call N + host-side
  (b); (c) slices parked; "TLA+ for slices" deferred to the slice trigger (§3.3).
- **Q7** reframes C2: accurate mode is re-rejected for the corrected reason (§2.8), and C2 is
  delivered as K-equilibration. ROADMAP's C2 row is retitled accordingly.

Both ROADMAP edits land with the DESIGN v1.19 merge, unless Eric overrides them.

**Decision summary.** §3 gives the reasons, §4 the stages.

| item | decision |
|---|---|
| C1(a) per-call N | **Plan per N + pack per N** on host API 13, with published footprint monotonicity (totals and write extent only), an internal layout certificate, and a re-pack-in-place protocol (stage 0). A single plan serving a range of N is **parked**: it gains nothing per call. |
| C1(b) auto-N | **Host-side**, using the existing `imd_err_min_moduli` / rank, plus a thin `PrecisionPlan` in its own Python module and a rigorous ρ-from-output recipe (stage 1). Device statistics and an a-posteriori certificate (lemma L6) come **later, gated** on a consumer with elementwise targets (stage 5). Choosing N on the device is rejected. |
| C1(c) Ozaki-I slices | **Parked with triggers**; the numerics ideas are recorded (§3.3). |
| C2 | **K-equilibration**: power-of-two weights on the contracted index (lemma L5). First as caller-side pre-scaling with an `imd_kweights` helper (stage 4). The helper has a mandatory FP64-exact clamp and an exact integer guard that is never worse in the §3.8 bound (L5c). It reports the gain G, which converts the plain operands' statistics into a rigorous request for the weighted call (L5d). Library-internal weights and weighted packs come later (stage 6). Both stages are gated on a CPU study at realistic sizes and on real operands (stage 2). **K-segmentation** becomes a documented caller recipe with lemma L7 (stage 3). **Accurate mode stays rejected**, for the corrected reason in §2.8; the never-worse partner-envelope rule is recorded in Appendix B. |

**Tags.**
- [V]: verified in code (file:line) or in the docs.
- [C]: computed on the CPU in this design phase. This is either exact integer emulation, bit-exact
  against `spec/oracle.py`, or the host model library's own output.
- [M]: performance-model prediction, not a measurement. Shipped v1.18 calibration, host API 13,
  DEFAULT (= SUSTAINED), COLD, GENERATE, g = N unless stated.
- [E]: an estimate built from [M] numbers and stated assumptions.
- [I]: inference, not checked.

**Provenance.**
- Design-phase scratch directory: `/home/ef/.claude/jobs/ca28e7d2/tmp/c1/`.
  - Scripts: emulator `emu.py`; scaling variants `exp_acc.py`, `exp_hyb.py`, `exp_lowN.py`;
    K-weight study `c2gate.py`, `c2gate2.py`; K-segments `c2cmp.py`; envelope rule `c2bcs/`;
    slices `slice_eval/`; the review's size check `judge/keq_m.py`.
  - Records: `acc.jsonl`, `lowN.jsonl`, `hyb.jsonl`, `run_all.out`, `run2.out`, `run3.out`,
    `perf_summary.md`, `c2cost.md`, `predict_cc.out`.
  - Second-round review scripts: R-api's host-model spot checks `review_api/spot*.py`; R-num's
    exact L5c/L6 checks `review/chk_lemmas.py` (and `review/chk_perf.py`).
- Every run checked first that ehrenfest was FREE, then used one thread under
  `tools/memcap.sh` and took at most a minute. R-num's [M] re-runs were not completed, because
  ehrenfest became BUSY; R-num checked their arithmetic for internal consistency instead.
- Stage 2 (§4.3) moves the study into `bench/accuracy/` and the raw records into immerdink-eval
  `results/c1/`.
- [M] numbers come from `.claude/worktrees/agent-ac829dae3600d2888/build/libimmerdink_host.so`
  (cbdf230, host API 13). Its `src/host`, `include` and `python` match this worktree's
  (`diff -rq`). This worktree's own `build/libimmerdink_host.so` is stale: host API 5, with no
  `imd_perf_*` or `imd_err_*` symbols (R-num confirmed this with `nm`). **Rebuild it before
  stage 2**, so that every stage-2 record names this worktree's library.
- Tag suffixes: "[C, review]", "[M, review]" and "[E, review]" mark numbers a second-round
  reviewer computed or estimated, under the same rules. A tag without a suffix is this draft's
  own.
- Performance numbers for external libraries stay in immerdink-eval. Algorithms and accuracy
  facts from papers are cited by tag: [OZ2], [SC25] and [ERR] as in DESIGN, and [KT26] =
  Kawakami–Takahashi, arXiv:2606.29129 (the scale-invariant fast mode).

---

## 1. Problem, goals, what exists

### 1.1 The items

- **C1 adaptive precision** (ROADMAP phase C, size L; Eric's decision column: "which
  variant(s); TLA+ for slices"). The options are (a) per-call N, (b) auto-N from the M5
  elementwise rule, and (c) an Ozaki-I slice mode. The target is iterative methods that need full
  precision only near convergence; N = 8 is about 1.6× faster than N = 14.
- **C2 accurate-mode scaling** (size M, after C1): more bits per modulus, so a smaller N at equal
  accuracy. DESIGN §7 rejects accurate mode for v1 because it "needs the other operand (Ā·B̄),
  which breaks operand packing. Revisit for inputs with very wide dynamic range".

**Goals.**
- **G1.** Run each contraction at the smallest N its current target needs, switching N between
  calls. Keep footprints exact, results bit-exact against `oracle.contract(..., N)`, calls
  capture-safe, and add no data movement.
- **G2.** Map a target to N on the hot path, and supply the statistics an elementwise target
  needs (ρ needs |x|).
- **G3.** Need fewer moduli at equal accuracy where the fast rule wastes its budget, while keeping
  D1 packing, exact integer rules and the bit-exact oracle. The realistic case is anti-correlated
  magnitudes along K (ACCURACY §5.3): `decay_opp` needs N = 20 for 1e−10 elementwise where
  uniform data need 13, and N = 14 gives only 6e−4.

**Non-goals.**
- A precision policy inside immerdink. Residual → target belongs to the caller; ehrenfest
  FOUNDATIONS D8 makes precision user-directed, and "the optimizer never lowers precision on
  its own".
- Choosing N on the device.
- Graphs or streams owned by immerdink.
- CC-specific features.

### 1.2 What exists [V]

1. **N is a plan option, 2..20.** It is `imd_options_t.num_moduli`, checked by `options_ok`
   (src/host/analysis.cpp:17-23). `imd_options_t` is frozen at 24 B and has no `struct_size`
   (`static_assert`, src/host/host_api.cpp:64), so any new per-plan knob needs a new struct or
   entry point.
2. **σ depends on N only through the plan constant R·2^F.**
   - R = (2T(N) − κ)² for RN and T(N)² for TRUNC. `set_shift_constants` sets it
     (analysis.cpp:279-289), and K1/KF/KL get it by value (`ScaleArgs.rhs`,
     src/contract.cu:73-81).
   - (S, γ) do not depend on N (`oracle.scale_vector`, spec/oracle.py:361).
   - σ(N+1) − σ(N) is usually 3 or 4, depending on the data [C, numerics reader]. Near RN
     acceptance it can be 5: the (2T − κ) factor makes the step up to about 4.94 bits, and
     `oracle.scale_vector` gives σ(3) − σ(2) = 5 at k = 32041 and k = 31685 (κ = 179,
     T(2) = 180) [C, review]. A6's conclusion (§3.1) does not change.
3. **So a pack serves exactly one N.**
   - Key word 1 is N. Word 2 is the rounding, stored as a boolean
     (`rounding == IMD_TRUNC ? 1 : 0`; `pack_key`, src/host/pack.cpp:78-89, line 80).
   - Keys are compared for exact equality (`key_mismatch`, pack.cpp:103-125).
   - A plan with a different N gets `PACK_MISMATCH_MODULI` (analysis.cpp:175).
   - DESIGN §6b.10 #15 states this and reserves an `IMD_PACK_FORMAT` value for a slice format.
4. **The kernels already select by N on every call.**
   - The K5 instantiation comes from `p.N` (`launch_k5_for`, contract.cu:660-669).
   - The pack emitter's word count is `pack_words(p.N)` (contract.cu:974, 1222, 1465).
   - KF and KL use `gsize = min(g, N)` (contract.cu:948, 1200), and the group loop runs over N
     (contract.cu:1437-1438).
   - The K4 epilogue looks up its modulus by global plane index, so N reaches K4 only through
     the shape keys.
5. **Footprints.**
   - Workspace regions, in order: sA sB fA fB vec RA RB W, with RA = g·b·m·k̂, RB = g·b·n̂·k̂ and
     W = N·b·m·n̂ (src/host/workspace.cpp:49-56). Every other region is independent of N, and W
     is last.
   - Pack regions: sig, flag, res, with res = N·b·r·k̂ + (⌈r⌉₁₆ − r)·k̂ (pack.cpp:35-57). The
     sig and flag offsets and `pack_ws_bytes` (pack.cpp:59) do not depend on N.
   - A plan's device bytes are only its RESIDENT vectors, 8(2m + 2n + 2k + 3b)
     (include/immerdink/immerdink.h:40-44).
   - These are facts of the current formulas. §4.1 publishes only the totals and the write
     extent as guarantees; the offset and device-byte facts stay internal certificates.
6. **Python.** `PlanCache` keys on (N, g, rounding, mode, vec_mode, k4_tune)
   (python/immerdink/device.py:678-691), and `freeze()` turns a cache miss into an error (for
   capture).
7. **§6f, accuracy-aware planning.**
   - `imd_err_min_moduli` (src/host/accuracy.cpp:197-247) does one lowering, then evaluates the
     closed form for each N. ELEMENTWISE takes `cond` = ρ (the M5 rule); ABSOLUTE takes
     `norm_product`.
   - RN with κ ≥ T gives +∞, so N that are too small are skipped. Rank marks configurations that
     miss the target as infeasible.
   - `imd_err_request_t.reserved` must be 0, "reserved for flags" (accuracy.cpp:100).
   - `err_request_valid` only checks struct_size ≥ v8 (accuracy.cpp:99), so an API-13 library
     **silently ignores** any field appended to the struct.
   - `imd_err_estimate` takes `dispatch`, not `mode` (accuracy.cpp:178).
8. **Performance model.**
   - Rank enumerates N × g × vec mode × packing. It copies `*base` and overrides only N, g and
     vec_mode (src/host/perf.cpp:1724), so `mode` and `rounding` pass through.
   - Capacity used: 13 of 16 phases, 12 of 16 classes, 90 of 128 kernel rows.
   - The hold-out covers N = 8 / 14 plus 10 / 16 only (DESIGN §6c.5), so **no N ≤ 7**.
9. **RN acceptance.** κ ≥ T(N) gives NOT_SUPPORTED at plan time (analysis.cpp:322) and at pack
   time (pack.cpp:213). N = 2 needs k ≤ 32041; N = 3 needs k ≤ 8.2e6 (params `rn_feasible_kmax`).
10. **K5's per-output code** holds |z| as L32(N) limbs plus its sign after `crt_finish`
    (contract.cu:515), just before `round_scaled` (contract.cu:516). That is where any
    per-output statistic goes.

### 1.3 Consumer profile

**Generic iterative methods** (Jacobi/DIIS fixed points, Krylov, Davidson, iterative refinement)
tighten precision as the residual falls. That means a few monotone phases, switched at iteration
boundaries where the caller already crosses to the host. ACCURACY §3 shows a contracting
iteration does not accumulate emulation error: drift is set in the first few iterations, and the
steady state is about one contraction's error/(1 − ρ). So the last phase's N sets the converged
accuracy.

**ehrenfest (DF-CCSD) is context, not a requirement** [consumer reader]:
- It has no immerdink integration yet; its Ozaki work has been frozen since 2026-09-06.
- Its accepted operating point is about 5.7e−10 per GEMM, i.e. immerdink N 8–9.
- It takes about 16 iterations to reach tol 1e−7, and it re-plans per phase (D13 tier 3).
- Dressed integrals and t2 change every iteration, so they are re-packed regardless of N.
  Invariant operands (bare Bov) are re-packed once per phase change, 1–3 times per run.
- Large DF blocks cannot hold several N-packs at once: benzene 5Z bBvv is 9.97 GB in FP64,
  about 17.7 GB as an N = 14 pack.

**No operand is ever needed at two N at the same time.** That is the only case a prefix (slice)
format would pay for.

---

## 2. Evidence

### 2.1 Method

- **`emu.py`** re-implements fast mode in numpy:
  - σ by the §3.2 integer rule (`oracle.largest_u`);
  - a′ = RN or TRUNC of ldexp(a, σ);
  - z = Σa′b′ exactly (20-bit limbs, FP64 BLAS on integers < 2⁵³, Python-int recombination);
  - one RN at the end.
- **It is bit-exact against `oracle.contract`:** 588 elements, N ∈ {3, 6, 8, 14, 17, 20}, RN and
  TRUNC, φ ∈ {0.5, 2, 4}, 0 mismatches and identical σ/τ.
- **The CRT precondition Σ|a′||b′| ≤ H was checked exactly** for every output of every scheme in
  every run, so z is exact wherever a result is quoted. The maximum of Σ|a′||b′|/H was 0.917
  for fast mode and 1.000 for the tightest partner-aware variant.
- **The K-weight study** (`c2gate*.py`) uses `oracle.scale_vector`, `quantize` and `round_dyadic`,
  with z exact in Python integers.
- **Data:**
  - (U − ½)·e^{φ·N(0,1)} with φ ∈ {0.5, 2, 4};
  - the M5 families of `gen` in bench/accuracy/contractions.py:45: uniform, decay_same,
    decay_opp, outliers, cancel;
  - `opp_nonsep`: anti-correlated, with every row and column decaying at its own rate;
  - `local_df`: 1D-chain locality;
  - `mixed`: half the rows dense.
- **Metrics:** the maximum over all outputs of the normwise error |D − x|/(‖a_i‖‖b_j‖) and of the
  elementwise error |D − x|/|x|, both against exact references. "Moduli saved" is interpolated on
  the fast-RN curve.
- **The sizes are small:** m = n ≤ 256 for the scaling variants, 12–24 for the K-weight and
  K-segment studies, 128–144 for the envelope rule. §2.6 shows the size matters for data with
  random structure.
- **The K-weight studies used one seed and norm ratios only.** No study ran the L5c guard
  (§4.4) on its data: `c2gate*.py` and `judge/keq_m.py` report error or
  log₂(‖a‖‖b‖/‖ã‖‖b̃‖), not guard acceptance. M-a (§4.3) measures it.

### 2.2 Fast mode: accuracy vs N [C]

Max normwise error, worst over the sizes 64²×256 … 256²×4096:

| data | N = 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| φ = 0.5, RN | 7.6e−5 | 2.7e−7 | 1.1e−9 | 4.5e−12 | 1.7e−14 | 1.3e−16 |
| φ = 0.5, TRUNC | 1.9e−4 | 7.7e−7 | 3.5e−9 | 1.1e−11 | 4.4e−14 | 2.8e−16 |
| φ = 4, RN | 4.0e−5 | 2.0e−7 | 9.5e−10 | 4.1e−12 | 1.9e−14 | 1.0e−16 |
| decay_opp, RN | 6.6e−9 (saturated) | 4.7e−9 | 1.2e−9 | 4.7e−12 | 2.2e−14 | 1.2e−16 |

- **Each modulus buys 15–16×** (3.9–4.0 bits), with E_max ≈ 2–3/T(N). This does not depend on k
  (256–4096) or on φ.
- **TRUNC is 2.0–3.2× worse**, about 0.3 modulus.
- **decay_opp's normwise error saturates for N ≤ 7** at about |x|/(‖a‖‖b‖). There
  Σ|a||b| ≈ 2^−24.6·‖a‖‖b‖, so D ≈ 0.
- **Fast mode leaves CRT budget unused:** the maximum of Σ|a′||b′|/H is 0.39–0.92 on dense data
  and below 5e−4 on decay_opp. Partner-aware scaling (§2.5) tries to spend that budget.

### 2.3 Cost vs N [M]

**Provenance of each row** (`perf.py`, `nsweep.py`, `predict_cc.py`):
- Square rows: spec `ik,kj->ij`, dense default strides, from `planner.Query(spec,
  ext).predict(N)`.
- DF row: spec `acQ,bdQ->acbd` with a = b = c = d = 64 and Q = 2032, dense default strides,
  from the same call.
- QZ row: spec `kiQ,acQ->kiac` with k = i = 21, a = c = 489 and Q = 1122, from
  `host.predict(spec, ext, N=N, accuracy=TYPICAL)`.
- Every row: g = N, DEFAULT regime, COLD, GENERATE, β = 0, in-line operands, no budget.
- **The spec matters by about 3 %.** R-api gets 2.77 / 4.61 ms at N = 8 / 14 for the dense
  `ik,jk->ij` form of 4096³ from predict and rank, against 2.86 / 4.70 below; regime and cache
  make no difference there [M, review]. The pack times and the packed-AB ratio reproduce
  exactly. DESIGN figures must name the spec, the strides and the request flags.

TFLOPS-eq (time):

| shape | N = 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| 1024³ | 30.2 (71 µs) | 24.4 (88) | 21.8 (99) | 18.5 (116) | 17.1 (125) | 14.3 (150) | 13.7 (157) | 11.3 (190) |
| 4096³ | 82.9 (1.66 ms) | 61.2 (2.25) | 48.1 (2.86) | 39.8 (3.46) | 33.9 (4.05) | 29.2 (4.70) | 26.0 (5.30) | 20.9 (6.58) |
| 8192³ | 106.1 (10.4 ms) | 75.4 (14.6) | 58.1 (18.9) | 47.5 (23.2) | 40.1 (27.4) | 34.5 (31.9) | 30.4 (36.1) | 24.4 (45.0) |
| DF acQ,bdQ→acbd 64⁴, Q = 2032 | – | 56.7 (1.20 ms) | 44.0 (1.55) | 36.3 (1.88) | 30.9 (2.21) | 26.3 (2.59) | 23.4 (2.92) | 18.6 (3.66) |
| QZ kiQ,acQ→kiac (o 21, v 489, Q 1122) | 32.3 (7.32 ms) | 25.9 (9.15) | 21.3 (11.09) | 18.3 (12.96) | 16.0 (14.82) | 13.9 (17.02) | 12.5 (18.90) | – |

- **Time ratio N = 14 / N = 8:** 1.52 (1024³), 1.64 (4096³), 1.69 (8192³), 1.67 (DF). With both
  operands packed (contract only) it is 1.75 on every shape.
- **Small and skinny calls gain much less** [M, review], t14/t8 at the best g:

  | shape | 256³ | 512³ | Jacobi view 1600×100×1600 | 256²×65536 | ≥ 1024³, DF |
  |---|---|---|---|---|---|
  | t14/t8 | 1.09 | 1.23 | 1.35 | 1.34 | 1.52–1.69 |

  A workload of many small plans gains about 10–25 % from a lower N, not 1.4–1.5×.
- **Predicted time is monotone in N at the best g** on sq1024, sq4096, DF, 256²×65536 and
  s256, and the workspace is monotone in N at g = 0 and g = 1 on all of them [M, review,
  `review_api/spot.py`].
- **Linear fits in µs:** 41 + 7.7·N (1024³, with K4 wave steps at N = 9, 13 and 17),
  363 + 310·N (4096³), 1441 + 2176·N (8192³), 120 + 177·N (DF).
  - The fixed part (FP64 reads in KF/KL, K0, K5's base cost) caps the gain at low N: N = 4 is
    only 1.35× faster than N = 6 at 4096³.
  - A chooser must therefore query the model, not assume time is linear in N.
- **K5's share of the call at N = 14** is 8.6 % (4096³), 5.1 % (8192³) and 15.6 % (DF) [M]. It
  was measured at 16.8 % on sq4096 (NOTES-D1:106: 0.887 of 5.276 ms).
- **Re-pack cost:** `imd_pack` of a 4096² operand is 313 / 414 µs at N = 8 / 14 [M] (measured
  0.437 ms at N = 14, NOTES-D1:107), against a call of 2.86 / 4.70 ms. That is about 11 % of one
  call, paid once per switch for each invariant operand.
- **The N ≤ 7 predictions fall outside the hold-out** (§1.2 item 8). [KT26] also reports INT8
  throughput that depends on the data at small N, through the fraction of zeros after scaling.
  M-c in stage 2 checks these predictions.

### 2.4 Precision schedules [M]

Sixteen iterations, assuming the iteration count does not change (M-d measures that).

**Ending at N = 14.** The schedule 5×N6 + 4×N8 + 4×N10 + 3×N14 against a fixed N = 14, so the
final precision is equal:

| shape | schedule: TFLOPS-eq (time) | fixed N = 14 | gain |
|---|---|---|---|
| 1024³ | 19.6 (1.75 ms) | 14.3 (2.40 ms) | 1.37× |
| 4096³ | 43.5 (50.6 ms) | 29.2 (75.2 ms) | 1.49× |
| 8192³ | 52.2 (337 ms) | 34.5 (510 ms) | 1.51× |
| DF | 39.7 (27.5 ms) | 26.3 (41.4 ms) | 1.51× |
| QZ kiQ,acQ→kiac | 19.6 (193 ms) | 13.9 (272 ms) | 1.41× |

**Ending at N = 9**, ehrenfest's historical operating point: 6×N6 + 10×N9 against a fixed N = 9
gives 1.09× (1024³), 1.12× (4096³) and 1.13× (8192³ and DF).

- Re-packing one invariant 4096² operand at each of the three switches costs about 1.1 ms, about
  2 % of the 50.6 ms schedule [M].
- So C1's value depends on the final precision the consumer needs: about 1.4–1.5× when the last
  phase needs N = 14, about 1.1× when it needs N ≈ 9.
- **It also depends on the shape.** The rows above are large calls. On the Jacobi view
  1600×100×1600 the same schedule is predicted at 0.79× of fixed N = 14 (1.27×), and on an
  o = 10, v = 96 PPL view (100 × 9216 × 9216) at 0.75× (1.33×) [M, review]. Many-small-plans
  workloads gain about 10–25 % (§2.3).
- **All of it comes from running at a lower N**, which per-N plans already allow (§3, A1 vs A2).
- **These are predictions at N = 6 and 8, outside the hold-out** (§1.2 item 8). M-c (§4.3) must
  validate them before any schedule gain using N ≤ 7 enters DESIGN as a claim.

### 2.5 Partner-aware row/column shifts (the accurate-mode family) [C]

Every variant is an exact integer rule, and the CRT precondition was checked for each:
- `acc_sc25`: faithful integer [SC25]. Ā = ⌈|a|·2^{5−γ}⌉ ≤ 64 (B̄ the same way), C̄ = ĀB̄ exactly,
  and d_i is maximal with 4^{d_i}·max_j C̄_ij ≤ H. It is sound by the ceiling lemma (Appendix B,
  LA1).
- `acc_exact`: the same split, but from the exact C̃ = |Â||B̂|. This is the limit of any rule that
  takes a max over partners, and it needs a 64-bit product.
- `hybrid_nw`: never below the fast shifts.
- `blocked64`: Cauchy–Schwarz per 64-wide K block, i.e. a GEMM of inner dimension k/64.
- `envelope` (P3): R_i = Σ_h U_ih·max_j U′_hj, where U are the ceilings at the fast scale.
  No GEMM, never below fast mode, independent of the K order and symmetric under swapping the
  operands (Appendix B).
- Hölder ℓ1/ℓ∞: budgets T_A·T_B ≤ H that ignore the partner, with the split tuned per problem
  (its best case).

Moduli saved at equal max-normwise error (mean over sizes and N = 6..12):

| data | acc_sc25 | acc_exact | hybrid_nw | blocked64 | Hölder |
|---|---|---|---|---|---|
| φ = 0.5 | −0.03 | −0.02 | +0.13 | −0.11 | −0.31 |
| φ = 2 | −0.06 | +0.16 | +0.24 | −0.11 | −0.35 |
| φ = 4 | −0.04 | +0.13 | +0.17 | −0.02 | −0.23 |
| decay_opp | +0.34 | **+3.07** | +3.02 | +3.02 | −0.16 to −0.24 |

On decay_opp, the smallest N reaching a max **elementwise** error of 1e−2 / 1e−6 / 1e−10:

| scheme | 1e−2 | 1e−6 | 1e−10 |
|---|---|---|---|
| fast | 13 | 17 | > 17 |
| acc_exact | 10 | 13 | 17 |
| hybrid_nw, blocked64 | 10 | 14 | 17 |
| acc_sc25 | 13 | 16 | > 17 |

**Envelope rule** (`c2bcs/bcs.jsonl`): moduli saved in the smallest N meeting a p99 elementwise
target of 1e−6 / 1e−8 / 1e−10 / 1e−12:

| data | moduli saved |
|---|---|
| φ ≤ 2 | 0 |
| φ = 4 | 0–1 |
| outliers | 1 (12/14/16/18 → 12/13/15/17) |
| decay_opp, 9 decades | 3 (14/16/18/20 → 11/13/15/17) |
| 3 decades | 1 |
| 12 decades | 5; 1e−10 becomes reachable at N = 17, where fast mode never reaches it at N ≤ 20 |
| `local_df` | 0 |
| `mixed` | 0–1 |

Wherever the envelope rule gains, it equals the full-GEMM bound.

**Findings.**
- **Row and column shifts cannot use the per-pair headroom on dense data.** The ideal per-pair
  headroom G_cs = log₂(‖a‖‖b‖/Σ|a||b|) has median 0.77 bits (φ = 0.5), 4–5 (φ = 2), 9–12
  (φ = 4) and 24.6 (decay_opp). Row/column shifts realize almost none of it on dense data,
  because each shift must serve its worst partner. On decay_opp they realize nearly all of it.
- **About 7.8 bits of σ + τ make one modulus** (24 bits ↔ 3.05 moduli); §2.8 derives this.
- **Faithful SC25's 6-bit ceiling erases the gain**, even on decay_opp (+0.34): every small entry
  is inflated to ≥ 1. The ≈ 3 moduli need a high-dynamic-range bound (exact, envelope or
  blocked).
- **Hölder budgets are always worse** than Cauchy–Schwarz.
- **This matches [KT26]:** a scale-invariant Cauchy–Schwarz fast mode, which is what §3.2's exact
  rule is, matches accurate mode's accuracy on φ-random data.
- **Cost, in moduli-equivalents of time** [E] (`c2cost.md`). Every variant adds one more
  operand read, because KF/KL can no longer pack group 1 before σ is known, plus the cost of the
  bound itself:

  | variant | 1024³ | 4096³ | 8192³ | DF |
  |---|---|---|---|---|
  | INT8 SC25 GEMM | 3.0 | 1.8 | 1.4 | 1.6 |
  | FP32 exact bound | 5.8 | 6.3 | 6.7 | 5.5 |
  | blocked64 | 2.5 | 1.0 | 0.6 | 0.9 |
  | envelope [I] | +10–17 %, ≈ 2–3.5 | +6–8 %, ≈ 1–1.2 | +3–4 %, ≈ 0.5–0.6 | +7–8 %, ≈ 1–1.3 |

  The envelope needs no GEMM, but it adds the extra read, two launches and a W = 3 pack at every
  N.
- **Net effect on decay_opp** at elementwise 1e−6: blocked64 is 1.06–1.16× faster than fast mode,
  the exact bound 1.00–1.06×, SC25 0.97–0.99×. The envelope rule is about 1.12× at 4096³ and
  1e−10 [I]. **On φ data every variant is a net loss.**

### 2.6 K-equilibration [C]

**The rule.**
- Power-of-two weights go on the contracted index: ã_ih = a_ih·2^{w_h} and
  b̃_hj = b_hj·2^{−w_h}. Every product a_ih·b_hj is unchanged, so x is unchanged.
- The fast rule then runs on (ã, b̃), and the error bound scales with ‖ã_i‖‖b̃_j‖ (L5, §4.4).
- For separable profiles |a_ih| ≈ r_i·α_h and |b_hj| ≈ β_h·c_j, Cauchy–Schwarz on (ã, b̃) is tight
  when 4^{w_h} ∝ β_h/α_h.
- The study's rule:
  - c_A(h) = max_i ilogb|a_ih| and c_B(h) = max_j ilogb|b_hj|;
  - Δ_h = c_B − c_A;
  - w_h = ⌊(Δ_h − med Δ)/2⌋ outside a deadband of 3 around the lower median, and 0 inside it
    (`keq_db3`). The plain rule has no deadband.

**Study** (`run_all.out`, `run2.out`; 24×24, k = 600): smallest N with max elementwise error
≤ 1e−6 / 1e−8 / 1e−10 / 1e−12, RN, over all outputs.

| data | fast | K-eq, deadband 3 | exact-integer SC25 | K-gain, median bits | G_cs, median bits |
|---|---|---|---|---|---|
| uniform | 9/11/12/13 | 9/11/12/13 | 9/11/12/13 | 0.00 | 0.42 |
| φ = 0.5 | 9/11/13/14 | 9/11/13/14 | 9/11/13/14 | −0.15 (plain rule) | 0.76 |
| φ = 2 | 9/11/13/14 | 9/11/13/14 | 9/11/12/14 | 0.91 | 4.35 |
| φ = 4 | 11/13/14/16 | 10/12/14/15 | 11/13/14/16 | 2.96 | 9.65 |
| decay_same | 8/9/11/13 | 8/9/11/13 | 8/10/11/13 | −0.12 (plain rule) | 0.05 |
| decay_opp | 15/17/18/20 | **9/11/12/14** | 15/16/18/20 | 24.41 | 24.63 |
| opp_nonsep | 15/17/18/>20 | **9/10/12/14** | 15/17/18/>20 | 21.10 | 22.17 |
| outliers | 14/15/17/18 | 10/12/14/15 | 13/15/17/18 | 10.54 | 15.88 |

At 16×16, k = 3600 (`run3.out`), fast → K-eq:
- decay_opp: 16/17/19/>20 → 9/11/12/14;
- opp_nonsep: 15/17/19/>20 → 10/11/13/15;
- outliers: 12/14/16/18 → 10/11/12/15;
- φ = 4: 11/12/14/16 → 10/12/14/15;
- uniform: unchanged.

The plain rule lost up to 1 modulus on φ = 0.5 and decay_same (10/11/13/15 and 8/10/12/13).

**Size dependence** (the review's check `judge/keq_m.py`, k = 1024): log₂(‖a_i‖‖b_j‖/‖ã_i‖‖b̃_j‖),
median / worst pair.

| data | rule | m = n = 16 | 128 | 512 |
|---|---|---|---|---|
| decay_opp | both | ≈ +24.3 | ≈ +24.3 | ≈ +24.3 |
| opp_nonsep | both | +20.4 to +21.1 (median) | same | same |
| outliers | plain and deadband (one pair reported) | +14.5 / −2.1 | +9.9 / −10.8 | +3.3 / −13.6 |
| φ = 4 | deadband 3 | +3.5 / −0.05 | +1.2 / −6.2 | −0.04 / −8.6 |
| φ = 2 | deadband 3 | +1.0 / +0.1 | −0.05 / −2.8 | −0.04 / −3.7 |

- **Structured anti-correlated profiles gain at every size.** decay_opp and opp_nonsep gain
  20–24 bits, about 5–6 moduli, independent of size.
- **The gains on randomly structured data are small-m artifacts.** At m = n = 512 the median gain
  is about 0 (φ) or +3.3 bits (outliers), and the worst output *loses* 3.7–13.6 bits (about
  1–3.5 moduli). The deadband does not prevent regressions at realistic sizes, so a strict opt-in
  or a guard is required. The L5c guard (§4.4) is exact and never worse **in the §3.8 bound**;
  it says nothing about the realized error. It was not run in any study here (§2.1), so M-a
  reports its acceptance rate, the worst pair, and the realized regressions among accepted
  cases.
- **Stale weights**, drawn from another sample of the same distribution as weights frozen in a
  pack would be:
  - Structured profiles keep the full gain: decay_opp 9/11/12/14; opp_nonsep 9/10/12/14 at
    k = 600 and 9/11/13/15 at k = 3600.
  - Random structure loses. Outliers at k = 600 reach 15/17/19/>20, at least 1–2 moduli worse
    than fast mode; at k = 3600 they equal fast mode. φ = 4 at k = 3600 reaches 12/14/16/18
    against fast mode's 11/12/14/16, 1–2 moduli worse.
- **Time at equal elementwise accuracy on decay_opp** [M]. These are **upper bounds**: they
  count neither the helper nor the pre-scaling (§4.5). Dense data gain nothing.

  | target | N, fast → K-eq | shape | TFLOPS-eq (time) | speedup, upper bound |
  |---|---|---|---|---|
  | 1e−10 | 19 → 12 (k = 3600) | 4096³ | 21.9 (6.27 ms) → 33.9 (4.05 ms) | 1.55× |
  | 1e−10 | 19 → 12 (k = 3600) | DF | 19.5 (3.49 ms) → 30.9 (2.21 ms) | 1.58× |
  | 1e−10 | 18 → 12 (k = 600) | 4096³ | | 1.47× |
  | 1e−6 | 16 → 9 | 4096³ | 26.0 (5.30 ms) → 43.5 (3.16 ms) | 1.68× |

- **Recurring costs** for operands rebuilt every iteration (t2, dressed integrals; §1.3) [E]:
  - Pre-scaling costs one FP64 read and write, 16 B per element, per changed operand per call,
    unless it is fused into the operand's producer kernel. That is about 0.3 ms per 4096²
    operand [E, review].
  - The helper costs one read of A and B without the guard and two with it, about 0.3 / 0.6 ms at
    4096³. It recurs only if it is re-run on each new pair, for example to keep the guard
    current.
  - At 4096³, 1e−10, the 1.55× becomes about 1.44× with one unfused changing operand and about
    1.35× with two [E, review]. Re-running the guarded helper every call as well gives about
    1.27× and 1.19× [E].
  - Scaled copies cost memory where a tensor is also contracted over other indices (§4.5
    Risks).
  - The Q13 gate and M-e count all of these terms.

### 2.7 K-segmentation [C, small]

The caller splits K into S contiguous bands and accumulates them with β = 1: S calls of one plan,
moving only the base pointers, each segment with its own σ and τ.

Data: `c2cmp.py`, the M5 generator, m = n = 12, k = 600, one seed. Each cell is max elementwise /
max normwise error:

| data, N | fast | S = 2 | S = 4 | integer SC25 |
|---|---|---|---|---|
| decay_opp, 8 | 1.2e+1 / 7.3e−10 | 1.1e−2 / 2.7e−14 | 2.2e−5 / 2.5e−16 | 2.8e+1 / 2.6e−10 |
| decay_opp, 10 | 2.7e−1 / 3.5e−12 | 1.5e−5 / 1.6e−16 | 6.5e−8 / 1.0e−18 | 8.8e−2 / 1.4e−12 |
| decay_opp, 12 | 7.4e−3 / 1.8e−14 | 1.0e−7 / 6.3e−19 | 3.9e−10 / 4.5e−21 | 2.1e−4 / 3.9e−15 |
| decay_opp, 14 | 2.2e−6 / 5.4e−17 | 9.2e−11 / 2.8e−21 | 5.2e−12 / 2.6e−23 | 5.5e−6 / 2.7e−17 |
| uniform, 8 | 1.6e−6 / 6.5e−10 | 1.6e−6 / 5.9e−10 | 4.5e−7 / 3.1e−10 | as fast |
| uniform, 14 | 4.6e−13 / 7.2e−17 | 3.7e−14 / 4.0e−17 | 6.5e−14 / 4.2e−17 | as fast |
| outliers, 8 | 1.6e−2 / 5.8e−10 | 1.6e−2 / 4.6e−10 | 1.6e−2 / 3.9e−10 | 6.3e−3 / 4.2e−10 |
| outliers, 12 | 7.7e−7 / 1.2e−14 | as fast | as fast | 6.3e−8 / 7.1e−15 |

- **Gain.** On a 9-decade monotone profile, S bands shrink each band's ‖a‖‖b‖/Σ|ab| from about
  10⁹ to about S·10^{9/S}. That is about 3.6 moduli at S = 2 and about 5 at S = 4, which matches
  the data. Aligned and outlier data gain nothing.
- **Cost** [E, first-round review]: S-fold K0, K5 and K4 W-plane writes (N·b·m·n̂ bytes per
  segment), plus the β = 1 read of C.

  | S | 4096³ | DF | 8192³ |
  |---|---|---|---|
  | 2 | +17 % | +31 % | +10 % |
  | 4 | +51 % | +93 % | +30 % |

  Net at decay_opp, 1e−10 elementwise, where fast mode needs N = 20: S = 2 at N = 14 is 1.08×
  (DF) to 1.28× (8192³) faster, and S = 4 is a net loss on DF.
- **Numerics.** Segmentation adds S − 1 FP64 roundings, so the result is no longer correctly
  rounded once, and its ELEMENTWISE bound is provably worse at high N (L7).
- **Against K-equilibration on decay_opp:** K-eq gives about 6 moduli at about 0 per-call cost;
  K-segmentation gives 3–5 at +10–93 %. K-segmentation needs only contiguous magnitude bands, not
  separable profiles, so it stays as a recipe.

### 2.8 Why K-weights beat σ/τ shifts per bit (from L3a)

L3a gives |y − x| ≤ δ(2^{−τ}‖a‖₁ + 2^{−σ}‖b‖₁) + kδ²·2^{−σ−τ}. Fast mode sets
2^{−σ} ≈ ‖a‖/d, within a factor 2s (L1d), with d = T − κ/2 (RN) or T (TRUNC) as everywhere in
these notes.

- **σ/τ increments Δσ, Δτ (accurate mode).** The first term becomes
  δ(2^{−Δτ}‖a‖₁‖b‖ + 2^{−Δσ}‖b‖₁‖a‖)/d. Each part shrinks only by the *partner's* increment, so
  a σ + τ gain of G bits (Δσ = Δτ = G/2) cuts the error by 2^{G/2}: about G/(2·3.9) ≈ G/7.8
  moduli.
- **K-weights.** Both parts scale with ‖ã‖‖b̃‖, so a norm gain of G bits cuts the error by 2^G:
  G/3.9 moduli.
- **The same headroom caps both.** Cauchy–Schwarz on (ã, b̃) gives ‖ã‖‖b̃‖ ≥ Σ|ã b̃| = Σ|ab|, and
  the σ + τ increment is at most log₂(H/Σ|a′||b′|) ≈ G_cs.
- **On decay_opp both realize about 24 of the 24.6 bits.** That is about 6 moduli for K-weights
  against about 3 for the best accurate variant, which is the data of §2.5–§2.6.
- After K-equilibration about 0.2 bits of headroom remain, so accurate mode on top would add
  nothing.
- **G/7.8 and G/3.9 are estimates from the bound's first term, confirmed by the data of
  §2.5–§2.6. They are not theorems.** DESIGN §7 must word them that way.

### 2.9 Ozaki-I slices against Ozaki-II [C]

- **At equal GEMM count, Ozaki-II wins from about 4 GEMMs on.** The orchestrator's emulator used
  balanced RN digits (β = 7, int8-native, and β = 8, optimistic) and triangle and square pair
  sets; FP64 and exact accumulation agree to two digits.
  - From 6 INT8 GEMMs Ozaki-II is more accurate: 2.0–2.7e−7 at N = 6 against 4.6e−7–5.9e−6 for
    the β = 8, S = 3 triangle (2–25×). At 10 GEMMs the gap is 400–9000×.
  - The crossover is about 4 GEMMs, at a normwise error of about 2e−5 to 8e−5.
- **Slices needed to match Ozaki-II** (worst over φ data):

  | Ozaki-II N | slices |
  |---|---|
  | 6 | S = 4 triangle (10 GEMMs) |
  | 8 | S = 5 (15 GEMMs) at β = 8, or 5–6 at β = 7 |
  | 14 | S = 8 (36 GEMMs) or 9 (45) |
- **Matched N under P2's prefix-canonical design** (§3.3), k = 4096:
  - uniform data: N ≈ 2S + 1. S = 3's 2.1e−8 lies between N = 6's 1.3e−7 and N = 7's 8.2e−9,
    a fractional N of about 6.6.
  - **φ = 0.5: N ≈ 2S** (S = 2 / 3 / 4 ≈ N = 4 / 6 / 8).
  - φ ≥ 2: N ≈ 2S − 1.
- **Time at matched accuracy** [E, recomposed from Ozaki-II phases], slice time / Ozaki-II time
  when matched on uniform data:

  | S | matched Ozaki-II N | slice / Ozaki-II |
  |---|---|---|
  | 2 | 5 | 0.64–0.83 |
  | 3 | 7 | 0.77–1.01 |
  | 4 | 9 | 0.89–1.19 |
  | 7 | 15 | 1.25–1.75 |

  - The ranges span 512³ to 8192³, DF, 2048²×65536 and 256²×262144; the long-k shapes give the
    low ends at S ≥ 4.
  - Matched on heavy-tailed data, the ratio is ≥ 1.02 at every S.
  - With φ = 0.5's N ≈ 2S, S = 3 only breaks even.
  - "Matched" means the smallest integer N at least as accurate, which flatters slices.
- **The prefix-canonical quantizer ⌊y + 128/255⌋ is biased by +1/510 unit for both signs.**
  - On data with nonzero mean the bias grows linearly in k, against √k for the random part. It
    is visible at S = 1, k = 65536: same-sign 1.7e−4 against uniform 8.6e−5.
  - So "long k favours slices" does not hold for same-sign data, and a zero-mean Hoeffding
    estimate needs a bias term.
- **An S_max = 8 slice pack** (8 B per element, like FP64) serves every S ≤ 8 bit-exactly, whereas
  Ozaki-II needs N B per element for each precision. That is a memory advantage only for
  operands needed at several precisions at once.

### 2.10 Statistics-driven N [C, small]

ρ bounds from L6 (stage 5), taken from a source call and passed to `min_moduli` as `cond`, against the
measured smallest N reaching max elementwise 1e−6 / 1e−8 / 1e−10 / 1e−12 (P1's study):

| data | source N0 | measured | predicted, BOUND | predicted, TYPICAL |
|---|---|---|---|---|
| uniform, 24², k = 600 | 8 | 9/11/12/13 | 11/12/14/16 | 10/12/14/15 |
| φ = 4, 24², k = 600 | 8 | 11/13/14/16 | 13/15/17/19 | 13/15/16/18 |
| decay_opp, K-eq, 24², k = 600 | 8 | 9/11/12/14 | 11/12/14/16 | 10/12/14/15 |
| decay_opp, fast, 24², k = 600 | 8 | 15/17/18/20 | uncertified (576 of 576 outputs) | uncertified |
| decay_opp, fast, 16², k = 3600 | 12 | 16/17/19/>20 | 18/19/>20/>20 | 17/19/>20/>20 |

- BOUND picks 2–3 moduli above the measured minimum, TYPICAL 1–2.
- A source call at too low an N certifies nothing on ill-conditioned data.
- The per-output certificate (L6b) is rigorous but not tight: 19–30× above the measured error at
  k = 600 (median about 100×), and 55–90× at k = 3600.

---

## 3. Options and verdicts

### 3.1 Verdicts

| # | option | verdict | why |
|---|---|---|---|
| | **C1(a)** | | |
| A1 | Plan per N and pack per N, with published monotone footprints and re-pack in place | **adopt (stage 0)** | Works today (§1.2 items 4–6). Per-call cost and result are identical to any range plan. A GENERATE plan holds no device bytes. Plan creation is measured at 0.011–0.012 ms for GENERATE and 0.025–0.123 ms for RESIDENT (NOTES-M4.md:213-218); that is host time, once per level, made before the iteration loop (§4.2). |
| A2 | N-range plan: one plan, a per-call N′ ∈ [N_lo, N_hi], launch-identical to the single-N′ plan (P1 S1) | **park** (trigger in §3.2) | Zero per-call gain by construction. M-size: new entry points and descriptor structs, a new lowering version, and changes to 4 TLA+ modules (Pipeline alone has 103 models). It saves only plan count, host bytes and creation time, none of which has been measured. |
| A3 | Multi-N pack (Σ N_i planes from one KL pass) | reject | Same bytes as separate packs. It saves one FP64 read per pack creation and needs N-set key semantics. |
| A4 | Round-to-odd reference-integer pack (RN of RO with ≥ 2 guard bits equals direct RN; 0 / 60 000 mismatches [C]) | reject | 5 / 8 / 9 / 11 B per element at N_max = 8 / 14 / 16 / 20, which is ≥ FP64 for N_max ≥ 14. Each call still reads ≥ 8 B per element. It saves only the S pass, which KL already fuses. |
| A5 | Scale-state cache ((γ, S, flag), 19 B per row) | reject | Saves only KL pass 2, which is served from L2. |
| A6 | Lower-N planes from a higher-N pack | impossible | 2 has no inverse mod 256, rounding is not modular, and σ(N+1) − σ(N) depends on the data. CRT over a prefix gives z mod P(N′) (PackedOperand `neg_n`). |
| | **C1(b)** | | |
| B1 | Host-side target → N (`imd_err_min_moduli`, then one rank over the allowed levels ≥ that N, with the accuracy set; §6f.3's `rank([n, n+1])` is the two-level case), plus a thin `PrecisionPlan` | **adopt (stage 1)** | Already exists in C; the helper only composes it per phase. The policy stays with the caller (D8). |
| B2 | Rigorous ρ from the previous output (§4.2) | **adopt (documentation, stage 1)** | No new library code: one pass over C plus the caller's row and column norms (a pass over A and B, unless the caller has them), and the existing NORMWISE bound. |
| B3 | K5 statistics plus an a-posteriori certificate (L6), with one pure host function `imd_stats_cond` that turns the statistics into `cond` (P1 S2, reduced) | **adopt later, gated (stage 5)** | Fuses B2 into K5. It needs no norms at all (L6 uses ‖a·2^σ‖ ≤ d), which saves the pass over A and B as well as the pass over C, and it adds a rigorous per-call elementwise certificate. Worth an ABI change once a consumer runs elementwise targets. |
| B4 | Choosing N on the device within a call | reject | N fixes the launch sequence, the K5 instantiation, W, the workspace and σ itself. Kernel scope forbids the sync. |
| B5 | Device-side selection feeding a caller-owned conditional (SWITCH) graph node (P1 S5) | park | No consumer runs captured adaptive iterations. |
| | **C1(c)** | | |
| C1 | Ozaki-I slice mode with prefix packs and an exact Horner combine (P2) | **park with triggers** (§3.3) | Faster only on benign data below about 1e−8 (§2.9). Break-even near 1e−10, 1.25–1.75× slower at FP64 level, and no win beyond S = 2 on heavy-tailed rows. Its capability (one pack for every precision) is not used by the workload (§1.3). L-size. |
| C2 | One scheme per phase: a slice pack in low phases, then one Ozaki-II re-pack at the end (P2 H4) | park with C1 | The gain covers only low-phase calls, about 1–3 % of a schedule's emulated time [I, first-round review]. |
| C3 | Residues derived from a slice pack (H2); mixed digit × residue products (H3) | reject | H2 reads as many bytes as re-packing from FP64 and needs a double-rounding quantizer. H3 costs S·N GEMMs. |
| | **C2** | | |
| D1 | [SC25] accurate mode, faithful 6-bit integer form | reject | ≤ 0.34 moduli even on decay_opp (§2.5). It adds one INT8 GEMM, KF/KL lose their group-1 fusion, and σ depends on the partner, which breaks D1. |
| D2 | Exact or blocked partner bounds (`acc_exact`, `blocked64`) | reject | About 3 moduli, and only on separable anti-correlated data. They cost 0.6–6.7 moduli of time (§2.5). The blocked bound depends on the K order: with K permuted it gains 0 [C, c2bcs]. |
| D3 | Never-worse partner-envelope rule, `IMD_EMULATE_ACCURATE` (P3) | **record, do not build** (Appendix B) | Sound and exact. But it gains about 3 moduli where K-equilibration gains about 6 (§2.8). It costs one more operand read and W = 3 at every N, and packed operands are NOT_SUPPORTED, so D1 is lost exactly for the reused operands. It is the fallback if the gate finds structured *non-separable* data where K-weights fail. |
| D4 | One-sided coupling; headroom packs (P3 stage 4) | reject | One-sided: only 1.3–1.4× smaller error with 25 bits of slack [C]. Headroom packs as drafted are unsound under RN (Appendix A, item 16). |
| D5 | Hölder or asymmetric budgets that ignore the partner | reject | Always worse than Cauchy–Schwarz (§2.5). |
| D6 | Non-power-of-two fine scales | reject | ≤ 0.44 bits per side, about 0.11 modulus, and it needs exact division by odd integers in K5. |
| D7 | **K-equilibration** (P1) | **adopt, staged and gated** (oracle rule in stage 2; stages 4, 6) | About 6 moduli on separable anti-correlated data. The library adds no per-call cost; the caller's pre-scaling and the helper do, for operands rebuilt every iteration (§2.6). Once w is fixed it does not depend on the partner. Exact (L5). It avoids regressions in the bound only with the guard (§2.6). |
| D8 | K-segmentation (P4) | **adopt as a documented recipe, with L7** (stage 3) | 3–5 moduli on banded anti-correlated data, with no API change, and packs still work. It costs +10–93 % per call and the single final rounding. An accuracy-API flag only if the gate shows value beyond D7 (Q15). |

### 3.2 The parked range plan (kept for revival)

If revived, the design is as P1 drafted it, with these corrections:
- **API:** `imd_plan_create_ex` takes a `struct_size`-versioned descriptor (N_lo, options with
  num_moduli = N_hi, flags), and `imd_contract_ex` takes a per-call N′.
- **Layout and launches:**
  - The workspace layout is fixed at (N_hi, g_hi), with g_hi = N_hi when g = 0.
  - A call at N′ issues exactly the single-N′ plan's launches (grids, `pack_words(N′)`,
    `launch_k5_for(N′)`, `gsize = min(g′, N′ − g0)`). Only the region offsets differ.
  - K5 reads only planes [0, N′), each written by this call.
- **Plan creation:**
  - RN acceptance is checked once, at N_lo; T(N) increases with N.
  - Per N′ the plan holds a shift-constant table (40 B) and K4 shape keys for every group size
    {g′} ∪ {N′ mod g′}.
  - It holds one K4Plan per distinct `k4_select` configuration, at most 4 when g = 0.
- **Packed slots:** each slot key is a template whose N word is replaced at call time.
- **TLA+:**
  - Pipeline: NHi/NCall, with `neg_range_k5_nhi`, `neg_range_layout_ncall`,
    `neg_range_slot_stride`.
  - IndexMap: `neg_range_kappa_nhi`.
  - PackedOperand: `neg_range_key_template`. The template inherits the stale-handle hazard of
    §4.1, so a revival also builds §4.1's optional StreamSpec model (cells carrying
    ⟨version, key⟩, a c3 after the re-pack).
  - HostAPI: `neg_range_plan_state`.
- **Trigger:** a consumer that cannot re-plan per phase, or a measurement showing that plan count,
  plan creation or RESIDENT memory matters.
- The per-call struct of stage 5 (`imd_call_t`, §4.6) reserves room for N′, so a revived range
  plan needs no new contract entry point.

### 3.3 Slice-mode notes (kept for revival)

- **Digits:** balanced int8 digits in [−128, 127], so the MMA bound and K_c′ are unchanged.
- **Quantizer:** P_S = ⌊a·2^{σ+8(S−1)} + 128/255⌋.
  - θ = 128/255 is the unique constant that makes the expansion prefix-canonical: the top S digits
    of any S_max expansion are the S-digit expansion, so results never depend on S_max.
  - The error lies in (−127/255, 128/255] of a unit.
  - The +1/510 bias of §2.9 must enter E_I and any typical estimate.
- **Shift:** the largest s with every a_h·2^s ∈ [−32768/255, 32512/255), so σ ∈ {5, 6, 7} − γ.
- **Exact accumulation and output:**
  - Diagonal planes are int32 iff 2^14·S·k ≤ 2^31 − 1, else int64.
  - The Horner combine needs at most 4 limbs for S ≤ 8 and k ≤ 2^40.
  - The output is one `round_scaled`.
- **ABI: an explicit scheme selector, not the rounding field.** The rounding field would fail for
  four reasons:
  - num_moduli would mean S in every validator;
  - S = 1 conflicts with kNMin = 2 (include/immerdink/detail/constants.hpp:16) in `options_ok`,
    `imd_err_estimate`, `imd_err_min_moduli`, rank and the pack descriptor;
  - `pack_key` stores the rounding as a boolean (pack.cpp:80), so a new rounding value would key
    as RN;
  - the ordered compare S ≤ S_max changes the D1 key contract, which is exact equality today
    (pack.cpp:103-104).
- **Triggers:** a consumer needs one operand at two or more precisions at once under memory
  pressure (e.g. mixed-precision iterative refinement), or M-d measures phase-switch re-packs at
  ≥ 5 % of a phase. If triggered, "TLA+ for slices" means a new OzakiSlices.tla (toy β = 4) and
  PackedOperand and Pipeline extensions, as P2 drafted them.

---

## 4. The chosen design, staged

| stage | size | gate | ABI | delivers |
|---|---|---|---|---|
| 0. Precision phases on host API 13 | S | — | none | C1(a): published monotone footprints, layout certificate, re-pack rule, Python guard (no TLA+ change, R1) |
| 1. `PrecisionPlan` and recipes | S | — | none | C1(b) on the host; ρ from the output (`cert_rho_recipe`); schedule example |
| 2. Evidence | S (CPU) + M (GPU) | — | none | combined C2 study at realistic sizes with `oracle.kweights`; small-N hold-out; scheduled iterations (convergence on the CPU, time on the GPU) |
| 3. L7 and the K-segmentation recipe | S | — | none | L7 in PROOFS (+ Lean), `cert_kseg_composition`, `oracle.contract_segments`; DESIGN §7 re-rejection |
| 4. `imd_kweights` + caller-side equilibration | S–M | stage 2 passes (Q13) | host API +1 | C2, step 1, with L5 in PROOFS (+ Lean), its certificates and `GuardSound` |
| 5. K5 statistics + certificate | S–M | a consumer with elementwise targets (Q16) | host API +1 | C1(b) on the device, with L6 in PROOFS (+ Lean) and its certificates |
| 6. Library-internal weights + weighted packs | M | stage 4 measured (M-e) + real operands; the open layout item of §4.7 | host API +1 | C2, step 2 |

Each lemma is proved in the stage that first relies on it (R2, §5.2). Its hand proof is
recorded in §4.4 now.

DESIGN v1.19 gets the following:
- §6g "Adaptive precision", with the stage 0–1 text and later the stage-5 API;
- a §3.8 addendum (L7, stage 3; L6 when stage 5 ships) and §3.9 "K-weights" (stages 4 and 6);
- updated §6a and §6b.7 contract text;
- rewritten §7 rows.

### 4.1 Stage 0 — precision phases on host API 13

**Numerics:** none. Each phase is a plain plan at its N, so every §3 rule, certificate and proof
applies unchanged.

**Published guarantee** (DESIGN §6a integration contract, "from API 13 on"). It covers only
what a caller needs to size buffers once (R-api must-fix 3). Fix the descriptors, mode, vec_mode
and packed roles. That also fixes k̂ (A(k) depends on k alone, §4b.3) and the KF/KL choice (k̂
alone). Then:
- **F1.** `imd_workspace_size[_packed](N, g)` is nondecreasing in N at any fixed g ≥ 1, and
  nondecreasing in g ∈ [1, N]. With g = 0, which is normalized to N (analysis.cpp:98), it is
  nondecreasing in N.
- **F2.** `pack_bytes` of `imd_pack_layout` / `imd_pack_bytes` is nondecreasing in N.
- **F3.** `imd_pack` writes every byte of [0, `pack_bytes`) (already in
  include/immerdink/immerdink.h:74-75) and **no byte beyond it** (new text, checked by guard
  zones in `packed_repack_n`). A packed contract reads only [0, `pack_bytes`) of a packed buffer:
  that is the buffer's ALIASING extent (immerdink.h:107), and CompatSpec's `NoOOBRead`
  (spec/PackedOperand.tla:353) models it.

**Internal facts, not ABI** **[cert: cert_footprint_monotone]**. They are true of today's
formulas, but no caller needs them, and a future format (§4.7's kw region after res, a slice
format, a range plan) may break them. Under §6a a published promise could not be withdrawn,
so they stay certificates that a format change must revisit:
- **I1.** In `imd_pack_layout`, the sig and flag offsets and bytes, the res offset and
  `pack_ws_bytes` do not depend on N; res(N′) lies inside res(N_max) (the drafted TLA+
  `LayoutPrefix`, moved here because `pack_layout` is static, pack.cpp:35-57).
- **I2.** `imd_plan_bytes` device bytes do not depend on N or g. Host bytes do depend on them,
  through the K4 shape keys.

*Proof.* Every §4a region is either independent of (N, g) or has N or g as a factor
(workspace.cpp:49-56). align₂₅₆ is monotone, and the total is a sum of aligned regions in a fixed
order. In the pack, res is affine in N (pack.cpp:44-47) and the earlier regions do not contain N.
F3's first half is §6b.4. Its second half holds if each pack kernel writes only regions of the
layout, which is how they are built [I]; the guard zones of `packed_repack_n` check it. ∎

A random check [C] found 0 violations: 3000 shapes × N = 2..19 × g ∈ {1, ⌊N/2⌋, N}, using
params `workspace_size` and `pack_layout`, with pack offsets equal across N. R-api's host-model
spot check agrees on 5 shapes [M, review].

**Consequences, written into §6a:**
- **C-ws.** One workspace of `imd_workspace_size(N_max, g_max)` serves every plan with the same
  descriptors, options and packed roles, N ≤ N_max and g ≤ g_max (g = 0 counts as N), with at
  most one call in flight per workspace (§6 rule 1).
  - Size: ws(N, g) ≤ ws(N_max, g) ≤ ws(N_max, g_max) by F1, which applies because
    g ≤ N ≤ N_max (`options_ok`, analysis.cpp:17-23). The contract checks accept a larger
    workspace (immerdink.h:112-113).
  - Contents: the bytes a call at another N left behind are never read. Pipeline's
    junk-initialized workspace invariants `NoStaleRead` and `EveryRowByteWritten`
    (spec/Pipeline.tla:1223, 1232) say every byte a call reads, that call wrote. (HostAPI
    `neg_ws_shared` is about concurrency, not about reuse across plans.)
  - Region *offsets* differ between plans; each call uses its own plan's layout.
- **C-pack.** One buffer of `pack_bytes(N_max)` holds the pack of the same descriptor at any
  N′ ≤ N_max. This follows from F2 and F3 alone: `imd_pack` at N′ writes exactly
  [0, `pack_bytes(N′)`), and a contract at N′ reads only that range. The bytes above it keep
  stale planes that nothing reads. The offsets of I1 are not needed.
- A caller may instead size per phase and re-allocate at a switch. Both are exact through the
  host queries.

**Re-pack-in-place rule** (added to §6b.7):
- A packed buffer holds the pack of its **last** `imd_pack` only. Every contract that reads it
  must pass a handle whose key is that pack's key.
- The library compares handle with plan, never bytes with handle (§6b.10 #12). So a contract
  that passes the previous pack's handle, with its plan, after a re-pack at another N reads the
  bytes as the old N and returns a wrong C with SUCCESS.
- A CUDA graph that captured a contract on the buffer must not be replayed after the buffer is
  re-packed with another key.
- Ordering is the existing WAR rule: re-pack only after every contract that read the old
  contents.

**Spec: no TLA+ change** (R1, §5.2; the §6f.5 argument). The drafted StreamSpec extension does
not fit the module, and what it was meant to show is either modelled already or tautological:
- StreamSpec's cells hold version tokens, not values (spec/PackedOperand.tla:411-425). Its issue
  order is pack, c1, c2, post (:325), and the WAR edge makes post wait for c1 and c2 (:402). So
  c2 can never observe a re-pack, and no constant can make "c2 bind the N′ handle after the
  re-pack". CEqual and NoOOBRead are CompatSpec value-level invariants (:353), not StreamSpec
  ones.
- **Ordering is modelled.** A re-pack is the existing post = "repack" with WAR
  (`neg_stream_no_war`). A contract after the re-pack is the existing pack → contract RAW case
  (`neg_stream_no_raw`). The re-pack is just a pack.
- **The stale handle is §6b.10 #3's documented limit.** The library never compares bytes with
  handles (§6b.10 #12), so a witness that a stale handle gives a wrong C would only restate the
  caller rule.
- **The layout prefix is static** (`pack_layout`, pack.cpp:35-57). It is the params
  certificate I1, not a protocol property.
- Pipeline, IndexMap and HostAPI do not change. Per-N plans and handles are existing objects,
  and C-ws rests on Pipeline's existing invariants.
- **Optional model, if Eric wants one.** In StreamSpec, make each cell a pair ⟨version, key⟩,
  with N in the key (R-num). Add an op c3 after post with a RAW edge post → c3, and bind each
  contract to its plan's key (R-api). The invariant is then "every contract reads cells whose
  last writer's key equals its plan's key, given the caller rule". `neg_repack_stale` (c3 keeps
  the old plan) violates it, and so does dropping the post → c3 edge. Optionally, add a
  CompatSpec model at N′ with the buffer sized for N + 1.

**Python.**
- `PackSpec.with_moduli(N)` (= `dataclasses.replace`; `PackSpec` is a frozen dataclass,
  python/immerdink/host.py:493-494).
- `device.PackedBuffer(buf, nbytes=None)` records the (key digest, N, rounding, pack_bytes) of
  the Pack last written into it, on the host at enqueue, under a lock. `Plan.contract` refuses a
  `PackedBuffer` whose record differs from the plan slot's pack (`PackMismatchError` "buffer last
  packed at N=8, plan expects N=14"). It allocates nothing. The digest is the library's FNV-1a
  `key_digest` (pack.cpp:91-101), good for diagnostics only, so this is a safety net, not a
  soundness argument. Raw buffers behave as today.
- **The record follows host enqueue order, not GPU order.** Under graph replay, or across
  streams ordered by events, the record can differ from what the GPU sees. It guards the plain
  stream-ordered pattern only.
- Its lock-and-compare cost is added to `bench/python_overhead.py`.

**Tests.**
- CPU:
  - `host_footprint_monotone`: F1, F2 and the internal facts I1–I2 for `imd_workspace_size`,
    `_packed`, `imd_pack_layout` and `imd_plan_bytes` against params. The grid is
    m, n ∈ {1, 15, 16, 17, 255, 256, 257}, k ∈ {2, 96, 97, 1024, 1025, 131008, 131009},
    b ∈ {1, 3}, N = 2..19, g = 1..N, both vec modes, packed roles ∅/A/B/AB, both k_align
    policies, plus 10⁴ random shapes.
  - The `python_device` CPU part: `PackedBuffer` refusal through a fake library.
- GPU (queued for a quiet machine, under `_wip/gpu.lock`):
  - `packed_repack_n`:
    1. Pack X at N = 14 into a buffer sized for N = 14 and contract.
    2. Re-pack at N = 8 into the same buffer and contract with the N = 8 plan.
    3. Both results must be bit-exact against the oracle.
    4. The stale N = 14 plan with a `PackedBuffer` must raise `PackMismatchError`.
    5. Guard zones must show that `imd_pack` at N = 8 writes exactly [0, `pack_bytes(8)`) (F3).
  - `workspace_shared_n`: one workspace sized for (14, g = 0); plans N ∈ {5, 8, 14} called
    alternately on one stream; all bit-exact; the LD_PRELOAD shim counts no allocation.
  - `graph_phases`: one graph per phase (N = 6 and 10), **sharing one workspace**, replayed
    alternately, bit-exact.

**Risks.**
- Graph replays after a re-pack cannot be guarded; the rule is documentation only.
- A future pack format with offsets that depend on N would break I1, which the certificate would
  catch. C-pack does not depend on I1.
- Plans per N multiply the RESIDENT vector bytes by the number of levels, 8(2m + 2n + 2k + 3b)
  each. GENERATE plans hold none.
- **RESIDENT plan creation synchronizes.** It builds its vectors on `cudaStreamPerThread` and
  waits, and it is not capturable (DESIGN.md:1576-1577). So every level's plans are created
  before the iteration loop (§4.2, `PrecisionPlan.configs`), and `PlanCache.freeze()` is used
  for capture. A phase switch that creates a RESIDENT plan on demand synchronizes.
- **With `k4_tune` ON, a plan per N multiplies the plan-time GPU autotuning** by the number of
  levels.

### 4.2 Stage 1 — `PrecisionPlan` and recipes

**Placement.** `PrecisionPlan` lives in its own module, `python/immerdink/precision.py`, not in
`planner.py`. §6e.4 promises that planner.py computes nothing in Python (DESIGN.md:4204), and
`PrecisionPlan` does compute: levels, the monotone floor and share groups. It calls only
`Planner.min_moduli`, `Planner.rank`, and the workspace and pack-layout queries; no GPU, no
policy.

```python
# python/immerdink/precision.py
class PrecisionPlan:
    def __init__(self, planner, contractions, *, levels=(6, 8, 10, 12, 14), rounding="RN",
                 mode="EMULATE", share=(), packs=None, reuse=None, budget=0, monotone=True)
        # planner: a planner.Planner (calibration, regime and cache come from it, not duplicated)
        # contractions: {name: (spec, extents[, strides][, calls_per_iteration])}
        # levels: sorted allowed N (3-5 levels recommended)
        # share: [(group, [names])] - contractions that read one packed operand use one N
        # packs: {name: "NONE" | "A" | "B" | "AB"} - packed roles for rank
        # reuse: {name: contractions per imd_pack} for packed roles; default calls_per_iteration
        #        (the pack is remade every iteration); an invariant operand is remade at each
        #        phase switch, so its reuse is the calls of a phase
    def configs(self) -> dict      # {name: {N: Candidate}} at every level, up front
    def phase(self, targets, *, iterations=None, fail_budget=None) -> Phase
        # targets: {name: planner.Accuracy}
    def footprints(self) -> Footprints  # per name: max workspace over configs() and the value
                                        # per level; per packed operand: pack_bytes per level and max
    def reset(self)                     # forget the monotone floor: a new solve, geometry or restart
class Phase:   # lazy view
    N: dict; g: dict; vec_mode: dict; status: dict
    time_s: float; tflops_eq: float     # calls-weighted; TFLOPS-eq = sum 2bmnk / time
    err_bound: dict; err_typical: dict
```

There is no `noise_floor`: a sum of errors across contractions in possibly different metrics
means nothing.

**`configs()`** ranks each contraction once per level (all g, the given vec modes, packs, reuse
and budget) and returns the best candidate at each level. The caller creates every one of these
plans before the iteration loop, and uses `PlanCache.freeze()` for capture (§4.1 Risks). With
`k4_tune` ON, that multiplies the plan-time GPU autotuning by the number of levels. A
fixed-N rank needs no accuracy: the error estimate depends only on (N, rounding, k), so
accuracy makes every candidate at one N feasible or every one infeasible.

**`phase()` rule,** for each contraction c:
1. n_c = `min_moduli(acc_c, n_lo = levels[0], n_hi = levels[-1])`. A lowering error raises.
   MISSED sets the status MISSED with N = levels[−1]. The RN κ ≥ T levels are +∞ in the
   estimate, so they are skipped automatically.
2. f_c = the smallest level ≥ n_c.
3. With `monotone`, f_c = max(f_c, the previous phase's **final** N_c), i.e. its N after the
   share-group raise and the rank choice. `reset()` clears it.
4. Each share group's floor is raised to the maximum f over its members.
5. **One rank call** per contraction over the levels ≥ f_c (a share group: over its one N only),
   with the accuracy set, the given vec modes, packs, reuse and budget. Its best feasible
   candidate is the phase's choice. This equals the fastest of `configs()[c][N]` over N ≥ f_c,
   which an implementation may use as its memo. It holds even where time is not linear in N
   (§2.3), and it replaces both the draft's `rank([N_c])` and §6f.3's `rank([n, n+1])`, which is
   the two-level case.
6. **TYPICAL targets.** p in §6f.2 is the failure probability *per call* (DESIGN §6f.2,
   [review] p). With `iterations` and `fail_budget` P given, `phase()` passes
   p = P / (iterations · Σ_c calls_per_iteration_c) to every TYPICAL target of the phase: a
   union bound over the phase's calls. Without them it uses each Accuracy's own `fail_prob`.

The cost is one `min_moduli` (about 1.7 µs) and one rank over at most |levels| N values per
contraction per phase switch. That is off the hot path, and memoizable through `Memo`.

**Not in the library:** the forcing rule, i.e. residual → target. It ships as an example in
`bench/accuracy/schedule.py` and the docs:
- after iteration t, give each contraction c an ABSOLUTE target η·r_t·w_c (Σ w_c = 1), with
  `norm_product_c` refreshed per phase;
- raise N only at an iteration boundary;
- use BOUND for the final phase and TYPICAL for intermediate phases, as the user chooses, with
  p = P/C as in step 6.

**Rigorous ρ from the previous output** (documented recipe, ELEMENTWISE targets)
**[cert: cert_rho_recipe]**.
- *Inputs.* A call at N with α = 1 and β = 0, so its output C is D. Upper bounds
  n_ij ≥ ‖a_i‖‖b_j‖ from the caller's row and column 2-norms, computed so that they really are
  upper bounds (§6f.1's existing obligation).
- *U.* Let U = the NORMWISE `bound` of `imd_err_estimate` at that (N, rounding, k), with
  ν = √k unless the caller's ℓ1 ratios hold. It is evaluated as compose() does,
  E + E·u + u (accuracy.cpp:48-58), times (1 + 2^−40); never as E·(1 + u), since 1 + 2^−53
  rounds to 1.
- *Claim.* Every output with D_ij finite and normal (|D_ij| ≥ 2^−1022; PROOFS L3c needs D
  outside the subnormal range) and |D_ij| > U·n_ij satisfies

  ρ_ij ≤ n_ij / (|D_ij| − U·n_ij),

  because |x| ≥ |D| − |D − x| ≥ |D| − U·n.
- *FP-safe evaluation*, rounding ρ̂ upward: t = nextafter(U·n_ij, +∞),
  den = nextafter(|D_ij| − t, 0), certified only if den > 0, and
  ρ̂_ij = nextafter(n_ij/den, +∞). Each RN result is within half an ulp, so one ulp outward
  bounds it. `cert_rho_recipe` (params) checks ρ̂ ≥ ρ in exact rationals on random rows and on
  rows with |D| just above U·n.
- *Uncertified outputs* are zero, subnormal or non-finite ones, and those with
  |D_ij| ≤ U·n_ij. The caller decides whether they must meet the target (§6f.1 "over the
  outputs that must meet the target").
- *Scope.* The result is rigorous for that call's data and a prediction for the next
  iteration's.
- *Cost* [E]. One pass over C, plus the caller's norms: a pass over A and B, about 2× a pass
  over C at square shapes. At 4096³, N = 14, that is about 3 % for C and 6 % for the norms, so
  about 9 % of a call when every norm is recomputed. Norms of an invariant operand are computed
  once per phase.
- σ-derived norms, ‖a_i‖ ≤ d·2^{−σ_i} (L1b), are valid for nonzero rows only. A zero row has
  σ := 0, so its bound d·2^0 is vacuous and can inflate the maximum by up to about T. So the
  recipe uses real norms unless the caller knows its zero rows.
- Stage 5 fuses this recipe into K5, and **needs no norms at all**: L6 uses ‖a·2^σ‖ ≤ d, which
  the scaling rule guarantees. That is the hook's real advantage over the recipe.

**Tests.** A new CPU test `host_precision` of the module, next to `host_planner`:
- `PrecisionPlan.phase` equals a brute-force `min_moduli` + one rank over the levels ≥ the
  floor per contraction, and equals the `configs()` lookup;
- N is monotone over phases until `reset()`, share groups get equal N, and the levels are
  respected;
- TYPICAL targets get p = P/C when `iterations` and `fail_budget` are passed;
- footprints equal the maximum of the C queries;
- MISSED and κ ≥ T levels are handled.

### 4.3 Stage 2 — evidence

**Prerequisite.** Rebuild this worktree's `build/libimmerdink_host.so` (stale, host API 5; see
Provenance), so every stage-2 record names this worktree's library.

**M-a (CPU, now): the combined C2 study.** It merges P1 S0, P3 stage 0 and P4 stage 3a.
- **Harness:** `bench/accuracy/c2_gate.py`, oracle arithmetic only. It compares immerdink's own
  scalings, so it belongs in the product repo. Raw records go to immerdink-eval `results/c1/`.
- **`oracle.kweights`** (spec/oracle.py) is added here, not in stage 4. M-a needs one exact
  reference for the §4.5 rule, the clamp, the L5c guard and the gain G, and until the Q13 gate
  passes that reference is C2's deliverable (Q8). Only its self-tests come with it; its
  certificates, PROOFS L5 and `GuardSound` wait for stage 4 (R2, §5.2).
- **Schemes:**
  - fast;
  - K-eq: plain, deadband ∈ {0, 2, 3, 5}, and guarded (L5c);
  - K-seg, S ∈ {2, 4};
  - K-eq + K-seg;
  - the envelope rule (Appendix B), as the reference for the best partner-aware σ/τ without a
    GEMM;
  - exact-integer SC25, as the reference for "accurate mode as published".
- **Fixtures:** the M5 families; φ ∈ {0.5, 2, 4}; same-sign; `opp_nonsep`; `local_df`; `mixed`;
  and a documented synthetic integral × amplitude family. The synthetic family has
  energy-ordered virtual orbitals, amplitudes g/(ε_c + ε_d − ε_i − ε_j), and DF-like integrals
  with decaying factors.
- **Real operands:** real DF-CCSD operand pairs (Bov, dressed Bvv, t2) if Eric approves a one-off
  FP64 dump (Q14). They run CPU-only, single-threaded, under memcap, outside ehrenfest's test
  windows (busy check, silent mode). Raw dumps and records stay in immerdink-eval `results/c1/`
  or scratch, never in the product repo.
- **Sizes and seeds:** m = n ∈ {16, 128, 512}, k ∈ {600, 3600, 16384}, at least 5 seeds.
- **Reported:**
  - moduli saved at equal max-elementwise and max-normwise error;
  - the change in the **worst** output;
  - stale-weight rates;
  - **the guard**, on every family and in particular on `opp_nonsep`, the synthetic CC family
    and the real operands: its acceptance rate, the gain G, the worst pair's
    log₂(‖a‖‖b‖/‖ã‖‖b̃‖), and the **realized** regressions among guard-accepted cases (the guard
    bounds the error bound, not the error);
  - the moduli saved through the L5d transform, i.e. what `min_moduli` picks for the weighted
    call from the plain statistics, against the measured minimum;
  - predicted TFLOPS-eq and time from `imd_perf_predict` with every cost term: for K-seg, the
    S-fold K0/K5/W writes and the β = 1 read of C; for K-eq, the helper (one or two operand
    reads) and the pre-scaling (16 B per element per changed operand per call unless fused),
    as in §2.6.
- **The CRT precondition is asserted for every output.** The study also settles the deadband
  value and Q11: the deadband is re-tuned at m = n ≥ 128 with at least 5 seeds.

**M-b (CPU, now): model trajectories.** Per-phase TFLOPS-eq and time for square and CC-shaped
cases, schedule totals against fixed N, and the re-pack cost per switch (`predict_pack`).
§2.3–§2.4 already hold most of this.

**M-c (GPU, later): small-N hold-out.**
- About 20 calls on shapes disjoint from the calibration set: sq1024, sq2048, sq4096, a DF shape,
  a CC shape, a skinny shape.
- N = 4–7, with dense and decay data.
- Judged against the §6c.5 targets. If a target is missed: per-class residuals first, then N = 4–7
  calibration cases. No new features (Q24).
- **M-c runs before any schedule gain that uses N ≤ 7 (§2.4) enters DESIGN as a claim.**

**M-d: scheduled iterations** (`bench/accuracy/schedule.py`). Convergence and time are measured
separately. On the drafted small systems a time bar alone cannot discriminate: the ACCURACY §3
Jacobi system is a GEMM view of 1600×100×1600, predicted at t14/t8 = 1.35 and schedule/fixed-14
= 0.79 [M, review]. Against a 0.8 bar that is a coin flip inside the model's p90 error for calls
under 100 µs (10–13 %).
- **Systems:**
  - ACCURACY §3's Jacobi system (v = 40, o = 10, ρ = 0.7);
  - the same with DIIS(8), with and without a history reset at the switches (Q25);
  - a contraction-dominated synthetic fixed point with a PPL-like `ijcd,abcd->ijab` and a
    ring-like term.
- **Schedules:** fixed N ∈ {6, 8, 10, 12, 14}, and forcing-rule ramps (η ∈ {0.1, 0.01}, levels
  {5, 6, 8, 10, 12, 14}, BOUND and TYPICAL with p = P/C).
- **M-d1 (CPU, possible now): convergence.** The bit-exact emulator (checked against
  `oracle.contract`) runs the iterations, so no GPU is needed. Systems at reduced size where
  needed (the PPL-like system at v ≈ 40), single-threaded, under memcap, when ehrenfest is FREE.
  - Metrics: iterations to reach tol ∈ {1e−7, 1e−10}; the final residual in long double; drift
    against a double-double reference iteration (CPU Dot2 contractions) and against the N = 20
    trajectory; for DIIS, iteration counts with and without the reset at each switch.
  - **Success:** the ramp reaches the fixed-N_final residual within 2× and with at most one more
    iteration, on every system, DIIS included.
- **M-d2 (GPU, later): time, on realistic shapes.** The o = 10, v = 96 PPL view
  (100 × 9216 × 9216; predicted schedule/fixed-14 = 0.75 [M, review]), DF (0.66) and 4096³
  (0.67). The Jacobi view (predicted 0.79) is reported but not gated.
  - Metrics: measured TFLOPS-eq and time per phase against the predictions; re-pack time at the
    switches.
  - **Success, per system:** the measured schedule/fixed-14 ratio lies inside that system's own
    prediction band, i.e. its predicted ratio, stated next to the bar, ± the §6c.5 p90 error of
    its call class; and it is ≤ 0.8 on the PPL view, DF and 4096³. The gain against a fixed
    N = 8–9 is reported even if small.

**GPU rules.** Run under `_wip/gpu.lock` (the tools take it), behind the quiet gate, under
memcap, and only when the ehrenfest busy check says FREE (silent mode otherwise). Report
TFLOPS-eq first, with time alongside.

### 4.4 Stage 3 — L7 and the K-segmentation recipe (CPU); the lemma texts L5–L7

**Stage 3 delivers** L7 in PROOFS with its Lean proof `kseg_compose`, `cert_kseg_composition`,
`oracle.contract_segments`, the K-segmentation caller recipe (§2.7, with L7's caveats), and the
DESIGN §7 rows below.

**The other lemmas are proved where they are first relied on** (R2, §5.2):
- L5, its certificates, `GuardSound` and Lean in stage 4;
- L6, its certificates and Lean in stage 5.

Their statements and hand proofs are recorded here now. R-num re-derived L5a/b/c, L6 and L7 by
hand.

New PROOFS lemmas are numbered after L1–L4; Lean goes in `proofs/`, with no new axioms.

**L5 (K-weight cancellation)** **[stage 4; cert: cert_kw_guard; TLA+: OzakiScaling `GuardSound`;
Lean `kweight_cancel`, `kweight_guard`]**

*Setting.* Rows a_i, columns b_j ∈ ℚ^k, and w ∈ ℤ^k. Define ã_ih = a_ih·2^{w_h} and
b̃_hj = b_hj·2^{−w_h}.

- **(a)** Σ_h ã_ih·b̃_hj = Σ_h a_ih·b_hj exactly.
- **(b)** L1–L4 quantify over all rational rows, so they hold verbatim for (ã, b̃):
  - z = Σ q(ã·2^σ̃)·q(b̃·2^τ̃) is exact;
  - D = RN(z·2^{−σ̃−τ̃}) is one rounding of the same x;
  - L3c holds with ‖ã_i‖‖b̃_j‖ and the ℓ1 ratios ν of the *weighted* rows (ν = √k always
    qualifies).
- **(c) Guard.** Let σ_i and σ̃_i be the fast shifts of a_i and ã_i at the same (N, k, rounding),
  and the same for the columns. Let G = min_i(σ̃_i − σ_i) + min_j(τ̃_j − τ_j) over nonzero
  finite rows and columns, and s² = 1 + k·2^{−60}. Then for every such pair
  ‖ã_i‖‖b̃_j‖ < 4s²·2^{−G}·‖a_i‖‖b_j‖, **for any integer G**. If G ≥ 3 (the guard, with
  k ≤ 2^40), then ‖ã_i‖‖b̃_j‖ < ‖a_i‖‖b_j‖.
  - *Proof.* ‖ã_i‖ ≤ d·2^{−σ̃_i} by L1b, and ‖a_i‖ > d·2^{−σ_i}/(2s) by L1d, both at the same d.
    So ‖ã_i‖/‖a_i‖ < 2s·2^{−(σ̃_i − σ_i)}, the same for the columns, and the product ratio is
    < 4s²·2^{−G}. At G ≥ 3 that is < s²/2 < 1, since s² ≤ 1 + 2^{−20}. ∎
  - *Margin 2* gives only a ratio < s² ≤ 1 + k·2^{−60}. It can exceed 1 only through S's ceiling
    slack, which is where s comes from (L1d step 2). So 3 is the smallest margin the proof
    supports; whether 2 fails in practice is not needed.
  - The conclusion is a statement about real norms, so it holds at every N once established at
    any N: one guard serves every phase. When G ≥ 3, every §6f bound computed from the
    **plain** operands' norm statistics, with ν = √k, is a valid bound for the weighted call,
    and NORMWISE keeps its meaning.
  - **ℓ1 ratios do not transfer** (R-api must-fix 2). Weighting flattens rows: on decay_opp, ν
    goes from about 1 to about √k. The guard bounds only the product ‖·‖‖·‖, and at G = 3 the
    factor 4s²·2^{−3} does not cover a ν ratio of up to √k. So the weighted call's request must
    use `l1_ratio` = 0 (ν = √k) or ratios measured on the **weighted** rows, never the plain
    rows' ratios.
  - The guard bounds the error *bound*, not the realized error ("never worse in the §3.8
    bound").
  - Evidence [C, review]: in exact arithmetic, margin 3 had 0 violations (65 accepted cases),
    and at margin 2 none was observed (13 cases).
- **(d) Transfer to the request** (R-num should-fix 1). From (c), with any reported G, the plain
  statistics give a rigorous request for the weighted call (ν = √k, or measured on the weighted
  rows):
  - ELEMENTWISE: cond′ = max(1, cond·4s²·2^{−G}). ρ̃_ij ≥ 1 by Cauchy–Schwarz, so the max keeps
    it a bound.
  - ABSOLUTE: norm_product′ = norm_product·4s²·2^{−G}.
  - NORMWISE relative to the **plain** norms: meeting target t needs the weighted call's
    NORMWISE bound ≤ t′ = t·2^G/(4s²). Then
    |D − x| ≤ bound·‖ã_i‖‖b̃_j‖ < t·‖a_i‖‖b_j‖.
  - *FP evaluation*, for k ≤ 2^40 (so 4s² ≤ 4(1 + 2^{−20})): ldexp by 2 − G or G − 2 is exact
    (absent overflow or underflow). Then one rounded multiply follows: by (1 + 2^{−19}) for
    cond′ and norm_product′, or by (1 − 2^{−19}) for t′. One RN error of u is absorbed:
    (1 + 2^{−19})(1 − u) > 1 + 2^{−20}, and (1 − 2^{−19})(1 + u) < 1 − 2^{−20} ≤ 1/s².
  - Without the transform, `imd_err_min_moduli` and rank under BOUND keep choosing the
    unweighted N: the plain statistics stay valid but pessimistic, and §2.6's gains cannot be
    reached through auto-N. Stage 4 ships the transform as `imd_err_request_kweighted`
    (§4.5).

**L6 (a-posteriori certificate)** **[stage 5; cert: cert_apost, cert_stats_constants,
cert_stats_bin; TLA+: OzakiScaling `CertSound`; Lean `apost_cert`]**

*Setting.* As L3, possibly with weighted operands (L5b). Take δ = ½ (RN) or 1 (TRUNC),
d = T − κ/2 (RN) or T (TRUNC), and κ = ⌈√k⌉. Define:
- B̄ = ⌈2δκd + kδ²⌉: for RN ⌈κT − κ²/2 + k/4⌉, for TRUNC 2κT + k;
- X = ⌈d²⌉: for RN T² − κT + ⌈κ²/4⌉, for TRUNC T².

The claims:
- **(a)** |z − x·2^{σ+τ}| ≤ B̄.
  - *Proof.* Multiply L3a by 2^{σ+τ}: the left side is ≤ δ(‖a·2^σ‖₁ + ‖b·2^τ‖₁) + kδ². Use
    ‖v‖₁ ≤ √k‖v‖ ≤ κ‖v‖, and L1b's ‖a·2^σ‖ ≤ d (which bounds the scaled row before
    quantization). Neither L1d nor s is needed. ∎
- **(b)** If |z| > B̄, let Y = |z| − B̄. Then ρ_ij = ‖a_i‖‖b_j‖/|x_ij| ≤ X/Y. If moreover
  y = z·2^{−σ−τ} is in the normal range (2^{−1022} ≤ |y|) and D = RN(y) is finite, then
  |D − x| ≤ (u + (1+u)·B̄/Y)·|x|.
  - *Proof.* |x|·2^{σ+τ} ≥ Y, ‖a_i‖‖b_j‖·2^{σ+τ} ≤ d² ≤ X, and |y − x| ≤ B̄·2^{−σ−τ} ≤ B̄|x|/Y.
    Then apply L3c step 2, which needs the normal range (DESIGN §6f.2: D must not underflow). ∎
  - The ρ bound is a property of the exact integers and holds whatever the range of y. The
    error bound does not: outputs with |y| < 2^{−1022} (gradual underflow, where no relative
    bound exists) or with D = ±Inf are **out of range**. Stage 5 counts them separately and
    never certifies them (§4.6).
- **(c)** With r = min{r ≥ 0 : Y·2^r ≥ X}, ρ_ij ≤ 2^r, and for an in-range output the
  elementwise error is ≤ u + (1+u)·B̄·2^r/X.

*Looseness.* Before binning, ρ̂/ρ < 4s²·(⌈d²⌉/d²)·(|z| + B̄)/(|z| − B̄). The factor ⌈d²⌉/d² is
negligible except at tiny d. Binning to a power of two adds up to another factor 2. The ratio
is unbounded as |z| → B̄⁺. Per-row ℓ1 sums of |a′| from K1 could replace κ in (a) by ν (an
option, not adopted).

*Evidence* [C, review]: L6a/b/c had 0 violations over 1500 random and adversarial row/column
pairs (1305 certified; max |z − x·2^{σ+τ}|/B̄ = 0.40), at N ∈ {3, 4, 5, 8, 14, 20}, RN and
TRUNC, exact and single-threaded.

*Widths.* bitlen(X) ≤ 155 (N = 20), so r ∈ [0, 155]. For RN, κ < T (acceptance) gives
B̄ < T² ≤ H, so B̄, X and Y fit L32(N) limbs. For TRUNC, B̄ can exceed H once κ ≳ T/2; every
output is then uncertified, which a plan constant records.

**L7 (segmented accumulation)** **[stage 3; cert: cert_kseg_composition; Lean `kseg_compose`]**

*Setting.* K is partitioned into S segments. D_s is the emulated product of segment s (α = 1,
β = 0); C_1 = D_1 and C_s = RN(C_{s−1} + D_s), as with β = 1 at α = 1 (§3.7: C ← fma(1, C, E)).
For output (i, j) let:
- n_s = ‖a_i^s‖‖b_j^s‖;
- E_s = L3c's E at (N, k_s, rounding, ν^s);
- γ = γ_{S−1} = (S−1)u/(1 − (S−1)u).

Assume finite inputs and no overflow or underflow.

- **Claim.** |C_S − x| ≤ Σ_s n_s·W_s, with W_s = u + (1+u)E_s + γ(1+u)(1+E_s).
  - *Proof.* L3c gives |D_s − x_s| ≤ (u + (1+u)E_s)·n_s, using |x_s| ≤ n_s. Recursive summation
    gives |C_S − ΣD_s| ≤ γ·Σ|D_s|, and |D_s| ≤ (1+u)(1+E_s)·n_s. Combine with the triangle
    inequality. ∎
- **NORMWISE and ABSOLUTE.** These are relative to N_S = Σ_s n_s, and N_S ≤ ‖a‖‖b‖ by
  Cauchy–Schwarz over the segments.
  - Against the unsegmented bound **with ν = √k**: E_s ≤ E(k), since ν_s ≤ √k_s ≤ √k,
    κ_s ≤ κ and k_s ≤ k. So the segmented bound never exceeds that one by more than the u-level
    term γ(1+u)(1+E)·‖a‖‖b‖.
  - A whole-row ν < √k supplied by the caller does **not** bound a segment's ν_s. For
    a = (1, ε, …), the whole-row ratio is near 1, but a segment without the first entry has a
    ratio near √k_s. So compare only with the ν = √k bound, or measure ν per segment.
  - ABSOLUTE gains (1+γ)·S·2^{−1075} for gradual underflow.
- **ELEMENTWISE.** The bound is W·ρ_S with ρ_S ≥ N_S/|x|. At high N the γ·ρ_S term dominates, so
  it is **worse** than the unsegmented u + Eρ(1+u). Segmentation gives up the single final
  rounding.

**Certificates** (spec/params.py, real widths, every N = 2..20, RN and TRUNC), by stage:
- Stage 0: `cert_footprint_monotone`. It covers F1, F2 and the internal facts I1 (including
  the drafted `LayoutPrefix`: equal sig/flag offsets across N, res(N′) inside res(N_max)) and
  I2.
- Stage 1: `cert_rho_recipe`. The §4.2 FP evaluation gives ρ̂ ≥ ρ in exact rationals, on random
  rows and on rows with |D| just above U·n.
- Stage 3: `cert_kseg_composition`. L7 against exact rationals, with rows from every M5 family
  plus constant rows; S ∈ {1, 2, 3, 4}; N ∈ {3, 4, 5, 8, 14, 20}; RN and TRUNC; per-segment ν;
  every metric; plus the ν counterexample (a = (1, ε, …)).
- Stage 4:
  - `cert_kw_rule`: the §4.5 integer rule against an independent brute force on random Δ
    vectors (ties, even and odd counts, deadband edges, floor of negative values, both clamps,
    the Q9 index order with repeated and one-sided labels).
  - `cert_kw_fp64_exact`: clamped weights make every scaled entry exact in FP64, on edge
    exponents including subnormals. R-num re-derived the four clamp bounds.
  - `cert_kw_guard`: L5c on random and adversarial rows in exact rationals:
    ‖ã_i‖‖b̃_j‖ < 4s²·2^{−G}·‖a_i‖‖b_j‖ for every pair at the reported G, and the strict
    inequality whenever G ≥ 3. It also checks L5d's FP evaluation: cond′ and norm_product′ at
    least, and t′ at most, the exact values. It does not search for a margin-2 witness; L5c
    states analytically why margin 2 is not enough for the proof.
- Stage 5:
  - `cert_stats_constants`: X and B̄ are exact ceilings; their widths fit L32(N) for RN; the
    TRUNC "uncertified" flag; r ≤ 155; k ≤ 2^40.
  - `cert_stats_bin`: the K5 bin rule (bitlen(X) − bitlen(Y), clamped at 0, then one exact
    shift-compare adjust) equals min{r ≥ 0 : Y·2^r ≥ X}, at the edges Y = 1, Y = X, X a power
    of two, and Y = X ± 1, for every (N, k) edge.
  - `cert_apost`: L6 against `emulate_element` on random, constant, adversarial and weighted
    rows (ρ ≤ 2^r, error ≤ certificate for in-range outputs, out-of-range ones flagged), with
    counterexamples for dropping kδ² or using ⌊√k⌋.
- Stage 6: `cert_kw_shift_range` and reruns (§4.7).

**Oracle** (spec/oracle.py), by stage:
- Stage 2: `kweights(A, B, plan, params)`, returning (w, report) in exact integers, with the
  report's G (§4.3 M-a).
- Stage 3: `contract_segments(A, B, C, N, rounding, bounds)`, a literal composition of
  `contract` with β = 1.
- Stage 5: `apost_bin(z, sigma, tau, plan)`, and `stats(view, ...)`, the integer fields of
  `imd_stats_t`.
- Self-tests cover the CRT assertion on every fixture, invariance of w under joint K
  permutation, and determinism.

**DESIGN §7 rows** (immerdink data only):
- "Accurate mode": still rejected. By an estimate from the §3.8 bound, confirmed by the data of
  §2.5–§2.6 (not a theorem), partner-aware σ/τ shifts realize about G/7.8 moduli out of a G-bit
  headroom. They saved ≤ 0.3 moduli on dense data and about 3 on separable anti-correlated data,
  need one more operand pass (losing KF/KL group-1 fusion) and break D1. K-equilibration turns
  the same headroom into about G/3.9 moduli, with no pass inside the library (§2.8). The
  envelope rule (Appendix B) is recorded as the proven partner-aware option.
- New rows: N-range plan (A2), multi-N and RO packs (A3, A4), device-chosen N (B4), slices (C1,
  parked).

### 4.5 Stage 4 — `imd_kweights` and caller-side equilibration (C2, step 1)

**The idea.** The caller stores pre-scaled operands, ã = a·2^w along K and b̃ = b·2^{−w}, and
contracts them with ordinary plans and packs. The library's numerics do not change; C is
bit-exact against `oracle.contract(ã, b̃, ...)`. immerdink supplies an exact, deterministic,
capture-safe helper that picks w, reports the L5c guard and the gain G, and a host function
that turns plain statistics into the weighted call's request (L5d).

**Index order** (Q9, both reviewers' rules merged). w is indexed over the **shared K labels**:
the plan's contracted labels that both A and B carry.
- Labels are ordered by their first occurrence in A's descriptor mode order, first label
  fastest (column-major). A repeated (diagonal) label counts once, at its first occurrence.
- The length k_s is the product of the shared labels' extents. k_s = k when every K label is
  shared. An extent-1 label contributes a factor 1 and changes nothing.
- A K label carried by only one operand (DESIGN §5 step 6: stride 0 in the other) is not
  indexed: w is constant along it, i.e. it gets w ≡ 0. Scaling it on one side would need the
  other operand materialized along a label it does not carry.
- The order does not depend on the partner or on C's strides, unlike the plan's canonical k̄
  (§5 step 8). Stage 6 must convert it (§4.7, open item).

**Rule** (integers only; one vector shared by every batch λ, Q12):
1. **Exponent extrema.** For each shared index h: c_A(h) and m_A(h) are the max and min of
   ilogb|a| over the finite nonzero entries of A at h (all λ, rows and one-sided K labels);
   c_B(h) and m_B(h) the same for B (all λ, columns and one-sided K labels). An index with no
   such entry on one side is undefined.
2. **Δ and median.** Δ_h = c_B(h) − c_A(h) where both are defined; Δ ∈ [−2097, 2097]. med = the
   lower median of the defined Δ, from a one-CTA histogram.
3. **Raw weight.** w0_h = 0 if Δ_h is undefined or |Δ_h − med| ≤ deadband; otherwise
   ⌊(Δ_h − med)/2⌋.
4. **Clamp, always.** w_h = w0_h clamped to [−max_abs, max_abs] (max_abs ≤ 127) **and** to the
   interval where scaling both operands is exact in FP64. The FP64-exact clamp is
   unconditional (R-num must-fix 1); there is no flag to turn it off. A side with no finite
   nonzero entry at h imposes no bound.
   - w > 0 (A scaled up, B down): w ≤ 1023 − c_A(h) (no overflow) and w ≤ m_B(h) + 1022
     (no B entry drops below the normal range).
   - w < 0 (A down, B up): w ≥ −1022 − m_A(h) and w ≥ c_B(h) − 1023.

   0 is always feasible, so the clamp is the nearest feasible value to w0_h. A constant offset in
   w has no effect (σ is scale-equivariant), which is why the rule centres on the median.
   - *Why unconditional.* Without the clamp, the caller's ldexp(a, w) / ldexp(b, −w) can round
     (an entry pushed below 2^−1022) or overflow. Three claims would then fail silently: L5a
     (Σ ã·b̃ = Σ a·b), "Exactness for the caller" below, and "flagged outputs are unchanged".
     The helper could also report guard = 1 for an exact ã that the caller cannot build in FP64.
5. **Guard and gain** (`IMD_KW_GUARD`). With w fixed, a second pass computes the §3.2 shifts of
   every nonzero finite row of A and of ã (and of the columns of B and of b̃) at the plan's
   (N, k, rounding). It works from the integers (m, e + w), so ã is never materialized. It
   reports gain_row_min = min_i(σ̃_i − σ_i), gain_col_min, their sum G (`gain`), and
   guard = 1 iff G ≥ 3 (L5c). If A has no nonzero finite row or B no nonzero finite column, G is
   undefined. Then guard = 1 vacuously, since every product is zero or flagged, and
   gain = INT32_MIN; the plain request applies unchanged.

**Exactness for the caller.** With the clamp, every scaled entry is representable, so the
caller's pre-scaling is exact. Non-finite entries stay non-finite (Inf·2^w = Inf), and no finite
entry becomes non-finite, so the same rows and columns are flagged. The native path for flagged
outputs computes the same real products a·b, hence the same roundings, so its result is
unchanged.

**API** (host API +1):

```c
typedef struct {                 /* 24 B; zero-tail rule (§4.9)                               */
  uint32_t struct_size;
  int32_t  deadband;             /* >= 0 (default 3)                                          */
  int32_t  max_abs;              /* 1..127 (default 127)                                      */
  int32_t  flags;                /* IMD_KW_GUARD = 1; other bits: INVALID_VALUE. The FP64-exact
                                    clamp is always applied (no flag).                       */
  int64_t  reserved;             /* 0 */
} imd_kweights_params_t;
typedef struct {                 /* 32 B, written by the call into caller device memory      */
  int32_t guard;                 /* 1: G >= 3, so L5c holds for every nonzero row/column pair;
                                    0: not; -1: not computed                                  */
  int32_t gain;                  /* G = gain_row_min + gain_col_min (L5c/d); INT32_MIN if not
                                    computed (never 0: G = 0 is a real value)                  */
  int32_t gain_row_min, gain_col_min;  /* min (sigma~ - sigma) over nonzero finite rows of A / columns of B */
  int32_t median;                /* med, INT32_MIN if no index has both exponents            */
  int32_t num_weighted, max_abs_w, clamped;
} imd_kweights_report_t;
/* libimmerdink_host: exact workspace bytes, and k_s (GPU-free) */
imd_status_t imd_kweights_ws_bytes(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                                   const imd_options_t* options, const imd_kweights_params_t* params,
                                   size_t* bytes, int64_t* k_s);
/* libimmerdink: w for this plan's contraction from the FP64 operands */
imd_status_t imd_kweights(imd_plan_t plan, const double* A, const double* B,
                          const imd_kweights_params_t* params, int8_t* kw /* k_s entries, Q9 order */,
                          imd_kweights_report_t* report /* device, may be NULL */,
                          void* ws, size_t ws_bytes, cudaStream_t stream);
/* libimmerdink_host: the weighted call's rigorous request from the plain operands' statistics
   (L5d), GPU-free. ELEMENTWISE: cond' = max(1, RN(ldexp(cond, 2 - gain) (1 + 2^-19)));
   ABSOLUTE: norm_product' the same without the max; NORMWISE: target' = RN(ldexp(target,
   gain - 2) (1 - 2^-19)); l1_ratio_a = l1_ratio_b := 0 (sqrt k), since plain-row ratios do not
   bound the weighted rows. Everything else is copied. Valid for k <= 2^40 (the range DESIGN
   §3.2 certifies) and any reported G; a negative G makes the request stricter (cond' grows,
   target' shrinks). INVALID_VALUE for an invalid request (as imd_err_estimate),
   gain = INT32_MIN (not computed), or an over/underflowing result. */
imd_status_t imd_err_request_kweighted(const imd_err_request_t* plain, int32_t gain,
                                       imd_err_request_t* weighted);
```

- **Plan.** Any EMULATE plan of the contraction works; it supplies the lowering, the scatter
  vectors and (N, k, rounding) for the guard. It reads the FP64 pointers passed, and its packs are
  ignored. DIRECT, K-empty, k ≤ 1 and k_s ≤ 1 plans return NOT_SUPPORTED.
- **Checks** are §6b.5-style: NULL, ALIASING (kw, report and ws against A, B and each other),
  MISALIGNED, INSUFFICIENT_WORKSPACE. `params` follows the zero-tail rule (§4.9).
- **Workspace:** the vec region (GENERATE) plus 16·k_s bytes of extrema plus a few words,
  aligned as in §4a, and reported exactly by `imd_kweights_ws_bytes`
  **[cert: cert_workspace extended]**.
- **Launches:**
  1. zeroing, every call (the workspace is reused);
  2. extrema, with per-CTA aggregation and integer atomicMax/atomicMin;
  3. one-CTA median and weights;
  4. optionally the guard pass.

  The output is bitwise deterministic. The helper is capture-safe: kernels only, on the caller's
  stream, with no allocation or sync.
- **Cost** [E]: one read of A and B without the guard, two with it. At 4096³ that is about 0.3 /
  0.6 ms, about 6 / 13 % of an N = 14 call. It is paid once per magnitude profile, or per call
  if it is re-run on operands that change every iteration (§2.6).
- **Perf model:** no phase. The helper is measured in M-e and counted in the Q13 gate.
- **Python:** `Plan.kweights(A, B, deadband=3, max_abs=127, guard=True, workspace=, stream=)`,
  returning device views of w and the report; `host.kweights_ws_bytes`;
  `host.Accuracy.kweighted(gain)`, which calls the C function and returns a new immutable,
  hashable Accuracy (`planner.Accuracy` is `host.Accuracy`, planner.py:62), so planner.py still
  computes nothing.
- **Accuracy API: no change to `imd_err_request_t` or to the estimate.** The weighted call's
  request is built in one of two ways:
  - from the plain operands' statistics through `imd_err_request_kweighted` (L5d), which
    applies the gain and resets the ℓ1 ratios to √k; or
  - from statistics measured on the **weighted** operands, ℓ1 ratios included.

  A caller must **not** pass the plain rows' `l1_ratio_a/b` with a weighted call. Weighting
  flattens rows (on decay_opp ν goes from about 1 to about √k), and the guard does not cover
  the ν ratio (L5c, R-api must-fix 2). Without the transform, plain-norm statistics with
  ν = √k remain valid bounds when guard = 1, but they are pessimistic: min_moduli keeps the
  unweighted N.

**Caller recipe** (documented, with an example in `bench/accuracy/`):
1. Run `imd_kweights` on a representative pair.
2. If guard = 1, or the caller accepts the risk, scale A's K-entries by 2^w and B's by 2^{−w},
   fused into the caller's own producer kernel where possible (otherwise 16 B per element per
   changed operand, §2.6). Use the Q9 index order: w depends only on the shared K labels.
3. Build the weighted request with `imd_err_request_kweighted(plain, report.gain)` and choose N
   with `min_moduli` / `PrecisionPlan` as usual.
4. Contract and pack the scaled tensors as usual.

The caller owns consistency: every partner of a pre-scaled packed operand must carry the same w.
A mismatch is a silently wrong product, documented as a caller rule. The reported G is valid for
the pair the helper ran on. For new data it is a prediction, unless the helper is re-run.

**Spec.**
- New small `DeviceReduce.tla`, run under memcap with bounded parallelism. It models one call's
  zeroing, per-CTA shared aggregation, global integer atomics, and a stream-ordered consumer
  kernel, over toy CTAs, threads and calls.
  - Invariants: `ExactExtrema` at the median kernel's start; `WeightsOfFinal`.
  - Negatives: `neg_kw_no_zero` (a previous call's extrema leak through the reused workspace),
    `neg_kw_rmw` (a non-atomic max loses updates), `neg_kw_median_early` (the median kernel runs
    on another stream without an event).
  - The same module gets the stage-5 statistics configurations.
- **OzakiScaling `GuardSound`** (R-num should-fix 4), toy-exhaustive. Over the toy enumeration
  of rows, columns and weight vectors at toy widths, margin 3 implies a strict decrease of the
  norm product, and ‖ã‖‖b̃‖ < 4s²·2^{−G}·‖a‖‖b‖ at every G. guard = 1 is a soundness claim the
  library reports, so this lemma follows the project pattern for lemmas: TLA+ toy + params +
  oracle + Lean.
- PROOFS L5 with its Lean proofs `kweight_cancel` and `kweight_guard`, and the stage-4
  certificates of §4.4. `oracle.kweights` already exists from stage 2.

**Tests.**
- `host_kweights`: workspace bytes and k_s against params, the Q9 order (repeated labels,
  one-sided K labels, extent-1 labels), and validation (struct_size, unknown flags, deadband < 0,
  max_abs ∉ 1..127, reserved ≠ 0, a nonzero tail beyond the known size).
- `host_accuracy`: `imd_err_request_kweighted` against params' exact transform for every metric,
  with ℓ1 ratios reset, and its INVALID_VALUE cases.
- GPU `kweights_helper`: w and the report bit-equal to the oracle on the M5 families, random Δ,
  ties, all-zero K indices, non-finite entries, all-equal Δ, subnormals at the clamp edges, k = 2,
  batches. Also determinism (repeats, two streams), capture/replay, and compute-sanitizer
  memcheck, racecheck and initcheck.
- `contract_kw_prescaled`: scaled contractions bit-exact against `oracle.contract(ã, b̃)`,
  including flagged rows.
- `bench/accuracy`: when guard = 1, the measured error stays within the unweighted bound, and
  within the bound of the L5d-transformed request at the N it selects.

**Measurement M-e (GPU).** On decay-like data at realistic sizes (4096³, 8192³, DF):
- helper time, with and without the guard;
- the pre-scaling pass, unfused, per changed operand;
- TFLOPS-eq and time at equal elementwise accuracy against fast mode at its higher N, **with
  every recurring cost included**: the helper amortized over the profile's lifetime (or per call
  if it is re-run), the unfused pre-scaling of operands that change per iteration, and the
  memory of scaled copies.

**Risks.**
- The value depends on the data: none on dense data.
- Weights that go stale on random-structure data cost up to 2 moduli. The guard detects this only
  when it is re-run on the current pair.
- Pre-scaling a tensor that is also contracted over other indices (e.g. t2 in PPL and ring) needs
  a scaled copy per weight vector, which costs memory. Stage 6 removes the copies.

### 4.6 Stage 5 — K5 statistics and the a-posteriori certificate (C1(b) on the device)

**Gate:** a consumer with elementwise targets (Q16). Until then the stage-1 recipe covers the
need.

**Per-call struct and entry point** (host API +1):

```c
typedef struct {                 /* 32 B; zero-tail rule (§4.9)                               */
  uint32_t struct_size;
  int32_t  reserved0;            /* 0 (reserved: a per-call N for a revived range plan, §3.2)  */
  void*    stats;                /* NULL or imd_stats_t* in device memory, 8-aligned           */
  int64_t  reserved1;            /* 0 (reserved: stage 6 K-weights pointer)                     */
  int64_t  reserved2;            /* 0 */
} imd_call_t;
imd_status_t imd_contract_ex(imd_plan_t plan, double alpha, const imd_operand_t* A,
                             const imd_operand_t* B, double beta, double* C, void* ws,
                             size_t ws_bytes, const imd_call_t* call /* NULL: as imd_contract_packed */,
                             cudaStream_t stream);
```

- It follows `imd_contract_packed`'s check order (include/immerdink/immerdink.h:101-115).
- Additions: `call->struct_size` < 32, reserved ≠ 0, or a nonzero tail beyond the known size →
  INVALID_VALUE; ALIASING also covers `stats`; MISALIGNED if `stats` is not 8-aligned.
- No numerics check happens at contract time.

**Output** (caller-owned; zeroed by the caller, e.g. with `cudaMemsetAsync`, which is
capture-safe, and re-zeroed before each graph replay that should start afresh; only accumulated
by immerdink). A zeroed buffer is the identity of every field's operation (add, OR, max):

```c
#define IMD_STATS_RHO_BINS 156
typedef struct {                       /* 1328 B */
  uint64_t rho[IMD_STATS_RHO_BINS];   /* unflagged outputs with |z| > Bbar, by r: rho_ij <= 2^r (L6c) */
  uint64_t uncertified;               /* unflagged outputs with 0 < |z| <= Bbar                    */
  uint64_t zero;                      /* unflagged outputs with z = 0                              */
  uint64_t out_of_range;              /* outputs also counted in rho[] whose |y| < 2^-1022 or
                                         D = +-Inf: rho bound valid, error bound not (L6b)          */
  uint64_t flagged;                   /* outputs of the native path                                */
  uint64_t calls;                     /* +1 per emulated call                                      */
  uint64_t other_calls;               /* +1 per DIRECT / SCALE / alpha = 0 call                    */
  uint64_t k_max;                     /* atomicMax of the logical k over emulated calls           */
  uint64_t k_min_c;                   /* atomicMax of (UINT64_MAX - k): k_min = UINT64_MAX - k_min_c */
  uint32_t n_mask, rounding_mask;     /* OR of 1 << N / 1 << rounding over emulated calls          */
  uint64_t reserved;
} imd_stats_t;
```

- **The hook** sits in K5's per-output code after `crt_finish` (contract.cu:515) and in
  `k5_crt_bfast` (contract.cu:584), for unflagged outputs only. It already has |z| as limbs,
  hence bitlen|z|, and σ + τ (contract.cu:515-516). Per output it does:
  - one multi-limb compare/subtract of B̄;
  - r = bitlen(X) − bitlen(Y), clamped at 0, then one exact shift-compare to adjust by one
    (`cert_stats_bin`);
  - the range test: |y| < 2^−1022 iff bitlen|z| ≤ σ + τ − 1022, exact on integers; D = ±Inf
    is tested on the rounded D;
  - a warp-aggregated (`__match_any_sync`) shared-memory histogram.

  Then, per CTA, one `atomicAdd(u64)` per non-empty bin and counter. One thread per call does
  the `calls`, `k_max`, `k_min_c` and mask updates.
- **Constants.** X(N, k) and B̄(N, k) are plan constants passed by value in `OutArgs`.
- **Other paths.** DIRECT, SCALE and α = 0 calls only increment `other_calls`.
- **Determinism.** Integer atomics only, so the result is identical under any schedule.
- **Accumulation.** Concurrent calls may accumulate into one buffer by design. Accumulated calls
  count **(output, call) pairs**: max_miss below is per pair. The uncertified outputs of one
  low-N call can leave no r* at all (§2.10).
- **What a bin means.** Each entry in bin r upper-bounds ρ_ij of that output by 2^r, for any
  N; how tight the bound is depends on N (L6 Looseness). So calls at different N can be
  accumulated for `cond`, but the histogram is not a property of the data alone. The per-call
  certificate needs one (N, rounding, k) per buffer.
- **Cost bar.** With `stats` NULL, no measurable change in time, **and** no change in K5's
  register count or occupancy, in every variant (E = 1, 2, 4, bfast) and at every N. A uniform
  `stats != NULL` branch costs no time when it is not taken, but it can still raise register
  allocation, and K5 runs at 82–96 % of its bound (ROADMAP A4). With stats on: at most 1 % of a
  4096³ call and 2 % of a DF call (K5 is 5–16 % of a call, §2.3). If the bar fails, use a K5
  template variant and the optional calibration key `k5.stats_op` (Q20).

**Accuracy API: one pure host function; `imd_err_request_t` is unchanged** (R3, §5.2,
R-api should-fix 3).

```c
enum { IMD_STATS_ZERO_EXEMPT = 1 };
/* cond from K5 statistics (L6b/c), GPU-free: cond := 2^{r*}, r* the least r with
   sum_{r' > r} rho[r'] + uncertified + (flags & ZERO_EXEMPT ? 0 : zero) <= max_miss,
   counted over (output, call) pairs, across any N. out_of_range outputs stay in their bins:
   the rho bound holds for them. NOT_SUPPORTED if no r* exists (nothing certified) or no
   emulated call was recorded (calls = 0); INVALID_VALUE for other flag bits. */
imd_status_t imd_stats_cond(const imd_stats_t* stats, uint64_t max_miss, int32_t flags, double* cond);
```

- The caller passes the returned `cond` in an **unchanged v8 request**. The rigor is the same as
  a request flag would give. Accuracy stays hashable by value, `Memo` keys and rank/predict are
  untouched, and the request's `reserved` word stays free for a later flag.
- Since no Accuracy object carries the statistics, no snapshot by value is needed for `Memo`
  to stay correct. (A design that put stats into Accuracy would have to copy them by value.)
- **Rigor.** With BOUND, the N returned meets the target on the source calls' data for every
  (output, call) pair except at most max_miss. For the next iteration's data it is a
  prediction.

```c
/* The a-posteriori elementwise certificate (L6c) over the calls accumulated in stats, for
   unflagged in-range outputs: the max over the bins, after skipping the max_miss' highest-r
   outputs, of u + (1+u) Bbar 2^r / X, inflated by (1 + 2^-40) as in §6f. Here
   max_miss' = max_miss - (uncertified + out_of_range + (ZERO_EXEMPT ? 0 : zero)), and
   NOT_SUPPORTED if that is negative. Skipping by count is conservative, since some skipped
   outputs may be out-of-range ones already counted. INVALID_VALUE unless calls >= 1,
   other_calls == 0, n_mask == 1u << N, rounding_mask == 1u << rounding and
   k_max == k_min == k: the k the caller passes is validated, never trusted. */
imd_status_t imd_err_posteriori(int32_t num_moduli, int32_t rounding, int64_t k, const imd_stats_t* stats,
                                uint64_t max_miss, int32_t flags, imd_err_t* out);
```

- Flagged outputs are outside the certificate; they have the native path's bound (§3.7).
- Out-of-range outputs are never certified: no relative bound exists under gradual underflow.
  The 2^−1075 absolute term of ABSOLUTE is not a relative bound.

**Not included: a norm statistic for ABSOLUTE** (Q18).
- P1's `np_key`, a max over outputs with z ≠ 0, is unsound: a nonzero row orthogonal to every
  column at this N is left out.
- Including zero rows is sound but useless, because σ := 0 inflates the maximum by up to about T.
- So ABSOLUTE keeps the caller's norms.

**Python.** `Plan.contract(..., stats=dev_buf)`; `host.stats_cond(host_view, max_miss=0,
zero_exempt=False) -> float`, whose result goes into an ordinary
`planner.Accuracy(metric="elementwise", target=, cond=c)`; `host.err_posteriori(...)`. The
156-bin scan runs once per statistics snapshot, i.e. once per phase decision, not in every
`min_moduli` call.

**Spec.**
- `DeviceReduce.tla`, statistics configurations: two calls on two streams share one caller
  buffer; invariant `FinalCountsExact` (the sum of both calls) for every interleaving.
  - `pos_stats_two_calls`.
  - `neg_stats_flush_before_barrier`, `neg_stats_global_rmw`.
  - `neg_stats_lib_zeroes`: the library zeroing per call races with a concurrent call. This is
    why the caller zeroes.
- HostAPI: `pos_stats_shared` (one plan, two threads, one stats buffer; the plan stays
  read-only).
- IndexMap: the stats argument checks in the §6b.5 order (`neg_stats_alias`).
- OzakiScaling JointSpec: `CertSound` (L6a/b on the toy enumeration, the in-range condition
  included), with negatives `neg_cert_no_kdelta2` and `neg_cert_floor_kappa`.
- PROOFS L6 with its Lean proof `apost_cert`; the stage-5 certificates of §4.4
  (`cert_stats_constants`, `cert_stats_bin`, `cert_apost`); `oracle.stats` and
  `oracle.apost_bin`.

**Tests.**
- GPU: the histogram and counters bit-equal to `oracle.stats` in every K5 variant (E = 1, 2, 4,
  bfast) and at every N.
  - Edges: |z| = B̄, B̄ + 1, z = 0, r = 0 and r = max; |y| just below and at 2^−1022; D
    rounding to ±Inf.
  - Accumulation across calls and N; two streams on one buffer; capture/replay with re-zeroing;
    DIRECT, SCALE and α = 0 calls (`other_calls`); `k_max` / `k_min_c` across calls of two k;
    ALIASING, MISALIGNED and a nonzero `imd_call_t` tail.
  - The cost bar (A/B), with register count and occupancy from the compiler and `ncu` for
    `stats` NULL in every variant and at every N.
- `host_accuracy`:
  - `imd_stats_cond` equals a brute force, with and without ZERO_EXEMPT; NOT_SUPPORTED when
    nothing is certified;
  - `imd_err_posteriori` matches params, and rejects mixed N, rounding or k, `other_calls` ≠ 0,
    and a k argument that differs from the recorded one.
- `bench/accuracy`: the measured error is ≤ the certificate on every in-range output.

**Risks.**
- The statistics lag by design: the next iteration's data differ.
- A source call at too low an N certifies nothing; bootstrap at the NORMWISE N for the target.
- The K5 overhead is unmeasured, register pressure included.

### 4.7 Stage 6 — library-internal weights and weighted packs (C2, step 2)

**Gate:** M-e (stage 4) measured, plus real-operand evidence (Q13, Q14). The open layout item
below must also be designed before the stage-6 build starts.

**Open design item: from caller-order weights to the kernels' k̄** (R-api should-fix 7). It
blocks the stage-6 build.
- w arrives in the Q9 order (shared K labels, A's mode order). The scaling and pack kernels
  index K in the plan's canonical k̄, which depends on the partner and on C's strides (§5
  step 8), and they need w per k̄ position, constant along one-sided labels.
- "Stage 6 converts internally" therefore needs either a new workspace region for the
  k̄-ordered weights (k̂ bytes), or extra vec entries mapping k̄ → Q9 index.
- The region count is fixed by the header contract: `imd_workspace_layout` returns
  `IMD_REGION_COUNT` regions (include/immerdink/immerdink_host.h:149-151), and
  tests/test_host_c.c:59 checks it. This is the same problem Appendix A #18 raises for P3's
  regions.
- Candidates, to be decided with a TLA+/params plan of their own:
  - (i) extend the vec region with a k̄-ordered weight table of k̂ bytes, written per call by
    the vec kernel (K0) under GENERATE. There is no new region, but vec grows.
  - (ii) a new region. That changes `IMD_REGION_COUNT` and needs a new layout entry point.
  - (iii) the caller supplies w in k̄ order through a host query exposing the plan's K order.
    This is Q9 (b), rejected for stage 4 because that order depends on the partner and on C.

**Kernels.**
- K1, k1_tiled, KF, KL, K2/K3 and `imd_pack` add w (A side) or −w (B side) to the element
  exponent. They do it in the γ/S computation (§3.2 steps 2–3) and in `quantize`
  (s = e + σ ± w). That is one int8 load per element, shared per tile (≤ 4 KB of shared memory),
  coalesced along k̄.
- K4, K5 and the CRT do not change. Flags and the native path use the original values.
- The unweighted path stays byte-identical, through a uniform `kw == NULL` branch or a template
  flag, whichever measures better.
- Bar: no regression unweighted, and ≤ 2 % of the pack phases weighted.

**Ranges** [I; cert required: `cert_kw_shift_range`, `cert_pack_words` rerun on weighted
ε-maximizing rows]:
- With |w| ≤ 127: γ̃ ∈ [−1201, 1150], σ ∈ [−1164, 1278] and σ + τ ∈ [−2328, 2556], all inside
  int16.
- `round_scaled` takes an `int` exponent (src/kernels/numerics.cuh:209); its correctness over the
  wider range is a certificate item.
- The emitter precondition |a′| ≤ T is unchanged, since the norm budget now applies to ã. So
  W(N), kPackXBits and ε ≤ ⌊log₂ T⌋ are unchanged.

**Where w comes from.**
- Both operands in-line: `imd_call_t` gains `const int8_t* kweights` (it replaces `reserved1`;
  same size), in the Q9 order and converted to k̄ as the open item above decides.
- One packed operand: w comes from that pack's `kw` region, and the in-line partner applies −w.
  The pack is the single source of truth.
- **Both packed with weights → NOT_SUPPORTED** (Q22). A caller-chosen id cannot prove that two kw
  vectors are equal; equal ids over different kw would give a wrong C with SUCCESS.
- A weighted pack with an unweighted packed partner is also rejected: the partner never applied
  −w.

**Pack ABI.**
- New entry points `imd_pack_create_ex`, `imd_pack_layout_ex` and `imd_pack_ex` take
  `imd_pack_desc_t` v2 (`kweights` 0/1), under the zero-tail rule (§4.9). The old functions keep
  v1 meaning (Q23).
- The format word gains a weighted bit, so older libraries report FORMAT for weighted keys.
- There is a new reason `IMD_PACK_MISMATCH_KWEIGHTS`.
- The `kw` region holds k̂ bytes, 256-aligned, after res, in the pack's adopted K order.
- Unweighted packs keep their bytes, formats 1/2 and keys.

**Accuracy meaning.** No change to `imd_err_request_t`. The §6f statistics refer to the weighted
operands (L5b).
- When the stage-4 guard held for the data, the plain operands' **norm** statistics stay valid
  upper bounds (L5c), so NORMWISE keeps its meaning.
- **ℓ1 ratios do not carry over** (L5c): pass `l1_ratio` = 0 (ν = √k) or ratios measured on
  the weighted rows.
- `imd_err_request_kweighted` (stage 4, L5d) builds the rigorous request from plain statistics
  and the reported G.

**Spec.**
- OzakiScaling JointSpec with a constant weight vector: `CRTPrecondition`, `FinalRounding`
  against x = Σab, and `ErrorBound` in weighted norms. Negatives: `neg_kw_one_side` (weights on A
  only), `neg_kw_same_sign`, `neg_kw_range` (σ overflows at toy width).
- PackedOperand: `pos_kw_from_pack`, `neg_kw_partner_own` (the in-line partner uses its own w
  instead of the pack's), `neg_kw_both_packed` (the witness for NOT_SUPPORTED).
- Pipeline: kw is a read-only input of the scaling and pack kernels; `neg_kw_helper_race` (the
  helper's output is read on another stream without an event).
- Certificates `cert_kw_shift_range`, `cert_pack_words` (weighted), `cert_error_estimate`
  (weighted rows), and `cert_workspace`/`cert_pack_layout` for weighted packs.
- Oracle: `scale_vector` and `quantize` with a dyadic weight (no FP `ldexp`, which would
  overflow), `contract(..., kweights=)`, `pack(..., kweights=)`, and `lower_packed` with the
  weighted key.

**Tests.**
- Weighted contractions bit-exact against the oracle on every launch path.
- Weighted pack bytes, including the kw region.
- NOT_SUPPORTED cases and key statuses.
- The unweighted path bit-identical to v1.18 (hash).

**Perf.** Optional calibration key `pack.kw_elem` (absent = 0, so predictions are bit-identical).
No new phase.

### 4.8 Spec work by module

| artifact | stage | change |
|---|---|---|
| PackedOperand.tla | 0 | **none** (R1): ordering is already modelled, the stale handle is §6b.10 #3's documented limit, and the layout prefix is a params certificate. Optional, if Eric asks: ⟨version, key⟩ cells, a c3 after post with RAW post → c3, `neg_repack_stale` (§4.1) |
| DeviceReduce.tla (new) | 4, 5 | helper reductions (`neg_kw_no_zero`, `neg_kw_rmw`, `neg_kw_median_early`); statistics (`pos_stats_two_calls`, `neg_stats_flush_before_barrier`, `neg_stats_global_rmw`, `neg_stats_lib_zeroes`) |
| HostAPI.tla | 5 | `pos_stats_shared` |
| IndexMap.tla | 5 | stats argument checks (`neg_stats_alias`) |
| OzakiScaling.tla | 4, 5, 6 | `GuardSound`, toy-exhaustive (4); `CertSound` (+ `neg_cert_no_kdelta2`, `neg_cert_floor_kappa`) (5); weighted JointSpec (+ `neg_kw_one_side`, `neg_kw_same_sign`, `neg_kw_range`) (6) |
| PackedOperand.tla | 6 | `pos_kw_from_pack`, `neg_kw_partner_own`, `neg_kw_both_packed` |
| Pipeline.tla | 6 | kw read-only input, `neg_kw_helper_race`; the layout of the k̄-ordered weights (open item, §4.7) |
| OzakiCRT, CRTRealization, K4Align | — | no change: integer back end independent of how the scaling is chosen (OzakiCRT `CRTExact`) |
| PROOFS / Lean | 3, 4, 5 | L7 `kseg_compose` (3); L5 `kweight_cancel`, `kweight_guard` (4); L6 `apost_cert` (5) |
| params.py | 0, 1, 3, 4, 5, 6 | `cert_footprint_monotone` (0); `cert_rho_recipe` (1); `cert_kseg_composition` (3); `cert_kw_rule`, `cert_kw_fp64_exact`, `cert_kw_guard` (4); `cert_stats_constants`, `cert_stats_bin`, `cert_apost` (5); `cert_kw_shift_range` and reruns (6) |
| oracle.py | 2, 3, 5, 6 | `kweights` (2); `contract_segments` (3); `apost_bin`, `stats` (5); the weighted quantizer (6) |

Every TLC/Apalache run goes through `tools/memcap.sh` with bounded parallelism.

### 4.9 Versioning

- **Stages 0–3: no ABI change.** `IMD_HOST_API_VERSION` stays 13. The F1–F3 guarantees are
  documented "from API 13 on"; they describe the current formulas and change no meaning. They
  cover only the totals and the write extent (§4.1). The offset and device-byte facts I1–I2 are
  certificates, not ABI, so a later format may change them.
- **Stages 4, 5 and 6** each take the next host API number in the order they ship (14 for the
  first). No number is reserved in advance, so the drafts' competing "API 14" claims are void.
- **New meanings only through new function names or validated words.** New functions
  (`imd_stats_cond`, `imd_err_request_kweighted`, `imd_pack_*_ex`), or validated flag and
  reserved words that older libraries reject when nonzero (`imd_call_t` reserved fields). Never
  through an appended tail field that an older library ignores silently. `imd_err_request_t`
  does not change in any stage. Its `reserved` word stays free (R3).
- **Zero-tail rule for every new `struct_size`-versioned input** (`imd_call_t`,
  `imd_kweights_params_t`, `imd_pack_desc_t` v2; R-api should-fix 4). The library reads
  `struct_size` bytes. Every byte beyond the size it knows must be 0, else INVALID_VALUE. So an
  older library rejects a newer caller's appended field instead of ignoring it. That closes the
  bug class of accuracy.cpp:99 and pack.cpp:133 for every new struct, not only through reserved
  flag words. Existing structs keep their current rule.
- `struct_size` stays append-only. The calibration format stays 2, with optional keys only.
  `imd_lowering_t` does not change.
- **Lemma numbers.** PROOFS gets L5 (K-weights), L6 (a-posteriori certificate) and L7
  (segments). Rules recorded but not adopted keep names local to these notes: LA1–LA6
  (Appendix B) and the slice lemmas of §3.3.

### 4.10 Risks (overall)

- **C1's value depends on the final precision** the consumer needs: 1.4–1.5× when ending at
  N = 14, 1.1× when ending at N ≈ 9.
- **It also depends on the shape.** Large calls (≥ 1024³, DF) gain 1.4–1.5× from a schedule
  ending at N = 14. The Jacobi view gains about 1.27× and the o10/v96 PPL view about 1.33×.
  Many-small-plans workloads (256³–512³) gain only about 10–25 % from a lower N (§2.3–§2.4)
  [M, review].
- **It rests on unvalidated N ≤ 7 predictions** (M-c) and on unmeasured iteration counts under
  ramps, DIIS included (M-d1).
- **The C2 evidence is synthetic and small,** one seed, and without the guard. Random-structure
  gains already vanished at m = n = 512; the structured gains must be confirmed on real operands
  (stage 2) before stages 4 and 6.
- **C2's speedups are upper bounds.** The helper and the pre-scaling recur for operands rebuilt
  every iteration, and they cut 1.55× to about 1.2–1.44× at 4096³ unless the pre-scaling is fused
  (§2.6).
- **Weights can go stale,** and caller-side pre-scaling may need copies (stage 4).
- **The re-pack and pre-scaling consistency rules are caller obligations** that the library
  cannot check (stage 0, stage 4).
- **The statistics overhead** in K5 is unmeasured (stage 5).

---

## 5. Design questions and their resolutions

Both reviewers answered every question.
- **R-num**: numerics, proofs and spec.
- **R-api**: API, pipeline, performance and integration.

Each entry gives the options, the resolution, who decided it, and the rationale. "Both" means
the two reviewers chose the same option; amendments from either are adopted unless the entry
says otherwise. Where they disagreed, the entry says so and decides on the merits. §5.2 records
the review decisions that fall outside the questions.

### 5.1 The questions

**Q1. What is the C1(a) deliverable?**
(a) Plan per N + pack per N, with F1–F3, the re-pack protocol and `PrecisionPlan`;
(b) an N-range plan now (§3.2);
(c) both.
**Resolved: (a), both.** F1–F3 are reduced to the totals and the write extent (R-api must-fix
3; §4.1). The range plan stays parked behind the §3.2 trigger.
*Rationale.* A range call is launch-identical to the single-N′ plan, so it gains nothing per
call: contract.cu already selects by p.N at every call (`launch_k5_for` :662, `pack_words`
:974/1222, `gsize` :948/1200, the group loop :1437). F1–F3 hold by construction of
`workspace_layout` and `pack_layout`. The range plan costs M-size work: new entry points, a new
lowering version, and four TLA+ modules (Pipeline alone has 103 models). It would save only
unmeasured host bytes and plan creation, which is 11–12 µs for GENERATE (NOTES-M4:213), once per
level.

**Q2. When are buffers sized?**
(a) Once at N_max (guaranteed by C-ws/C-pack);
(b) per phase, which lets smaller early phases use coarser caller tiles;
(c) document both.
**Resolved: (c), both.** `PrecisionPlan.footprints` reports the N_max sizing as the default
(graph-capture friendly) and the per-level values as the option.
*Rationale.* C-ws follows from ws(N, g) ≤ ws(N_max, g) ≤ ws(N_max, g_max), valid because
g ≤ N ≤ N_max (`options_ok`), and the contract checks accept larger buffers
(immerdink.h:112-113). RA/RB scale with g and W with N (workspace.cpp:49-56), and the pack's
res is affine in N (pack.cpp:44-47). Per-phase sizing matters under memory pressure (ehrenfest's
17.7 GB DF packs), and it lets low-N phases use coarser caller tiles.

**Q3. How is the stale-handle hazard of re-packing in place guarded?**
(a) Documentation, a `neg_repack_stale` witness and the Python `PackedBuffer`;
(b) a device-side digest header in packed buffers, which reverses §6b.10 #12, changes the byte
contract, and still cannot protect graph replays;
(c) documentation only.
**Resolved: (a) without a stage-0 TLA+ witness**: documentation plus `PackedBuffer`; (b) is
rejected. **The reviewers disagreed on the witness.** R-num kept it, rebuilt on ⟨version, N⟩
cells. R-api dropped it (its must-fix 1). **Decided for R-api on the merits** (R1, §5.2). The
witness would only restate the caller rule, because the library never compares bytes with
handles. Ordering is already modelled. R-num's cell model and R-api's c3 op are recorded together
as the optional model of §4.1.
*Rationale.* The hazard already exists for any re-pack with another key; §6b.10 #3 says the
library cannot catch "a wrong buffer of the right kind". A header would reverse §6b.10 #12 and
break §6b.4's rule that the bytes are a pure function of (key, values), and it still could not
protect graph replays. Checking bytes against the handle needs a sync, which kernel scope
forbids. `PackedBuffer` is cheap and opt-in, and raw buffers behave as today.

**Q4. How much does `PrecisionPlan` do?**
(a) A thin target → N per phase helper, with the forcing rule as example code;
(b) only document the §6f.3 pattern;
(c) a library forcing rule or default schedule.
**Resolved: (a), both, with R-api's amendments** (§4.2):
- its own module `precision.py`;
- it takes a `Planner`;
- one rank over the levels ≥ the floor, with the accuracy set;
- `reuse` for packed roles;
- a monotone floor with `reset()`;
- no `noise_floor`;
- `configs()` up front.

R-num's amendments are adopted as well: p = P/C for TYPICAL; the monotone floor is the previous
phase's final N, after the share-group raise.
*Rationale.* It only composes `min_moduli` and rank, which already exist, and it keeps policy
with the caller (ehrenfest D8: precision is user-directed). planner.py's contract is "nothing
computed in Python" (DESIGN.md:4204). A single rank call over the levels matches §6f.3's
guidance, and it holds even where time is not linear in N.

**Q5. What accuracy basis does `PrecisionPlan` use by default?**
(a) BOUND;
(b) TYPICAL;
(c) BOUND for the final phase and TYPICAL for intermediate phases, as passed by the caller.
**Resolved: (a) as the default, both; (c) documented** as an explicit per-phase opt-in, with
fail_prob = P/C over the phase's calls (§4.2 step 6). M-d measures both.
*Rationale.* BOUND is the library's rigorous contract (§6f.2), and TYPICAL is opt-in by design.
TYPICAL saves about ⅔ of a modulus, but it fails on rows of many equal entries (ROADMAP B2), and
its p is per call, so a schedule must divide P across the run. In intermediate phases a miss only
costs iterations, which only the caller can judge.

**Q6. Slice mode and ROADMAP's "TLA+ for slices"?**
(a) Park with triggers (§3.3) and record the numerics ideas;
(b) specification only now (slice lemmas, OzakiSlices.tla, PackedOperand ordered compare),
without kernels;
(c) build it.
**Resolved: (a), both.** "TLA+ for slices" is deferred to the §3.3 trigger. ROADMAP C1's
decision column becomes: C1 = (a) + host-side (b); (c) parked. **Reported to Eric for
confirmation (Status); he may override it, but it does not block stages 0–3.** (R-num: "does
not block C1". R-api: "for confirmation". The two are compatible.)
*Rationale.* No consumer needs one operand at two precisions at once (§1.3), and that is the only
case a prefix format pays for. Slices win only on benign data at about 1e−8 or looser, the cheap
part of any schedule. They are 1.25–1.75× slower at FP64 level (S = 7), even before the
optimistic matching of Appendix A items 19–20. A slice format also breaks the exact-equality
pack keys (pack.cpp:103), the boolean rounding word (pack.cpp:80) and kNMin = 2. These pitfalls
are recorded, so a revival stays cheap.

**Q7. Which C2 scheme?**
(a) K-equilibration (staged, gated) plus the K-segmentation recipe, with accurate mode
re-rejected and the envelope rule recorded;
(b) the envelope accurate mode (`IMD_EMULATE_ACCURATE`);
(c) [SC25] accurate mode;
(d) drop C2.
**Resolved: (a), both.** ROADMAP C2's row is retitled from accurate-mode scaling to
K-equilibration, with accurate mode re-rejected. **Reported to Eric** with Q6 (Status).
*Rationale.* Both use the same headroom, but K-weights get about twice the moduli per bit. In the
§3.8 bound, K-weights scale both error terms with ‖ã‖‖b̃‖ (about G/3.9 moduli), while σ/τ
shifts cut each term only by the partner's increment (about G/7.8). These are estimates from the
bound, confirmed by the data (§2.8): about 6 against about 3 moduli on decay_opp. K-equilibration
keeps D1 packing, the single final rounding and exact integer rules. Every accurate variant adds
an operand read, loses KF/KL group-1 fusion and costs 0.6–6.7 moduli of time (§2.5). [SC25] in
its 6-bit form gains ≤ 0.34 moduli. [KT26] independently finds that a scale-invariant CS fast
mode matches accurate mode (2606.29129 lines 709-718, 857-872, checked by R-num).

**Q8. What is the first form of K-equilibration?**
(a) Caller-side pre-scaling plus the device helper (stage 4), library-internal later (stage 6);
(b) library-internal first;
(c) an oracle/Python helper on the CPU only.
**Resolved: (a), both,** with both stages only after the stage-2 gate. Until the gate passes,
the rule exists only in `oracle.kweights` (added in stage 2, R2) and the CPU study. The FP64-exact
clamp is mandatory (R-num must-fix 1; R4).
*Rationale.* Pre-scaling leaves the library's numerics and pack format untouched: C is bit-exact
against `oracle.contract(ã, b̃)`, and the native path is unchanged because the real products are
identical. The helper is kernels only and capture-safe, with exact sizing from
libimmerdink_host. Library-internal weights touch every scaling kernel, the pack ABI and three
TLA+ modules, so they wait for measured demand.

**Q9. In what order is the weight vector indexed?**
(a) Column-major over A's K labels in A's descriptor mode order (first label fastest);
(b) the plan's canonical k̄, which needs a host query exposing the K order and ties the weights
to orders that depend on partner and C strides (§5 step 8);
(c) (a) plus an optional `imd_kweights_apply` that writes scaled copies.
**Resolved: (a), amended; no `imd_kweights_apply`** (§4.5 "Index order"): over the K labels
**both** operands carry, in A's descriptor mode order, first label fastest, with a repeated
(diagonal) label counted once at its first occurrence. Extent-1 labels change nothing. A K label
carried by only one operand is not indexed, so it gets w ≡ 0.
**The reviewers differed on one-sided labels.** R-api indexed "A's distinct K labels with
extent > 1", which includes A-only K labels. R-num excluded labels the partner does not carry.
**Decided for R-num on the merits:** a weight along a label B does not carry would need B scaled
by 2^{−w} along that label. That is impossible without materializing B, because B's entry is
shared across the label (stride 0, DESIGN §5 step 6).
*Rationale.* The caller applies w to its own tensors in stage 4, so the caller's order is the
usable one. The canonical k̄ depends on the partner and on C's strides, so (b) would expose
unstable orders. Stage 6's conversion to k̄ needs its own layout design first (§4.7, open item).

**Q10. Never-worse guard?**
(a) The exact σ-based guard of L5c (margin 3) reported by the helper, with the caller accepting
or dropping w;
(b) no guard, strict opt-in;
(c) floating-point norm ratios.
**Resolved: (a), both,** with three amendments:
- the helper also reports G, and `imd_err_request_kweighted` turns it into a rigorous request
  (R-num should-fix 1; L5d, R5);
- the ℓ1-ratio caveat applies (R-api must-fix 2);
- the toy-exhaustive `GuardSound` check is added (R-num should-fix 4).

The guard is "never worse in the §3.8 bound", not in the realized error, and M-a measures it.
*Rationale.* It is an integer rule on existing §3.2 machinery. R-num re-derived the proof: the
ratio is < 4s²·2^{−3} < 1 for k ≤ 2^40, while margin 2 would give only < s². Its conclusion is
N-free, so one guard serves every phase, and it keeps the plain-norm §6f bounds valid at
ν = √k. Floating-point norm ratios (c) would be neither exact nor reproducible, the only
non-exact rule in the design. (b) is ruled out by §2.6's size check: worst outputs lose
3.7–13.6 bits at m = n = 512.

**Q11. Which weight rule?**
(a) Deadband-median balanced, with the deadband a parameter re-tuned in stage 2 at
m = n ≥ 128;
(b) plain balanced;
(c) one-sided, from the reused operand only.
**Resolved: (a), both.** The deadband is re-tuned in M-a at m = n ≥ 128 with at least 5 seeds.
The guard is the actual barrier against regressions.
*Rationale.* The plain rule lost up to 1 modulus on φ = 0.5 and decay_same at small sizes, and
the deadband did not prevent regressions at m = n = 512, so it is a tuning knob, not a guarantee.
Centring on the median is harmless (σ is scale-equivariant) and keeps |w| small inside the
clamp. The one-sided rule (c) cannot balance both operands' norms, which is where §2.8's 2×
comes from.

**Q12. Are weights shared across batches λ?**
(a) Shared;
(b) per λ (b·k bytes).
**Resolved: (a), both,** unless M-a finds batch-dependent profiles.
*Rationale.* One k_s-vector keeps caller-side scaling and a stage-6 kw region simple. Per-λ
weights cost b·k bytes and a batch index in every scaling and pack kernel, with no evidence of
benefit. The guard already ranges over every (λ, row). The magnitude structure of the motivating
cases (orbital energies, DF factors) comes from the contracted labels, not the batch.

**Q13. What gates stages 4 and 6?**
(a) ≥ 2 moduli saved at equal target on calls of the 2048³ class or larger;
(b) a predicted net gain ≥ 5 % TFLOPS-eq at equal accuracy for a recurring contraction a
consumer confirms is realistic;
(c) a needed target becomes reachable at N ≤ 20.
**Resolved: (b) or (c), both; (a) only as a proxy.** Conditions from both reviewers:
- it is measured on real or consumer-confirmed operands at the 2048³ class or larger (Q14);
- the cost counts every term: the helper (amortized over the profile's lifetime, or per call if
  it is re-run), the unfused pre-scaling of operands that change per iteration (16 B per element
  per call), and the memory of scaled copies (§2.6, M-e).

*Rationale.* Moduli saved is not time saved: the fixed part of a call caps the gain at low N
(§2.3), and an unfused pre-scaling pass alone takes 1.55× down to about 1.35–1.44× at 4096³.
Random-structure gains vanished at m = n = 512, so real data must decide. Option (c) has a value
that no time ratio captures.

**Q14. Where does real-operand evidence come from?**
(a) The synthetic integral × amplitude family now, plus a one-off ehrenfest FP64 dump of a few
DF-CCSD operand pairs;
(b) synthetic only.
**Resolved: (a), both.** The dump needs Eric's go-ahead. It runs CPU-only, single-threaded,
under memcap, outside ehrenfest's test windows (busy check, silent mode). Raw dumps and records
stay in immerdink-eval `results/c1/` or scratch, never in the product repo.
*Rationale.* The whole C2 value case rests on the claim that real anti-correlated profiles are
separable along K. Only real operands can confirm it, and the dump is cheap and read-only.

**Q15. How is K-segmentation exposed?**
(a) A documented recipe plus L7, with no API;
(b) a SEGMENTS flag (a new function or a validated word, not an appended field) plus a field, if
stage 2 shows value beyond K-equilibration;
(c) a library segment loop or an exact-combine K5.
**Resolved: (a), both.**
*Rationale.* It already works as S calls of one plan with β = 1 and moved base pointers, packs
still work, and L7 bounds it rigorously (comparing against ν = √k only). It costs +10–93 % per
call and the single final rounding, and K-equilibration dominates it on separable data. An API is
justified only if M-a shows value beyond K-equilibration.

**Q16. When do K5 statistics get built?**
(a) Gated on a consumer with elementwise targets, with the stage-1 recipe first;
(b) now, with stage 4.
**Resolved: (a), both.** **The reviewers differed on the recipe's cost.** R-api took "one pass
over C, about 3 % of a call". R-num pointed out that the recipe also needs the caller's row and
column norms. **Decided for R-num on the merits:** n_ij ≥ ‖a_i‖‖b_j‖ is an input of the recipe.
So its cost is one pass over C plus a pass over A and B, unless the caller already has the norms:
about 9 % at 4096³ when everything is recomputed [E] (§4.2).
*Rationale.* The recipe gives a rigorous ρ now with no ABI or K5 change. K5 statistics add an
entry point, a device struct, a hook in the hottest per-output code and an unmeasured cost bar.
Their real advantage is that they need no norms at all (L6: ‖a·2^σ‖ ≤ d), and that justifies an
ABI change only once a consumer needs elementwise targets.

**Q17. How are the statistics binned?**
(a) ρ̂ bins plus separate uncertified and zero-z counters;
(b) certified-bits bins (bitlen|z| − bitlen B̄, tied to the call's N);
(c) both.
**Resolved: (a), both,** plus an `out_of_range` counter for |y| < 2^−1022 or D = ±Inf
(R-num must-fix 2), recorded k (`k_max`, `k_min_c`; R-num should-fix 5), and `other_calls`.
Both reviewers' rewording is adopted: each bin entry upper-bounds ρ_ij for any N, and its
tightness depends on N. The bins are not "independent of N".
*Rationale.* One histogram serves both cond-from-stats and, given (N, rounding, k) validated
against the buffer, the per-call certificate u + (1+u)·B̄·2^r/X for in-range outputs.
Certified-bits bins would tie the buffer to one N and could not be accumulated across phases.

**Q18. Where does the ABSOLUTE norm statistic come from?**
(a) Drop it from v1; use the caller's norms;
(b) min σ over nonzero rows from the scaling kernels, which covers in-line operands only, since
packed operands were scaled at pack time;
(c) include zero rows, which is sound, but a single zero row can inflate the maximum by up to
about T.
**Resolved: (a), both.**
*Rationale.* `np_key` is unsound: a nonzero row whose z is 0 against every column at this N
drops out of the max. Including zero rows is sound but useless, since σ := 0 inflates the bound
by up to about T. (b) misses packed operands. The caller's norms cost one pass and are exact.

**Q19. Who zeroes the statistics buffer?**
(a) The caller;
(b) immerdink, per call.
**Resolved: (a), both,** e.g. with `cudaMemsetAsync` (capture-safe). Re-zeroing before each
graph replay that should start afresh is documented.
*Rationale.* Zeroing in the library on every call would race with concurrent calls that share a
buffer (`neg_stats_lib_zeroes`), and it would prevent deliberate accumulation. Caller zeroing
matches every other caller-owned output under kernel scope. A zeroed buffer is the identity of
every field's operation (§4.6).

**Q20. How are statistics switched in K5?**
(a) A uniform runtime branch on `stats != NULL` first;
(b) a K5 template variant (19 N × 4 variants × 2).
**Resolved: (a), both;** (b) only if the stage-5 bar fails. The bar includes K5's register count
and occupancy with `stats` NULL, in every variant and at every N (R-api).
*Rationale.* Template variants double K5's 19 × 4 instantiations, with their compile-time and
library-size cost (cf. the +10–13 % measured for v1.14's W templates). A branch that is not taken
costs no time, but it can still raise register allocation, and K5 runs at 82–96 % of its bound
(ROADMAP A4). So the bar measures registers and occupancy explicitly.

**Q21. Do zero-z outputs count as misses in cond-from-stats?**
(a) Yes, by default, with an exemption flag;
(b) always exempt;
(c) always counted.
**Resolved: (a), both,** now as `IMD_STATS_ZERO_EXEMPT`, a flag of `imd_stats_cond` (R3) rather
than of the request.
*Rationale.* z = 0 shows only |x| ≤ B̄·2^{−σ−τ}, not x = 0, so exempting by default would be
unsound for a rigorous path. Callers that know x is structurally zero can opt in.

**Q22. Which weighted-pack combinations are allowed in stage 6?**
(a) At most one weighted packed operand; both-packed weighted → NOT_SUPPORTED;
(b) a caller-chosen `kweights_id` key word, where a caller error gives a wrong C with SUCCESS;
(c) kw passed per call for packs too, with no stored region.
**Resolved: (a), both.** A weighted pack with an unweighted packed partner is also rejected.
*Rationale.* The pack's kw region is the single source of truth, applied as −w by the in-line
partner, which is exactly checkable. With a caller-chosen id (b), a caller error becomes a wrong
C reported as SUCCESS, which the library cannot detect without a device-side compare and a sync.
(c) brings back a per-call consistency obligation the library cannot check.

**Q23. What is the weighted-pack ABI?**
(a) `_ex` entry points plus a weighted format bit plus a KWEIGHTS reason;
(b) a v2 descriptor tail field, which older libraries silently ignore because `pack_lower` checks
only a minimum `struct_size` (pack.cpp:133);
(c) a flag bit in the validated `order` word.
**Resolved: (a), both,** with the zero-tail rule for `imd_pack_desc_t` v2 (§4.9).
*Rationale.* R-num verified that `pack_lower` checks only `struct_size ≥ sizeof(imd_pack_desc_t)`
(pack.cpp:133), so an API-13 library would silently ignore a tail field. A new format bit makes
older libraries report FORMAT, a clean rejection. Overloading the `order` word (c) mixes
meanings. Under §6a a change of meaning gets a new function name.

**Q24. How is the performance model covered at small N?**
(a) Hold-out first (M-c), recalibrating only if the §6c.5 targets are missed;
(b) add N = 4–7 cases to the calibration now.
**Resolved: (a), both.** M-c (about 20 calls on shapes outside the calibration set, N = 4–7) is
a prerequisite for publishing any schedule gain that uses N ≤ 7 (§2.4).
*Rationale.* The hold-out covers only N = 8, 10, 14 and 16, while the schedules lean on N = 6
predictions, and [KT26] reports INT8 throughput at small N that depends on the data. Validate
before refitting: adding calibration cases first would spend GPU time, and possibly add model
features, without knowing whether they are needed.

**Q25. What about DIIS at precision switches?**
(a) Measure in M-d (with and without a history reset) and document the result for callers;
(b) leave it to the caller.
**Resolved: (a), both.** The convergence part runs on the CPU with the bit-exact emulator
(M-d1, R-api must-fix 4). The GPU is used only for timing on realistic shapes (M-d2).
*Rationale.* ACCURACY §3 shows only that a contracting Jacobi iteration does not accumulate
emulation error. A DIIS history that mixes vectors computed at two precisions is untested, and
it decides whether ramps keep the iteration count, which is M-d's success criterion. Iteration
counts do not depend on GPU timing.

### 5.2 Review decisions outside the questions

- **R1. Stage-0 TLA+.** **The reviewers disagreed.** R-num (should-fix 3) wanted the model
  re-scoped: cells ⟨version, N⟩, contracts bound to their plan's N, `neg_repack_stale`
  violating "every contract reads cells whose last writer's N equals its plan's N", and
  `LayoutPrefix` moved to params. R-api (must-fix 1) wanted it dropped.
  - **Decided: drop it** (§4.1 "Spec"), justified as §6f.5 does.
  - *Merits.* As drafted, the model cannot be expressed: in StreamSpec, c2 can never observe a
    re-pack. Re-scoped, it would check two things. The ordering is already modelled
    (`neg_stream_no_war`, `neg_stream_no_raw`). The N binding is the caller rule the library
    cannot check (§6b.10 #3, #12), so its witness would be tautological.
  - Both agreed that `LayoutPrefix` belongs in `cert_footprint_monotone` (I1).
  - The combined model (R-num's ⟨version, key⟩ cells plus R-api's c3 with RAW post → c3) is
    recorded as optional, should Eric want one.
- **R2. What stage 3 front-loads.** **The reviewers disagreed.** R-num wanted L5 and L7 proved
  in stage 3, with L6's hand proof and `cert_apost` there too and only L6's Lean deferred. R-api
  wanted stage 3 to keep only L7, `oracle.contract_segments`, the K-seg recipe and the §7
  rewrite, with L5 and its certificates in stage 4 and L6 and its certificates in stage 5.
  - **Decided for R-api, amended:** `oracle.kweights` moves forward into stage 2. M-a needs one
    exact reference for the rule, and both reviewers' Q8 answers place the rule in the oracle
    before the gate.
  - *Merits.* Nothing in stage 3 relies on L5 or L6. guard = 1 becomes a claim the library
    reports only in stage 4. L6 is used only by stage 5. Both are gated and may never ship, so
    proving them first spends effort ahead of value. The hand proofs, re-derived by R-num, and
    R-num's design-phase exact checks (L5c and L6, §4.4 Evidence) are recorded now. Nothing is
    lost by the deferral.
- **R3. Stage-5 accuracy API.** R-api (should-fix 3) replaced the request changes with the pure
  host function `imd_stats_cond`. R-num did not address the API shape, but its validations (k
  recorded and checked; out-of-range outputs) are carried into `imd_stats_t` and
  `imd_err_posteriori`. **Adopted.**
  - The rigor is the same, and `imd_err_request_t` stays v8.
  - Accuracy stays hashable by value, so no stats snapshot is needed for `Memo`.
  - The 156-bin scan leaves the hot path.
  - `IMD_ERR_UNCERTIFIED` and `IMD_ERR_COND_FROM_STATS` are dropped.
- **R4. FP64-exact clamp.** R-num (must-fix 1) offered two choices: an unconditional clamp, or
  INVALID_VALUE for `IMD_KW_GUARD` without `IMD_KW_FP64_EXACT`. **Decided: unconditional,** and
  the flag is removed.
  - Stage 4's only use is caller-side pre-scaling, and that needs exact scaled entries.
  - A flag would only add a way to get wrong results silently.
  - Stage 6's in-kernel weights use the dyadic quantizer instead and need no FP64 clamp.
- **R5. Guard → planner.** R-num (should-fix 1) asked for the transform to be documented,
  reported as G and certified. R-api (must-fix 2) asked for the ℓ1 caveat. **Adopted as one pure
  host function, `imd_err_request_kweighted`** (L5d, §4.5). It applies G and resets the ℓ1
  ratios, so the caveat cannot be forgotten. This is in the minimal-API style R-api prefers
  (cf. R3).
- **R6. Published footprint guarantees** (R-api must-fix 3). **Adopted:** only F1, F2 and F3
  (totals and write extent) are ABI. The offset and device-byte facts are certificates (I1, I2),
  because §4.7's kw region and any slice format or range plan could break them, and an ABI
  promise cannot be withdrawn.
- **R7. Per-call certificate in the normal range** (R-num must-fix 2). **Adopted:** the
  `out_of_range` counter, never certified (§4.6, L6b). cond-from-stats is unaffected, because
  ρ ≤ X/Y holds on the exact integers.
- **R8. M-d split** (R-api must-fix 4). **Adopted:** convergence on the CPU (M-d1), time on
  realistic shapes judged against each system's own prediction band (M-d2), and each system's
  predicted ratio stated next to the bar.
- **R9. Zero-tail rule** (R-api should-fix 4). **Adopted** for every new `struct_size`-versioned
  input (§4.9).

---

## Appendix A. Claims corrected in review (not carried into this design)

P1–P4 are the four drafts (§Status).

1. **P1, gain attribution.** P1 credited N-range plans with the 1.41–1.69× per-call gain. That
   gain comes from running at a lower N, which per-N plans already give; a range call is
   launch-identical by design (§3, A2).
2. **P1, schedule comparison.** P1's "1.55–1.70× over a fixed N = 14" schedule
   (6×N6 + 5×N8 + 5×N10) ends at N = 10's accuracy. The fair comparison is 1.23× against a fixed
   N = 10. Schedules that end at N = 14 give 1.37–1.51× (§2.4).
3. **P1, `np_key`.** A max over outputs with z ≠ 0 is unsound as a `norm_product` for
   ABSOLUTE: a nonzero row orthogonal to every column at this N is left out. Dropped (§4.6, Q18).
4. **P1, ρ̂ slack.** "ρ̂ overestimates ρ by at most 4s² (about 2 bits)" is false near the
   threshold. The ratio is < 4s²(|z| + B̄)/(|z| − B̄) before binning and ×2 after, unbounded as
   |z| → B̄ (L6).
5. **P1, weight relation.** "w_h² ∝ β_h/α_h" should read 4^{w_h} ∝ β_h/α_h, because w is an
   exponent.
6. **P1, K-eq gains.** The gains on outliers and φ data, and "the deadband removes every
   regression", were artifacts of 24×24 and 16×16 sizes. At m = n = 512 the worst outputs lose
   3.7–13.6 bits (§2.6).
7. **P1, stale weights.** The stale-weight loss on φ = 4 is 1–2 moduli (2 at three of the four
   targets), not "about 1".
8. **P1, K5 share.** "K5 is 8–9 % of a large call" holds only for some shapes: 5.1 % (8192³),
   8.6 % (4096³), 15.6 % (DF) [M]. It was measured at 16.8 % on sq4096, N = 14 (NOTES-D1:106).
9. **P1, both-packed weights.** Two weighted packs matched by a caller-chosen `kweights_id` turn a
   caller error into a wrong C with SUCCESS, not merely an accuracy loss. Hence Q22.
10. **P1 and P4, the case against accurate mode.** "Accurate mode gains ≤ 1 modulus" holds only for
    the 6-bit SC25 form. Exact, envelope and blocked partner bounds realize about 3 moduli on
    decay_opp. The valid reason for rejection is K-weights' 2× conversion (§2.8).
11. **P4, Reader 1's headroom figure.** P4 explained Reader 1's 14.7–19.6 bits as "ideal
    headroom". That is unsupported: the ideal is 24.6 bits, and the exact-bound rule realizes
    about 24.
12. **P4, K-segmentation bound.** "Never worse in the bound" holds for NORMWISE and ABSOLUTE only.
    ELEMENTWISE is worse at high N and gives up the single rounding (L7).
13. **P4, K-segmentation cost.** The cost left out the S-fold K4 W-plane writes and the β = 1 read
    of C; §2.7 has the corrected figures.
14. **P4 and P3, appended request fields.** Fields appended to `imd_err_request_t` (P4's
    `segments`, P3's `gain_a`/`gain_b`) are silently ignored by API-13 libraries, because
    `err_request_valid` checks only the minimum size (accuracy.cpp:99). A new meaning needs a new
    function (as stage 4's `imd_err_request_kweighted` and stage 5's `imd_stats_cond` do) or the
    validated `reserved` word (accuracy.cpp:100), which stays free (§4.9).
15. **P3, gains in fast mode.** `gain_a`/`gain_b` would also tighten fast-mode estimates:
    `imd_err_estimate` has no mode argument (accuracy.cpp:178), and `imd_err_min_moduli` accepts
    only EMULATE and DIRECT (accuracy.cpp:216).
16. **P3, headroom packs.** With e_j < 0 and "no clamp, no fast fallback", headroom packs are
    unsound under RN: |RN(2^e·y)| ≤ 2^e·⌈y⌉ needs e ≥ 0. Counterexample: y = 3, e = −2 gives
    RN(0.75) = 1 > 0.75, i.e. up to 2× per entry, so Σ|a′||b′| can exceed H. The safe bound is
    2^{e+1}·⌈y⌉ (use e − 1), or TRUNC on that side.
17. **P3, L9's proof.** The chain 2^d·c_h < 2^d(2^{uF+1} + 1) ≤ 2^77 + 2^76 fails for uF < 0
    (TRUNC at small N and large k: uF down to −14, d up to 90, c_h = 1). The cap |a′| < 2^78 still
    holds, via |a′| ≤ ⌈|a|·2^σ⌉ ≤ 2^{uF+1+d} ≤ 2^77.
18. **P3, region count.** "`IMD_REGION_COUNT` keeps its value while `imd_workspace_layout` returns
    11 regions" contradicts the documented output (include/immerdink/immerdink_host.h:149-151)
    and breaks tests/test_host_c.c:59.
19. **P2, benign matching.** "Benign data: N ≈ 2S + 1" includes φ = 0.5, whose own data give
    N ≈ 2S (S = 2 / 3 / 4 ≈ N = 4 / 6 / 8). The worst case over φ data is 2S − 2 (orchestrator's
    emulator), and S = 3's advantage disappears on φ = 0.5.
20. **P2, matched N.** "Matched N" is the smallest integer N at least as accurate: uniform S = 3
    (2.1e−8) was matched to N = 7 (8.2e−9, 2.6× more accurate), while N = 6 gives 1.3e−7. So the
    time ratios are optimistic.
21. **P2, quantizer bias.** The prefix-canonical quantizer's +1/510-unit bias is coherent on data
    with nonzero mean and grows linearly in k. The long-k extrapolation (2S + 2 at 65536, about
    +3 at 7e5) does not hold for same-sign data, and the zero-mean E_typ needs a bias term.
22. **P2, ABI.**
    - `pack_key` encodes rounding as a boolean (pack.cpp:80), so RB = 2 would key as RN.
    - `key_mismatch` is exact equality (pack.cpp:103-104), so "plan S ≤ pack S_max" changes the D1
      key contract.
    - S = 1 conflicts with kNMin = 2 in every validator.
23. **P2, operating point.** ehrenfest's ≈ 5.7e−10 operating point corresponds to N = 8 on P2's
    own uniform k = 4096 table (5.7e−10 at N = 8), and to N 8–9 on the ACCURACY maxima, not to 9.
24. **P2, evaluation-workspace comparison.** The comparison P2 cited as agreement never ran
    immerdink below N = 8, while N = 6 already reaches 1e−6 (2.7e−7 max normwise). It says
    nothing about slices against small-N Ozaki-II.
25. **All drafts, numbering.** "L5" was claimed by P1, P2 and P3, "L3d" by P3 and P4, and "host
    API 14" / "lowering v7" by P1, P2 and P3. Resolved in §4.9.

**Second review round (R-num, R-api) — claims of the previous version of these notes:**

26. **σ step.** "σ(N+1) − σ(N) ∈ {3, 4}" is false near RN acceptance: it is 5 at N = 2 → 3 with
    κ = 179 (§1.2 item 2). A6 is unaffected.
27. **Optional FP64 clamp.** With `IMD_KW_FP64_EXACT` optional, L5a, caller exactness and
    "flagged outputs unchanged" fail silently, and guard = 1 can be reported for an ã the caller
    cannot build. The clamp is now unconditional (R4).
28. **Certificate outside the normal range.** `imd_err_posteriori` would have claimed a relative
    bound for outputs that underflow gradually or overflow. Now they are counted out of range
    and never certified (R7).
29. **"Accuracy API: no change; keep existing statistics."** This was unsound for ℓ1 ratios,
    since weighting flattens rows. Pass ν = √k or weighted-row ratios, or use
    `imd_err_request_kweighted` (L5c, R5).
30. **"Exact never-worse guard."** The guard is never worse only in the §3.8 bound, and it had
    not been run in any study (§2.1, §2.6).
31. **Stage-0 TLA+ witness.** It was inexpressible in StreamSpec as drafted (c2 cannot observe a
    re-pack), and it mixed CompatSpec invariants into StreamSpec. Dropped (R1).
32. **F2/F3 as ABI.** N-independent offsets and plan device bytes are of no use to callers and
    would constrain later formats. They are now certificates (R6).
33. **M-d's success bar.** A time ratio of 0.8 on the 1600×100×1600 Jacobi view (predicted 0.79)
    cannot discriminate. Now: convergence on the CPU, time per system against its own band (R8).
34. **Recipe cost "one pass over C, about 3 %".** This omitted the caller's norms, a pass over A
    and B (Q16).
35. **"ρ bins are a property of the data, independent of N."** Each bin bounds ρ_ij for any N,
    but its tightness depends on N (Q17).
36. **"Plan creation 7–8 µs".** No source. The measured figures are 0.011–0.012 ms GENERATE and
    0.025–0.123 ms RESIDENT, and RESIDENT creation synchronizes (§3.1 A1, §4.1).
37. **`rank([N_c])` in `phase()`.** It contradicted §3.1 B1 and §6f.3's `rank([n, n+1])`. Now a
    single rank over the levels ≥ the floor (§4.2).
38. **"E_s ≤ E(k)" in L7.** This holds only against the ν = √k bound, not against a caller's
    whole-row ν (§4.4).
39. **§2.6 speedups as plain figures.** They are upper bounds that omit the helper and the
    pre-scaling (§2.6).
40. **§2.3 rows without a spec.** They are about 3 % off the `ik,jk->ij` form. The spec, strides
    and flags are now stated (§2.3).
41. **L6 looseness** omitted the factor ⌈d²⌉/d² (§4.4). **§2.8** used d for both the σ/τ
    increment and T − κ/2.

## Appendix B. The partner-envelope accurate rule (recorded, not adopted)

P3's never-worse coupled scaling, kept here as the proven partner-aware option for a future
revival.

**Constants.** s(N) = 62 − t(N), where t = ⌊log₂ T⌋ = `eps_max[N]` (constants.hpp):
55 / 31 / 8 / 0 / −3 / −15 at N = 2 / 8 / 14 / 16 / 17 / 20.

**Rule** (per batch):
- **A0.** The fast shifts σF, τF, unchanged; RN acceptance unchanged.
- **A1.** U_ih = ⌈|a_ih|·2^{σF_i + min(s,0)}⌉·2^{max(s,0)} ∈ [0, 2^63] (u64), from the (m, e)
  integers as the §3.3 quantizer computes. U′ is defined the same way.
- **A2.** The envelopes are E^A_h = max_i U_ih and E^B_h = max_j U′_hj (integer max, so
  independent of order).
- **B1.** R_i = Σ_h U_ih·E^B_h and C_j = Σ_h E^A_h·U′_hj, both < 2^166 for k ≤ 2^40 (6 limbs).
- **B2.** λ(X) = +∞ if X = 0; otherwise the largest L with X·2^{L−2s} ≤ H, found with one exact
  compare.
- **B3.** d_i = min(⌊max(λ(R_i), 0)/2⌋, 76 − uF_i), and e_j the same way; σ = σF + d,
  τ = τF + e.

**Lemmas.**
- **LA1 (ceiling).** |q(2^d·v)| ≤ ⌈2^d|v|⌉ ≤ 2^d·⌈|v|⌉ for integer **d ≥ 0**, for both roundings.
  It is false for d < 0 under RN (Appendix A, item 16).
- **LA2.** c_ih·2^s ≤ U_ih, where c_ih = ⌈|a_ih|·2^{σF_i}⌉.
- **LA3.** Σ_h U_ih·U′_hj ≤ min(R_i, C_j).
- **LA4 (mixed certificate).** For every (i, j), Σ|a′||b′| ≤ H. If d = e = 0, L1c applies.
  Otherwise Σ|a′||b′| ≤ 2^{max(λ(R_i), λ(C_j)) − 2s}·(R_i or C_j) ≤ H. The clamp and the *floor*
  of λ/2 are both necessary.
- **LA5 (ranges).** σ ≤ 1150 (kSigmaMax 1151), σ + τ ≤ 2300 (certified 2302), ε ≤ 76 (kEpsMax
  77), and |a′| ≤ 2^{uF+1+d} ≤ 2^77, so the W = 3 emitter's 78 bits suffice. This uses the
  corrected proof of Appendix A, item 17.
- **LA6 (never worse).** σ ≥ σF, so L1d gives 2^{−σ_i} < 2^{−d_i}·2‖a‖s/d̃. L3c then holds with
  gains g_a ≤ min d and g_b ≤ min e, and at g = 0 it reduces to §6f exactly.

**Pipeline.**
- KA scales each operand and contributes to the envelopes with atomicMax.
- KC re-reads the operand, couples, writes σ in place, and packs group 1 with W = 3.
- K2/K3, K4 and K5 are unchanged.
- New regions EA and EB (8·b·k̂ each) and AR are appended after W.
- Packed operands get NOT_SUPPORTED.
- Cost: +1 operand read, +2 launches, and a W = 3 pack at every N (§2.5).

**What a revival must solve.**
- Gains in the accuracy API need a mode argument or a flag, because `imd_err_estimate` has none
  (accuracy.cpp:178).
- New workspace regions conflict with the documented `IMD_REGION_COUNT` output
  (immerdink_host.h:149-151; tests/test_host_c.c:59).
- Transferring gains across N is unproven.
- "Envelope = full-GEMM bound" was shown only on separable fixtures.
- Packed operands need an answer, e.g. headroom packs with the corrected negative-increment
  rule.

**Revival trigger.** Stage 2 finds a realistic family with structured but **non-separable**
magnitudes where this rule saves ≥ 2 moduli beyond K-equilibration and K-segmentation on calls
of the 2048³ class or larger.
