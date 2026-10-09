# NOTES-C1 — Adaptive precision (ROADMAP C1) and partner-aware scaling (C2): design draft for DESIGN v1.19

## Status

**Status at the C1/C2 close-out, 2026-10-07: C2 done; C1 open (next: MC-3a, then M-d2)**
(branch `claude/ozaki-kernels-immerdink-1089fd`, not pushed). This block is the current
summary; C1's record continues with MC-3a. **Update, 2026-10-09: MC-3a has been executed
(§5.11): every §6c.5 class target is met on every set, one pre-registered in-sample
condition is not met under its literal reading, so no pass is declared; M-c stays FAILED,
the marks stay, the shipped calibration is unchanged, MC-3 is closed without a calibration
change, and M-d2 may run on its pre-registered bands. Independent review of MC-3a
(2026-10-09, binding, ruling A, §5.11): the in-sample condition is read against the shipped
fit's table, so MC-3a does not pass; M-c stays FAILED, the shipped calibration stays, and
whether C1's done condition "M-c has passed" is amended is Eric's decision after M-d2 (open
item for him; not amended here).** The bullets below that say MC-3a "has
not started" or M-d2 "waits" are the close-out's; §5.11 is current. The "Implementation status" below is the
detailed log, written as the work went; where they differ, this block is current. Every figure
here names its source; the libraries are `build/libimmerdink_host.so` sha256 `466acdb4…`
(host API 13, byte-identical to the design-phase [M] library, IR14) and, for the GPU records,
the `libimmerdink.so` builds named per item.

**Definition of done (§4.11, IR29), checked item by item against the branch and immerdink-eval:**
- **C2 is done.**
  - The M-a summary is published: immerdink-eval `findings/c1-evidence.md` (records
    `results/c1/m_a/`, `real/`, untracked).
  - Stage 3 has shipped, with L3c and L7 in Lean: DESIGN §3.8 "Segmented accumulation", PROOFS
    L7, `proofs/Proofs/{Error,Summation,Segments}.lean` (71 headline theorems in
    `proofs/AxiomCheck.lean`, standard axioms only, `tools/check_proofs.sh` passing after the
    review fixes at 4176732). Stored log: immerdink-eval
    `results/c1/final_review/check_proofs.log`, re-run at the final review on 57d7f97 (the
    Lean sources unchanged since 4176732; Lean 4.34.1, Mathlib `d13f23b`): 8932 jobs, 71
    theorems, only propext / Classical.choice / Quot.sound.
  - The stage-4 gate is decided by its pre-registered rule: **failed, stage 4 closed**, with the
    revival trigger "a real-operand family meeting (i)" (§4.5, DESIGN §7, ROADMAP C2).
  - The stage-6 gate is decided: **deferred**, with its trigger (§4.7, DESIGN §7, ROADMAP C2).
  - ROADMAP C2 is retitled per Q7 (IR18).
- **C1 is not done: two of its five conditions are not met.**
  - Met: stages 0 and 1 shipped with their tests and certificates (`host_footprint_monotone`,
    `host_precision`, `cert_footprint_monotone`, `cert_rho_recipe`).
  - Met: the round-2 stage-0 GPU tests pass, at 4ba78b1 (`libimmerdink.so` `c4bdcc4f…`;
    immerdink-eval `findings/c1-round2-stage0-gpu-tests.md`). The first-call measurement
    (IR10), one of §4.11's round-2 stage-0 GPU items, was not repeated there: it was measured
    at stage 0 (1e6e709, `libimmerdink.so` `b8baa75d…`; `findings/c1-stage0-first-call.md`)
    and still stands, because no file under `src/` has changed since 1e6e709 (`include/` in
    comments only). Its provenance is weaker than the rest of the branch (final review): the
    16 records carry only a date (no commit, library sha256 or host API), were written by
    `bench/first_call.py` before its first commit (8b58101) and were never re-run on
    committed code; the build is the summary's account, not a record field (DESIGN §6g.6).
    A re-run on the current build is in "What remains" below.
  - **Not met: M-c has not passed.** It ran on 2026-10-07 and failed one §6c.5 target: the
    < 100 µs median error is 15.8 % against 15 % (16.1 % in a repeat). The Q24 refit was not
    adopted (§5.9 MC-1). The only authorized path to a pass, the MC-3 feature(s) as amended
    (§5.9 MC-3a), ran on 2026-10-09 (§5.11): one optional key (`warm.kl_drop`, DESIGN §6c.9)
    fitted at 0.70, a fresh narrow-KL hold-out (4.3 / 15.7 and 2.7 / 6.6 %), the v1.18 hold-out
    on both sets, `mc.jsonl` with and without the repeat and a fresh full M-c run (1.8 / 7.6,
    1.8 / 7.4, 3.3 / 13.2 %) all meet the class targets; the in-sample condition fails
    literally on two K4 entries (+1.6 and +1.0 points, from Q24's records, not from the key),
    so no pass is declared. Source: immerdink-eval `findings/c1-m-c.md`,
    `results/c1/m_c/mc.jsonl` (sha256 `24fa699d…`, case list at 530ff81, profiling build at
    530ff81 with `libimmerdink.so` `9181dc3f…` and `libimmerdink_host.so` `466acdb4…`, the
    shipped calibration `d31f62ca…`, stock clocks with no lock as in §6c.5's validation).
  - **Not met: "M-d1 and M-d2 are reported with their success criteria evaluated".** M-d1 is
    reported and meets its criterion on every system; **M-d2 has not run**: the binding
    ordering made it wait for MC-3a (§5.9 "M-d2 ordering", §5.10 MD2-0); MC-3a's recorded
    outcome (§5.11) closes MC-3 without a calibration change, so it may now run. Its protocol and
    driver are pre-registered (`bench/accuracy/md2_time.py` at 35eeb46, sha256 `90fa7f92…`).
  - Met: the ROADMAP C1 row shows the Q6 decision column and "stage 5 deferred (triggers
    T1–T3)" (the latter added at this close-out).
- Every deferral names its trigger in DESIGN §7 or the ROADMAP: met (stage 4, stage 5, stage 6,
  the N-range plan, Ozaki-I slices, K-segmentation as a library mode, accurate mode, and the
  `PrecisionPlan` shared-pack role "A=B", whose trigger is in the ROADMAP C1 row since the
  close-out review).
- *Checks at the close-out* (2026-10-07; FREE, CPU only, one thread, under memcap; on fec6c42
  plus the close-out's edits, which touch only docs and docstrings; `ninja -n`: nothing to
  rebuild; `libimmerdink_host.so` `466acdb4…`, `libimmerdink.so` `c4bdcc4f…`): `tools/test.sh`'s
  18 CPU-only ctest tests pass (`host_precision`, `host_footprint_monotone`,
  `c2_engine_selftest`, `sweep_disjoint_selftest`, `md2_time_selftest` and `params_check`
  among them), `spec/params.py --check` passes 33 of 33 and `spec/test_oracle.py` ALL PASS.
  The GPU tests were not re-run: no file under `src/` or `include/` has changed since their run
  at 4ba78b1 (the later changes are bench tools, `imd_bench --data decay_opp` among them, and
  two CPU tests).
  Logs: immerdink-eval `results/c1/closeout/` (untracked); summary `findings/c1-closeout.md`.
- *Close-out review* (approve-with-fixes, 2026-10-07): this block retitled (C1 is open, not
  closed), IR26's premise corrected (§4.6, "Correction"), the trigger check's scope named, the
  first-call clause and the `PrecisionPlan` "A=B" item with its trigger added, "C1 is complete
  without stage 5" reworded. *Re-checked* after a FREE busy check, CPU only, one thread, on
  1c48022 plus the review edits (docs and the `monotone` docstring): `tools/test.sh`'s
  `cpu_only` list read verbatim, 18 of 18 pass; library sha256s unchanged. Logs:
  immerdink-eval `results/c1/closeout/review/` (untracked).
- *Second review* (approve-with-fixes, 2026-10-07, of c345582): the "Correction" in §4.6 and
  its pointers now keep the bar for the right reason (a time bar, measured directly) and
  allow `ncu` as a diagnostic only with `--clock-control none` (`ncu` 2026.3.1 defaults to
  `boost`, not `base` as c345582 said); HANDOFF's machine notes gain that `ncu` rule; the
  trigger check's passes are told in order; the first-call overlap reads "much of it";
  §4.11 says why C1 being open does not break IR29's "never as open"; the DIIS clause gives
  the 1.18e−15 to 2.09e−15 range (also in `bench/accuracy/schedule.py`'s docstring).
  *Re-checked* after a FREE busy check, CPU only, one thread, on c345582 plus these edits:
  `tools/test.sh`'s `cpu_only` list read verbatim, 18 of 18 pass (172.2 s); `schedule.py` runs
  (default and `--relaxed`); library sha256s unchanged. Logs: immerdink-eval
  `results/c1/closeout/review2/` (untracked).
- *Final review* (three independent reviewers, approve-with-fixes, 2026-10-07, of 57d7f97:
  the numerics, the host and Python code, and the documents and provenance). Applied:
  - *Must-fix:* `params.kseg_request` returns None when no N is certified, never 0.0, which
    rank reads as "no target" (§5.7 S3-1, DESIGN §3.8 steps 2–3: t′ > 0, basis BOUND);
    DESIGN §6g.6's first-call overlap reads "much of it" (46–68 % at 1024³); K-equilibration's
    cost on the real operands is TFLOPS-eq first with time alongside and the outside-hold-out
    tag ("2–47 % lower TFLOPS-eq ... (+2 % to +89 % time)", Status above, §4.5, DESIGN §7,
    ROADMAP C2).
  - *Numerics:* DESIGN §3.8's cheap r (upper band norms, LOWER whole-row norms, zero rows
    skipped, min(1, ·), RN norms enclosed); the rigorous ρ_S carries the absolute underflow
    term `params.kseg_eta(S)` (`rho_recipe`'s new `eta`), so it no longer assumes every D_s
    normal, which no caller can check (PROOFS L7c (7)); `cert_kseg_composition` checks
    `kseg_wbar_up` exactly up to S = 2²⁶ (an S − 1 mutant fails), γ's size with a two-band
    K-empty witness (L7 met to 0.996; a halved-γ mutant fails), ρ_S on outputs whose last
    band underflows, and the None case; spec/test_oracle.py (13) adds a K-empty first band
    with β = 0.3, 3.0, −0.7 (a two-rounding β step fails it); `host_precision` checks U with
    ℓ1 ratios; PROOFS and proofs/NOTES.md name exactly which L7c forms Lean states, and why a
    zero band row enters only through `kseg_compose`.
  - *Host and Python:* `PrecisionPlan` refuses, at construction, a share group whose shared
    roles read different packs or that no level can pack (DESIGN §6g.10; it had accepted
    two operands of different shapes and sized one pack for them); `phase()`'s per-call
    cost (0.2–0.7 ms on `schedule.py`'s contractions) and the cheaper pattern (re-run it
    only when a step-1 `min_moduli` result changes, a claim `host_precision` checks) are in
    DESIGN §6g.10 and the docstrings; `PackedBuffer` assumes one host thread per buffer
    (DESIGN §6g.5); `host_precision` lowers every `configs()` and `phase()` choice with its
    `options()` and `pack_specs` (no PACK_MISMATCH, its workspace) and a share group's
    members with one PackSpec, and `python_device`'s CPU part makes those Plans and Packs
    through its fake library; `bench/first_call.py` records the host API.
  - *Records and documents:* the first-call records' provenance limit is stated (above,
    DESIGN §6g.6); MC-3a's runs record library sha256s, host API and the dirty flag at run
    time (§5.9 MC-3a item 6); `run`'s bit-exact check is to fill C before each call at the
    next re-registration, required before any M-d2 run (§4.3; not applied now, since that
    re-registers the driver); the synthetic judge-check records are moved out of
    `results/c1/m_d2/` (§4.3 "M-d2" records); the M-b tally names its 9 information-only
    figures; DESIGN's change log no longer says every record the gates rest on is
    reproduced; DESIGN §7's K-segmentation row names R2b; HANDOFF lists the items for Eric;
    PLAN and SCOPE carry the outcomes; the Lean check has a stored log (above).
  - *Verdicts on the review questions:* L7 (L7a–L7d, the request transform, ρ_S) is sound;
    the Lean proofs build with standard axioms only and state what PROOFS claims;
    `rho_from_output` and `rho_recipe` are sound with strong tests; `oracle.contract_segments`
    is a faithful reference and `oracle.kweights` correct; `cert_footprint_monotone`
    supports F1/F2 and I1–I3; the ABI and version claims are consistent (host API 13, no
    bump); planner.py computes nothing; the GPU-lock usage is correct; C2's definition of
    done is met and C1's open state is reported consistently; no external-library
    comparison is in the product repo; no claimed re-run was missing.
  *Re-checked* after FREE busy checks, CPU only, one thread, under memcap, on 57d7f97 plus
  these edits: listed in immerdink-eval `findings/c1-final-review.md` (logs in
  `results/c1/final_review/`, untracked).

**What shipped.** No ABI, calibration-format or TLA+ change: host API 13. `src/` has not changed
since the design draft (39972b7), `include/` only in comments.
- *C1(a), precision phases* (DESIGN §6g.1–§6g.9): a plan and a pack per N; F1–F3, C-ws and
  C-pack published from API 13 on; certificates I1–I3; the re-pack-in-place rule;
  `PackSpec.with_moduli` and `PackedBuffer`. No plan-time preload: the first call at a
  never-launched N takes at most 9 µs more than a steady one end to end (enqueue to
  completion), with 12–35 µs more host enqueue time, much of it overlapping the GPU; a cold
  capture's first replay is 0.98–1.10× the steady one (1e6e709 build, `libimmerdink.so` `b8baa75d…`;
  immerdink-eval `findings/c1-stage0-first-call.md`).
- *C1(b) on the host* (DESIGN §6g.10–§6g.12): `PrecisionPlan` and the rigorous
  `rho_from_output`, and the M-d1 caller guidance, carried into DESIGN §6g.12 at this
  close-out (§1.3 keeps it with its evidence). M-d1 (CPU, bit-exact emulation): the forcing-rule ramps match fixed N = 14 on
  Jacobi, DIIS(8) with and without a reset, and PPL + ring; the lagged ρ̂ missed in 1 of 228
  MET calls. Source: `findings/c1-evidence.md`, records `results/c1/m_d1/`, reproduced bit for
  bit at 928b49c (jacobi, davidson, cg) and 5925219 (diis, ppl); the ρ-miss records are tied to
  committed code by sha256 (§5.6 S2-18).
- *C2:* `oracle.kweights` and the M-a study, the C2 deliverable now that the gate has failed
  (Q8); the K-segmentation caller recipe with its rigorous bound and request transform
  (`cert_kseg_composition`, `oracle.contract_segments`), and L3c, γ_k and L7 in Lean.
- *Tooling kept for the open work:* M-c's sweeps and `bench/sweep_disjoint.py`, the M-d2 driver,
  and the CPU ctests `c2_engine_selftest`, `sweep_disjoint_selftest` and `md2_time_selftest`.

**What was closed or deferred, and why.**
- *Stage 4 (caller-side K-equilibration), closed by its gate.* No real R1–R6 contraction of the
  ≥ 2048³ class (psi4 DF and MP2 operands of benzene and water on ccsd.py's contraction list)
  passes the guard: G ≤ 0 on all of them, and with the caller's costs K-equilibration reaches
  2–47 % lower TFLOPS-eq than fast mode there at the normwise targets (+2 % to +89 % time;
  N = 9–12) [M, model-only, outside hold-out] and 1–36 % lower at the ew8 targets (+1 % to
  +57 % time; N = 13–16) [M, model-only, outside hold-out at N = 13 and 15]. (ii) is not met
  as pre-registered either: 2–19
  sampled outputs per guard-accepted Tier-L fixture above 2× fast's error, max ratio 247.
  Source: `bench/accuracy/c2_summary.py` over records reproduced bit for bit at 5925219
  (`results/c1/repro_5925219/`), M-a's [M] blocks reproduced by M-b on `466acdb4…`; the
  deadband d* = 3 comes from Tier S, which is not reproduced on committed code (4 of 245 runs
  smoke-checked), and the verdict holds at d = 0 too. Revival trigger: a real-operand family
  meeting (i), judged with the re-registered (ii), after the d*-deciding Tier S subset is re-run
  (§4.5, §5.6 S2-21).
- *Stage 6 (library-internal weights), deferred:* its gate needs a shipped stage 4. Trigger
  §4.7.
- *Stage 5 (K5 statistics and the device certificate), deferred (IR26):* no consumer issues
  elementwise targets (ehrenfest D8; ccsd.py issues no accuracy request), and the host-side
  `rho_from_output` gives a rigorous ρ now. Triggers T1–T3 (§4.6). *Trigger check at the
  close-out* (IR28's "a trigger check only"; read-only, 2026-10-07). The first check, at
  1c48022, covered `~/repos/ehrenfest` (c7983281) only. The close-out reviewer widened it to
  the ehrenfest checkouts on this machine, 21 in all: `~/repos/ehrenfest` (c7983281),
  `~/repos/ehrenfest-main` (36a76b07) and all 19 worktrees under `~/repos/ehrenfest-ipdps`.
  The review's fix round re-ran it over all 21; that run is the record (immerdink-eval
  `results/c1/closeout/review/trigger_check.txt`, 2026-10-07T23:21:53Z), and the second
  review repeated it on the same HEADs. No Python file there imports immerdink, and none uses
  `err_request`, `min_moduli`, `PrecisionPlan` or `rho_from_output` (its "ELEMENTWISE" matches
  are its tiling code's elementwise operations, not targets); the other mentions of immerdink
  are notes on the shared GPU and its lock (`docs/formal/device_release/SPEC.md`, scratch notes
  and a scratch GPU-check script). So T1 and T2 have not fired; T3 has no record.
- *Also decided:* accurate mode re-rejected (§2.8; its Appendix B trigger is not met, 0 cases);
  the N-range plan and Ozaki-I slices parked with triggers (§3.2, §3.3); K-segmentation and B6
  stay caller recipes (Q15 (a); B6 shows no value at the sizes measured).

**What remains for C1** (round 2, GPU; updated 2026-10-09): MC-3a is done (§5.11) and MC-3
is closed without a calibration change (independent review, 2026-10-09, binding ruling A:
the one in-sample condition is read against the shipped fit's table, so MC-3a does not pass,
§5.11 "Verdict"). Whether C1's done condition "M-c has passed" is amended is Eric's decision
after M-d2 (open item for him; not amended). What remains is M-d2 (§4.3; the driver re-registered with the NaN-filled bit-exact check first, then
eight `run` processes into one records file and one `judge`). In the same GPU round, a
re-run of `bench/first_call.py --sweep` on the current build (about a minute of GPU), whose
records name the commit, the library sha256s and the host API (final review; DESIGN §6g.6). Until M-c passes, figures at N ∉ {8, 10, 14, 16} keep
the mark [M, model-only, outside hold-out]. Open outside C1: a `PrecisionPlan` member role
"A=B" for a self-contraction reading one shared pack through both roles (M-b below; named
"AB" before the close-out review, which renamed it because `"AB"` is the pack-mode value for
two separate packs). Trigger, in the ROADMAP C1 row: the ehrenfest-side integration of the
B2 performance model declaring DF-factor sharing.

**Revised after two independent reviews, 2026-10-07, then amended by the binding decisions of the
independent design reviewer, 2026-10-07** (branch `claude/ozaki-kernels-immerdink-1089fd`). The
text below was written in the design phase, before anything was built, tested or run on the GPU;
what has been built since is recorded under "Implementation status".

**Implementation status.**
- **Stage 0 (§4.1) is done, 2026-10-07: DESIGN v1.19 §6g**, approved with fixes by an independent
  reviewer; the fixes are applied, and the reviewer's verdicts on the implementation's design
  questions are §5.4. A re-review (approve-with-fixes) amended one verdict: F1 no longer
  promises an up-set of succeeding N (§5.4 S0-2). Its fixes are applied too.
  - **IR14:** this worktree's libraries were rebuilt first (`tools/test.sh`, host API 13; `nm`
    shows `imd_perf_rank`, `imd_perf_predict_pack`, `imd_err_min_moduli`).
    `build/libimmerdink_host.so` has sha256 `466acdb469b06e404a202c0d3f2ce5ab4f27d3c0f881984ecd4b9122648810fc`,
    byte-identical to the agent-ac829dae library of the [M] figures, so every design-phase [M]
    figure is a figure of this worktree's library. The review rounds' rebuilds left it
    unchanged. The 1e6e709 build of `build/libimmerdink.so`, which produced the stage-0 GPU
    records, had sha256 `b8baa75d2d2869341431c9a7667b7a61ab03392d9ced21b16ef2b473abfbec85`.
    Later rebuilds after header-comment-only changes (0197351 and the re-review's) changed its
    bytes but no source: no file under `src/` has changed since 1e6e709. The 0197351 build had
    sha256 `9a9ad2ea12689905c1bd8e1b83d95fa2406b529f765ecbe6750d1ee0014424de`, and the
    re-review's build has `c4bdcc4f696a6c87ec8049ed7c6ebf1ce4eae44a775ce5cde6c7783d298c4740`.
    `bench/first_call.py` now writes the library's path, its sha256 and the git commit into
    each record.
  - **IR9:** the 61 design-phase files are in immerdink-eval (Provenance below), listed with
    their sha256 in `findings/c1-design-phase-manifest.md`; the scripts and the manifest are
    committed there (7bdac3d), the records stay untracked.
  - **IR10, closed by measurement: no plan-time preload.** Cold capture works at every N
    measured (`graph_phases`; `bench/first_call.py`), and the first replay of a cold capture is
    0.98–1.10× the steady one, against the 2× bar. The first call at a never-launched N takes at
    most 9 µs more than a steady one, end to end (DESIGN §6g.6; immerdink-eval
    `findings/c1-stage0-first-call.md`).
  - **IR17, evidence passed: F3's second half is published** (DESIGN §6a). `pack_write_extent`
    and `pack_write_extent_memcheck` cover 113 packs and assert that the cases reach all 42
    launch shapes of `imd_pack`'s KL, the radix > 1 row mapping included (added after the
    review found every earlier case at radix 1), plus `kp_prep`'s two instances and a gathered
    K; sentinels and memcheck are clean, and the 16-byte-short mutation is reported.
  - **IR18:** ROADMAP C1's decision column and C2's title are updated (Q6, Q7).
- **Stage 1 (§4.2) is implemented, 2026-10-07: DESIGN v1.19 §6g.10–§6g.12**, Python only
  (host API 13). `python/immerdink/precision.py` (`PrecisionPlan` with the IR2 composition,
  masks and BUDGET, joint share groups, the DIRECT sentinel; `rho_from_output`),
  spec/params.py `rho_recipe` and `cert_rho_recipe`, the CPU test `host_precision` (a brute
  force from host.py's queries), and the example `bench/accuracy/schedule.py`. It was
  approved with fixes by an independent reviewer; the fixes are applied, and the verdicts on
  the implementation's design questions are §5.5. The binding ones: a share-group member at
  DIRECT leaves the group for the phase, `BUDGET` only on a met target (S1-5, S1-3), and a
  1-D norm bound of `rho_from_output` is always axis −2 or −1 (S1-7). A re-review approved
  with one more fix, applied: a `rho_from_output` bound of ndim ≥ 2 must lie along its own
  axis and carry none or all of C's batch dims (MF3, S1-7). It made S1-10 binding as
  implemented (a shared role's reuse stays fixed while a member is at DIRECT) and confirmed
  that groups joined only through a DIRECT member are ranked separately (S1-5). A second
  re-review approved with one more fix (MF4, S1-7, binding), applied: on a batched C
  (C.ndim ≥ 3) a `rho_from_output` bound is a scalar or has exactly C.ndim dims, and every
  1-D or 2-D bound raises, since per-batch scalars written without keepdims, (b,) or (b, 1),
  were paired with C's rows or columns where b = m or b = n and certified ρ̂/ρ down to 0.0081;
  a 2-D C keeps the S1-7 rule. `host_precision` checks it, a mutation that drops the rule
  fails it, and the only in-repo caller (`bench/accuracy/md1_converge.py`, 1-D bounds on a
  2-D D) is unaffected. A third review approved MF4 as implemented (S1-11, binding) with
  should-fix items, applied: a test of bounds shared over one batch axis only, norm bounds
  converted to float64 exactly or refused (an integer above 2⁵³, a long double, Fractions,
  Decimals), and the GEMM view of a tensor output stated (S1-7). The
  provenance of DESIGN §6g.10's cost figures (IR9 style) is
  immerdink-eval `findings/c1-stage1-phase-cost.md`, with the script committed there
  (39d9f9f) and both records untracked, listed with their sha256; a re-run on 51dba1e
  reproduces the first record to within 0.04 ms.
- **Stage 2's CPU evidence (§4.3: M-a and M-d1) is done, 2026-10-07**, approved with fixes by
  an independent reviewer; the fixes are applied, and the verdicts on the implementation's
  design questions are §5.6. A re-review (approve-with-fixes) found the provenance statements
  wrong; its fixes are applied too, and its verdicts are §5.6 S2-14–S2-18. A second re-review
  (approve-with-fixes) found that the provenance text still claimed more re-running than had
  been done; it is corrected below, the remaining records are re-run on committed code (all
  bit for bit), and its verdicts are §5.6 S2-19–S2-20. A third re-review (approve-with-fixes)
  found that "gates (i) and (ii) rest on reproduced records only" left out the deadband
  (d* = 3 comes from Tier S, not re-run); corrected below with the d = 0 figures, and its
  verdicts are §5.6 S2-21–S2-22.
  - *Code:* `oracle.kweights` with its multi-partner mode and test_oracle (12) at 5cd8ff3; the
    study (`bench/accuracy/kw_engine.py`, `c2_fixtures.py`, `c2_gate.py`, `c2_summary.py`,
    `md1_converge.py`) at 7260a95; the review fixes at f97696a; the re-review's at 928b49c.
    The CPU ctest `c2_engine_selftest` (one of `tools/test.sh`'s CPU-only tests) runs the
    engine's self-test and checks the several-partner and batched paths against
    `oracle.kweights`, asserting its own coverage.
  - *Summary:* immerdink-eval `findings/c1-evidence.md`. It replaces the `findings/c1-m-a.md`
    named in §4.3 and covers M-d1 too. The raw records are in `results/c1/m_a/` and
    `results/c1/m_d1/`, the psi4 operands in `results/c1/real/`, all untracked.
  - *Provenance* (corrected in the re-review, from the `commit` field of every file). The study
    code was uncommitted when the first records were written, so they name the commit at run
    time, not their code:
    - 8b58101: Tier S (`S.jsonl`) and the pilot;
    - 52054e9: Tier L (`L.jsonl`), the real operands (`real.jsonl`), `multi`, `batch`,
      `real_pre_ew8`, the 7 real-operand B6 records, and M-d1 jacobi, diis, ppl and
      `rho_levels14`;
    - d22a686: M-d1 cg and davidson (with the superseded `cg_v1`, `davidson_v2` and
      `davidson_strongcoupling`) and `rho_local`.

    Three reproduction checks tie them to committed code (corrected in the second re-review:
    the earlier text claimed more re-running than had been done). Each compares every field
    of a re-run record with the stored one exactly (immerdink-eval `scripts/c1/repro_check.py`;
    provenance and resource fields skipped; a field only the re-run has, added after the
    stored record was written, is listed; a field only the stored record has fails unless
    explicitly allowed):
    - *Smoke check, first round:* Tier S only, 128 × 600, phi4 and decay_opp, seeds 0–1, on
      7260a95, bit-identical apart from the then-new ρ ≤ 1e8 column: 4 of Tier S's 245 runs.
    - *Spot check, re-review, at 928b49c* (every study file committed;
      `results/c1/repro_928b49c/`): 3 of the 66 Tier L records (decay_opp 2048 × 3600 seed 0,
      opp_nonsep 2048 × 16384 seeds 0–1, seed 1 with seed 0's stale weights), 2 of the 11 real
      ones (R2 and R2b of benzene/cc-pVDZ), 1 of 2 multi, 1 of 2 batch, 1 of the 7 real-operand
      B6 records (R2b), and all of M-d1 jacobi (11), davidson (6) and cg (7). All bit for bit;
      beyond Tier L's ρ ≤ 1e8 column, the stored R2b real record's `--b6` block is absent from
      its re-run (run without `--b6`; the B6 job reproduces it). It did not run M-d1 diis
      (22 records), ppl (11) or `rho_levels14` (4), and the jacobi re-run does not exercise
      their code: the DIIS branch of `md1_converge.iterate` (Pulay extrapolation, the history
      reset at a precision switch) and the PPL class (two calls per iteration, the ring
      transposition, its own setup). Its records alone make gate (ii) fail: at d*,
      opp_nonsep 2048 × 16384 has 7 sampled outputs above 2× fast's error at seed 0 and 6 at
      seed 1 (max ratio 247), guard14 = 1 (G = 14) in both.
    - *Full re-run, second re-review, at 5925219* (every study file committed;
      `results/c1/repro_5925219/`, the driver immerdink-eval `scripts/c1/repro.sh`): all 66
      Tier L, all 11 real (with `--b6`), 2 multi, 2 batch and the 7 real-operand B6 records,
      and M-d1 diis (22) and ppl (11): **121 records, all bit for bit**, no stored field
      missing; the only fields added are Tier L's ρ ≤ 1e8 column and the real B6 blocks'
      later `rho_cap` / `excluded_outputs`. So every 52054e9-era record that is not
      superseded is reproduced on committed code: M-d1 jacobi in the spot check at 928b49c
      (`md1_converge.py` has the same sha256 at both commits), the rest in this re-run at
      5925219 (the commit messages of 04363db and immerdink-eval beb07e7 credit all of it to
      5925219; corrected in the third re-review). With the spot check's davidson and cg,
      every M-d1 record behind the success criterion and the caller guidance (jacobi, diis,
      ppl, davidson, cg) is reproduced.
    - *What gates (i) and (ii) rest on* (third re-review: the earlier "on reproduced records
      only" left out the deadband):
      1. The records they are evaluated on are all reproduced: guard = 0 on every real
         record (R1–R7; gate (i) reads R1–R6), and all 16 guard-accepted Tier-L records at
         d* (decay_opp and opp_nonsep at every size, 0–7 sampled outputs above 2× each, 2–19
         per fixture, max ratio 247).
      2. The deadband they are evaluated at, d* = 3, comes from Tier S, which is not
         reproduced on committed code (next item).
      3. The verdict holds at both deadbands that Tier L and the real operands compute, 0 and
         3 (they compute no other, so a different d* could not be evaluated on these records).
         At d = 0 every R1–R5 record (R2b included) has guard14 = 0 (G14 from −5 to 0), and
         R6 and R7 have Δ ≡ 0, so w ≡ 0 at any d. Gate (ii) fails at d = 0 too: the same 16
         Tier-L records (6 fixtures) are guard-accepted, with 1–14 sampled outputs above 2×
         fast's error per fixture (0–6 per record), max ratio 99.1 (opp_nonsep 2048 × 3600).
         Checked on `repro_5925219/`.
    - *Not re-run:* Tier S beyond the smoke check (241 of 245 runs at 8b58101). Tier S, d* = 3
      and the trend over m are **not reproduced on committed code (8b58101-era,
      smoke-checked 4 of 245)**, nor is Tier S's share of gate (iii); before a stage-4
      revival (its trigger: a real-operand family meeting (i)) or any paper or external use
      of Tier S figures, the d*-deciding subset is re-run on committed code under a newly
      declared budget (§5.6 S2-21). Also not re-run: the pilot (a sizing run no table uses);
      the superseded files (`real_pre_ew8`, `md1_rho_levels14`, `md1_rho_local`, `cg_v1`,
      `davidson_v2`, `davidson_strongcoupling`); and the records written since the first
      round, which carry a dirty flag and the sha256 of every study file instead: the review
      round's 30 B6 records (51dba1e, a858225) and the IR7 re-run's 4 `md1_rho` records
      (d89988a), whose files match f97696a's (S2-18).
  - *CPU budget:* M-a about 51 of its 90 CPU-min: the first round's about 36 (Tier S 24,
    Tier L 8, real operands 3, self-tests and pilot 1), including about 6 min of duplicated
    work by an orphaned process (§5.6 S2-7), then the review's B6 runs at m = 512 (1.6 min),
    the re-review's spot check (1.7 min) and the full re-run at 5925219 (11.8 min); M-d1 about
    27 of its 30, including the IR7 re-run (1.7 min), the re-review's spot check (4.8 min)
    and the diis and ppl re-runs at 5925219 (5.7 min).
  - **Mechanical outcomes** (the pre-registered rules, IR27):
    - d* = 3 (Q11; from Tier S, not reproduced on committed code: 8b58101-era, smoke-checked
      4 of 245). Every deadband rejected all 30 random-structure runs, and the median gains
      on decay_opp and opp_nonsep are within 0.25 moduli; ties go to 3. The gate verdict
      below holds at d = 0 as well (*What gates (i) and (ii) rest on*, item 3, above).
    - **Stage 4 fails its gate (§4.5) and is closed**, with the revival trigger "a real-operand
      family meeting (i)". (i): no real R1–R6 contraction of the ≥ 2048³ class has guard = 1
      (G ≤ 0 on all of them). (ii) is not met as pre-registered either: each guard-accepted
      Tier-L fixture has 2–19 sampled outputs above 2× fast's error, max ratio 247. (iii)
      holds: no FP64-clamp violation. `oracle.kweights` and M-a remain C2's deliverable (Q8).
    - **The gate was applied at the end of round 1**, although IR28 places the gate decisions
      in round 2. That is admissible because the failing part of (i), guard = 1 on a real
      R1–R6 contraction, is a CPU-only quantity that no round-2 measurement can change. M-e
      needs a stage-4 build and is not run.
    - **Stage 6 is deferred** (§4.7): its gate requires a shipped stage 4.
    - Q12 (a) stands for this cycle, but vacuously on real data (§5.6 S2-12). Q15 (a) stands
      (§5.6 S2-13). The Appendix B revival trigger is not met: 0 cases, and the largest saving
      beyond both alternatives is 1 modulus.
    - B6 (IR11) shows no value at the sizes measured. No split beats running every row at
      N_hi, on the real operands (FP64 x) or on synthetic outliers, opp_nonsep and φ = 4 at
      m = n = 128 and 512 (exact x). At m = n = 512 the reason is the refinement call's fixed
      part: in [M], t(N_hi, m' rows) costs 53–57 % of t(N_hi, full) even for m' = 1–46 rows
      (59–65 % of it packs the whole partner at N_hi), while one modulus less on the bulk
      saves 1.7–3.8 % (immerdink-eval `scripts/c1/b6_split_cost.py`). B6 stays a caller
      recipe; a much larger m with few high-ρ rows is untested.
    - M-d1: every forcing-rule ramp meets §4.3's success criterion on Jacobi, DIIS(8) with and
      without a reset, and PPL + ring. **IR7 is closed:** with the shipped `rho_from_output`
      (51dba1e's), the lagged ρ̂ missed in 1 of 228 MET calls (PPL at 1e−6, realized 1.013e−6,
      after an output passed near zero). The report-only cases became caller guidance (§1.3).
  - **Not run in stage 2:** M-b (the [M] re-runs of IR9 and the share-group raise cost of
    IR15; CPU, round 1), and M-c and M-d2 (GPU, round 2). Stage 3 could start once the M-a
    summary was published (§4.11). Stage 3, M-b and M-c have since run (below); M-d2 has
    not.
  - **Done at the close-out (2026-10-07):** the M-d1 caller guidance (§1.3: the DIIS reset,
    Davidson's recompute and its +5 iterations, CG's relaxation scaled by the iteration count,
    and the lagged ρ̂'s miss rate) is in DESIGN §6g.12, pointed to from §6g.10, the
    `PrecisionPlan` docstring (`monotone`) and `bench/accuracy/schedule.py`'s docstring.
- **Stage 3 (§4.4) is done, 2026-10-07**, after the M-a summary (IR12, IR28); CPU and Lean
  only, no ABI change (host API 13). Implemented at 7b04b04, then approved with fixes by an
  independent reviewer; the fixes are applied at 4176732, and the verdicts on its design
  questions are §5.7.
  - *DESIGN v1.19 §3.8 "Segmented accumulation":* the K-segmentation caller recipe (§2.7, Q15
    (a)), L7's bound in every metric with its caveats, a rigorous request transform for
    choosing N (L7d, §5.7 S3-1) and M-a's evidence on when it pays. The §7 rows are rewritten
    and cite M-a (IR12): final, not "provisional", since the M-a summary is published. [M]
    figures at N ∉ {8, 10, 14, 16} stay tagged until M-c (IR4).
  - *PROOFS L7* (L7a recursive summation γ_{S−1}, L7b the bound, L7c the metric forms, L7d
    the request); L5 and L6 stay reserved.
  - *Lean (IR5): done.* L3c (the ν form `l3c_unit`, the NORMWISE / ELEMENTWISE / ABSOLUTE
    compositions with the normal-range final rounding and the subnormal term, the native
    chain `fma_chain`), the summation lemma γ_k (`Summation.recsum`) and L7 (`kseg_compose`,
    `kseg_rows` from the band rows, `kseg_normwise`, the exact half of L7d) in
    `proofs/Proofs/{Error,Summation,Segments}.lean`; `tools/check_proofs.sh` passes (71
    headline theorems after the review fixes, standard axioms only). So the "C2 done"
    condition "stage 3 has shipped, including L3c and L7 in Lean" (§4.11) is met.
  - *spec/params.py:* `kseg_gamma`, `kseg_unit`, `kseg_request` and `cert_kseg_composition`
    (§4.4's grid, plus underflowing products, the request transform, an actual violation for
    the whole-row ν counterexample, the lost single rounding at N = 20, and the necessity of
    γ). *spec/oracle.py:* `segment_bounds`, `contract_segments`; spec/test_oracle.py (13).
  - *ROADMAP* C2 records stage 3; Q6 and Q7 were applied in stage 0 (IR18).
  - *Review fixes* (§5.7): a band of k_s = 1 is K-empty and runs DIRECT, as the library
    dispatches it (`contract_segments` had emulated it, so it was not the bit-exact
    reference there; test (13) now also runs the recipe's calls through `contract_tensor` on
    moved sub-views, and the certificate has K-empty bands); DESIGN §3.8's ELEMENTWISE ρ_S
    is the rigorous N_S/(|C_S| − W̄·N_S) from a previous segmented output (PROOFS L7c (7),
    Lean `kseg_wbar_le` / `kseg_rho_le`, `params.kseg_wbar_up`, `cert_kseg_composition`
    (7)); the request transform is scoped to α = 1 or 2^e and β = 0, with r and ρ_S rounded
    upward, its guarantee is W ≤ t̄ and "t′ ≤ 0" means no N is certified; the underflow
    hypothesis of NORMWISE and ELEMENTWISE, L7d's normal-range step and the reference of a
    non-contiguous partition are stated; `contract_segments` takes A = B = None at α = 0
    and rejects non-integer bounds; the certificate's tautological S = 1 check compares with
    `error_estimate`'s own NORMWISE bound; ROADMAP C2's comparison with K-equilibration is
    scoped to the guard-accepted Tier-L cases.
- **Round 2, the stage-0 GPU tests (§4.11): pass on committed code, 2026-10-07**, at 4ba78b1
  (host API 13; `libimmerdink.so` sha256 `c4bdcc4f…`, the stage-0 re-review build above, and
  `libimmerdink_host.so` `466acdb4…`; `tools/test.sh`'s build step compiled nothing).
  `packed_repack_n`, `workspace_shared_n`, `graph_phases` (cold captures at N = 10 and, packed,
  N = 12, bit-exact), `pack_write_extent` (113 packs, 612 packed contracts, all 42 KL launch
  shapes, 12 of them at radix > 1) and `pack_write_extent_memcheck` (clean; the 16-byte-short
  mutation is reported) pass, and so does everything else: `tools/test.sh`'s 56 ctest tests (the
  GPU ones in chunks under the GPU lock), `spec/params.py --check` (33 of 33) and
  `spec/test_oracle.py`. Nothing needed fixing. So C1's done condition "the round-2 stage-0 GPU
  tests pass" is met.
  - *IR17:* F3's second half stays published as stage 0 left it (DESIGN §6a, §6b.4, §6g.2,
    §6g.8, the `imd_pack` comment in `immerdink.h`); it is not re-published.
  - *IR10:* the verdict (no plan-time preload) is DESIGN §6g.6, now naming its 1e6e709 build.
    The first-call benchmark was not re-run: no file under `src/` has changed since that build.
  - *Records:* the logs are in immerdink-eval `results/c1/round2_tests_4ba78b1/` (untracked,
    with a manifest of the commands, the conditions and each log's sha256); summary
    `findings/c1-round2-stage0-gpu-tests.md` (b352778).
- **M-b (§4.3; IR9, IR15 (3), IR23) is done, 2026-10-07**: CPU only, one thread, under memcap,
  FREE only, on this worktree's `build/libimmerdink_host.so` (sha256 `466acdb4…`, host API 13;
  `ninja -n` finds nothing to rebuild at 4ba78b1). Summary immerdink-eval
  `findings/c1-m-b.md`, scripts `scripts/c1/m_b_rerun.py` and `m_b.py` (96f7c4f; review
  fixes 601253f), records `results/c1/m_b/` (untracked, sha256 in the summary).
  - *The design-phase [M] records reproduce byte for byte:* the design-phase scripts, run
    unchanged with only the library path redirected, print `perf.json`, `perf_summary.md`,
    `c2cost.md`, `nsweep.out`, `phases.out`, `predict_cc.out` and `slice_eval/cost.out`
    identically. So do stage 2's [M] records: M-a's `perf` blocks (66 Tier L, 9 real),
    `c2_summary.md` and `b6_split_cost.txt`. No prediction changed.
  - *The cited figures:* 191 [M] and [E] figures of §2.3–§2.7, §4.2, §4.3 and DESIGN §3.8,
    recomputed from fresh queries: 9 are information only (§2.7's [M] composition of the S
    calls, which M-b added, so there is no citation to compare); of the other 182, 169 print
    as cited; 7 differ in the last digit only
    (≤ 0.42 %: mis-roundings of unchanged predictions just below a tie, corrected in place after
    the M-b review, §2.3 and §2.6); 6 differ by more. One is a text error, corrected in
    §2.3: the 4096² re-pack is 9–11 % of one call, not "about 11 %" (8.8 % at N = 14). Five
    are [E] estimates, kept and annotated: §2.7's first-round K-segmentation costs at S = 4 are
    3× the rounded S = 2 figures (reconstructed: +52 / +95 / +31 %, and +32 % for DF at S = 2),
    and §4.2's "about 9 %" is 9.5 % at 0.9 TB/s. §2.7 now also gives the [M] composition of the
    S calls, M-a's method: +13 / +24 / +8 % at S = 2, +40 / +72 / +24 % at S = 4.
  - *PrecisionPlan's own choices* reproduce the §2.4 ratios (g = N at every level), and M-d2's
    bars are 0.750 (PPL view), 0.664 (DF) and 0.673 (4096³) unrounded (§4.3).
  - *The share-group raise cost* (IR15 (3); §3.3; [M], outside the hold-out at levels not in
    {8, 10, 14, 16}): on ehrenfest ccsd.py's pairwise DF contractions at the psi4 sizes, 0 in
    11 of 12 forcing-rule cases with the real norm products and +16.5 % in the twelfth
    (benzene/cc-pVDZ at 1e−8, through R5) on the default levels 6/8/10/12/14 (+8.1 % on the
    integer levels 6–14 and on 6/8/9/10/14); −32 % to +25 % of the separate-packs time per
    iteration when one member's floor is a level apart, up to +102 % from N = 6 to 14. The
    joint ranking chose the group floor in all 292 scenarios (316 with the other grids). The
    B role of R1, R2, hh and Lld1 is modelled in-line and the norms are the bare factors'
    (§3.3 limits).
  - *Review fixes (approve-with-fixes, §5.8), 2026-10-07:* the raise is not what a slice
    revival could save; §3.3 and DESIGN §7 now give that bound from the same records: 0 in
    all 12 forcing cases, up to 9.1 % (one level apart) or 16.0 % in the synthetic patterns
    [M, model-only]. The raise figures carry their level grid and modelling limits (§3.3); the
    pack keys are checked at every level 6–14; the 7 mis-rounded cells are corrected (§2.3,
    §2.6), as are §4.2's peak-bandwidth figure (8.5 %, not 8.6 %) and §2.4's "at most 0.004".
    `m_b.py` was re-run at eval 601253f (CPU, one thread, under memcap, FREE): the earlier
    records reproduce bit for bit on their old fields (`figures.jsonl`, `schedules_pp.jsonl`
    byte for byte, 292 of 292 scenarios).
  - *Open (outside C1 scope):* `PrecisionPlan` cannot express one contraction reading a
    group's single pack through both roles: `_share` rejects a member listed twice, and a role
    is "A" or "B". The DF-CCSD self-contractions R1, R2, hh and Lld1 need it (equal A and B
    pack keys at every level 6–14). Trigger: B2, the ehrenfest integration declaring DF factor
    sharing. Candidate extension: a member role "AB" (both roles read the group's pack, its
    cost counted once). *Renamed "A=B" at the close-out review (2026-10-07): `"AB"` already
    names the pack mode with two separate packs (DESIGN §6g.10). The trigger is the
    ehrenfest-side integration of the B2 performance model declaring DF-factor sharing (not
    the ROADMAP B2 row, which is done); it is in the ROADMAP C1 row.*
- **M-c (§4.3; IR4, Q24) ran, 2026-10-07, and failed: one §6c.5 target was missed** (binding
  verdict, §5.9 MC-2). The Q24 recalibration was tried and **not adopted** (MC-1, upheld), so
  the shipped calibration is unchanged and the [M, model-only, outside hold-out] marks stay;
  there is no partial pass and no per-class lifting of the marks. C1's done condition on M-c
  is not met. The reviewer authorized the MC-3 feature as the only path to a pass (§5.9 MC-3),
  amended at the second review as MC-3a: it starts from a per-phase account of the two KL
  effects, not from a cache-only key. M-d2 waits until that path closes (§5.9, "M-d2
  ordering").
  - *Sources:* summary immerdink-eval `findings/c1-m-c.md`; scripts `scripts/c1/m_c_*`;
    records `results/c1/m_c/` (untracked, sha256 in the summary).
  - *What ran:* 160 timed calls: 8 shapes × N ∈ {4, 5, 6, 7, 9, 10, 12, 13} plus the
    anchors 8 and 14 × normal and decay_opp data (`bench/sweep_mc.txt`, 530ff81).
    - The shapes and specs are those of the cited [M] figures (squares `ik,kj->ij`; 2000³ in
      place of 2048³, which `bench/sweep_disjoint.py` finds in the calibration set as
      `c_sq2048`).
    - `imd_bench --data decay_opp` is new: `c2_fixtures.py`'s decay_opp generated on the
      device.
    - The run used DESIGN §6c.5's method unchanged: `bench/run_sweep.sh` (quiet gate, GPU
      lock per case, 3 s warm-up), stock clocks with no lock, and `perf_validate.evaluate`
      (BURST, REPEAT, shipped calibration).
    - 4 chunks, each after a FREE busy check; 2.9 min of GPU time. The profiling build's
      `libimmerdink_host.so` is `466acdb4…`, and no kernel source has changed since the
      calibration's ec44099.
  - *Result* (median / p90 / max |error| of the total):
    - < 100 µs: **15.8 / 28.4 / 29.2 %, median missed** (target 15 / 30 %); a repeat of the
      small calls gives 16.1 %.
    - 100 µs – 1 ms: 2.5 / 14.7 / 28.3 %; ≥ 1 ms: 3.0 / 7.3 / 9.8 %. Both met over all
      calls.
    - Per N (reported, not judged): the ≥ 1 ms median is 1.8–5.4 % (5.4 % at N = 9, where
      skinny is +6.5 / +6.7 % through a K4 configuration step, 64×64 → 64×128; 5.0 % at
      N = 5) and the p90 at most 9.6 %. The mid class exceeds its targets only at N = 10,
      and only through the two Jacobi calls just above 100 µs.
    - Per data, the < 100 µs medians (normal 13.5 %, decay_opp 18.2 %) are each one sq1024
      N = 4 call, and the repeat reverses them (normal 18.1 %, decay_opp 14.1 %): noise, not
      a data effect.
    - TFLOPS-eq measured / predicted: for example 4096³ at N = 6 is 62.3 / 62.5 (2.205 ms), DF
      at N = 6 is 55.8 / 58.0 (1.223 ms), and the PPL view at N = 6 is 7.26 / 7.10 (2.340 ms).
  - *Cause: two effects, each necessary, neither sufficient alone.* The < 100 µs class holds
    26 calls: 14 sq1024 calls at N ≤ 10 (|error| 1.1–18.2 %) and 12 Jacobi calls at N ≤ 9
    (26–29 %). Its median, the mean of the 13th and 14th, is the mean of the two sq1024 N = 4
    calls: (13.5 + 18.2) / 2 = 15.8 % in the first run and (18.1 + 14.1) / 2 = 16.1 % in the
    repeat (`report_shipped.md`, `report_shipped_repeat.md`, "Every call").
    - *Small N on the KF path:* sq1024 at N = 4 is predicted 16.1 / 20.6 points more
      optimistically (normal / decay_opp) than the mean of its own anchors' errors (N = 8 and
      14); 14.8 / 15.6 points in the repeat. That baseline inflates the gap, because sq1024's
      error rises with N (−13.5 % at N = 4 to +10.6 % at N = 14). Against the N = 8 anchor
      alone the gap is 8.2 / 15.4 points (repeat 12.8 / 6.8). With these two calls at either
      baseline, the median would be 7.1 % (repeat 9.7 %).
    - *The Jacobi view (KL path, narrow n):* it is predicted 26–29 % too fast at every
      N ≤ 10, the anchor N = 8 included, on both data sets. At N = 12–14, the anchor N = 14
      included, it is −14.2 to −15.4 %: there the model's K4 time (configuration 64×128)
      steps up more than the measured one (N = 12: predicted 32.9 µs, measured 24.6 µs),
      which offsets part of the error. The phase marks show two KL effects, not one cache
      effect (second review, §5.9 MC-3a; immerdink-eval `results/c1/m_c/phases.md`, both runs):
      - *KL A, the wide operand's pack (a 20 MB read):* the REPEAT rule (§6c.1 "Cache
        state") credits it 26.8 µs. The measurement keeps 27–50 % of that credit at N ≤ 10
        (N = 8: measured 45.1 µs, REPEAT 29.6, COLD 56.3) and 55–65 % at N = 12–14. The
        saving is partial, not absent.
      - *KL B, the narrow operand's pack (1600 × 100, 1.28 MB; credit ≤ 1.7 µs):* it takes
        ×1.65–1.89 its REPEAT prediction and ×1.49–1.72 its COLD prediction at N ≤ 10
        (×1.94–2.02 and ×1.77–1.85 at N = 12–14). That is a KL path error on narrow packs,
        in every cache state. It is 47–59 % of the call's miss at N ≤ 10 and 92–105 % at
        N = 12–14.
      - The call totals match the COLD prediction (−1.5 to +10 %) only because the two
        cancel: COLD over-predicts KL A and under-predicts KL B. So the call-level
        counterfactual, the Jacobi calls at their COLD total (median 3.4 %, repeat 5.0 %), is
        a cancellation. Per phase (first run / repeat): KL A at its measured time gives 5.9 /
        7.6 %, KL B at its measured time 11.8 / 12.0 %, both 4.7 / 5.1 %.
    - The Q24 cases of the class (`q_kl1`, `q_kl2`, `q_kl4`) show both effects: KL B takes
      ×1.50–1.81 its REPEAT and ×1.31–1.65 its COLD prediction (27–95 % of the miss), and KL A
      keeps 5–52 % of its credit. The in-sample `c_skin1280` and `c_k3hit` share only the KL A
      part (KL A keeps 14–33 %; KL B ×1.00–1.09).
    - KF calls keep most of their saving, but not in the modeled shape: sq1024's KF phase
      keeps about 74 % of its modeled credit at N = 4 (measured 27.2 µs, REPEAT 21.6, COLD
      43.6) and about 157 % at N = 14 (35.7 / 44.5 / 59.8 µs); `q_kf1` keeps 80 % at N = 4 and
      163 % at N = 12. So the small-N KF optimism may be the cache rule's shape and not an
      N effect. MC-3a examines it on `q_kf1` and the cache sweep's `x_s1024`, not on
      `mc.jsonl`.
    - These figures are computed from the records at full precision (immerdink-eval
      `scripts/c1/m_c_phases.py`). The first review round's 16.2 / 20.6 (repeat 14.9 / 15.7),
      3.5 % and "+1.1" were differences of the report's one-decimal errors.
  - *N itself:*
    - N = 9–13 track the anchors (median difference from the mean of the anchors' errors
      −0.3 to +1.2 points).
    - N = 4–7 are optimistic: 2.0–4.3 points faster than the mean of the anchors' errors
      (median over the 16 shape × data series), at most 20.6 points (sq1024 KF, decay_opp,
      N = 4).
    - decay_opp is up to 5.8 % faster than normal data at N = 4 on the large shapes ([KT26]),
      within the targets.
    - Indicative §2.4 ratios from these medians (not M-d2; never used to set or adjust M-d2's
      bars, §5.9): the schedule against fixed N = 14 measured 0.684–0.767 (normal data) and
      0.676–0.765 (decay_opp) on the large shapes, against [M] 0.664–0.793; on the Jacobi
      view 0.830 / 0.833 against 0.786.
  - *Q24:* 43 N-specific calibration calls on 7 shapes disjoint from M-c and the hold-out
    (`bench/sweep_calib_n.txt`, 78bc523; 0.5 min of GPU). The message of commit 78bc523 says
    42; the file holds 43 (8 + 8 + 5 + 4 + 8 + 5 + 5), the count used here and in the
    findings.
    - A refit on the shipped fit's records plus these (`n1`; the baseline refit reproduces the
      shipped parameters) meets M-c narrowly (< 100 µs median 14.7 %) and keeps the product
      hold-out. That 14.7 % is not hold-out evidence: the Q24 cases were chosen from M-c's
      residuals, and `q_kl2` (1296 × 144 × 1296, `abcd,cdij->abij`) is in the Jacobi view's
      own class.
    - It is not adopted (§5.9 MC-1, upheld twice). It misses on the repeat run (15.0 %). It
      moves path parameters (`kl.wave` 2.16 → 4.45 µs, ×2.06, which partly fits KL B's real
      path error; K4's effective L2 0.4 → 0.3 × nominal; and more, MC-1), which moves the PPL
      view from +2 … +10 % to +11 … +20 % and M-c's ≥ 1 ms p90 from 7.3 to 14.0 %.
    - The fix is a model feature beyond Q24, now authorized as amended (§5.9 MC-3 and
      MC-3a; NOTES-B2 §12 "Remaining"): at most one optional key per physically distinct
      effect, two in all (the REPEAT credit; the KL narrow-pack time).
  - *Review (approve-with-fixes, 2026-10-07; verdicts §5.9).* The first write-up (a382ef6,
    whose message keeps that framing) put the miss on the Jacobi view alone ("a shape, not
    N"). The records show the missed median is set by the two sq1024 N = 4 calls, so the cause
    is now stated as both effects here, in §4.3, DESIGN §6g.9, NOTES-B2 §12, the ROADMAP and
    HANDOFF, and in the findings. Also applied: the Jacobi view's N = 12–14 errors, the n1
    change list (MC-1), the planes the candidate key would count and why no setting of the
    two cache keys can work (MC-3), the decay_opp ratio range, and a CPU self-test of
    `bench/sweep_disjoint.py` (ctest `sweep_disjoint_selftest`: its matching rule, its exit
    codes, and the committed M-c and Q24 sweeps against the fit sweeps and the hold-out).
  - *Second review (approve-with-fixes, 2026-10-07; verdicts §5.9, "Second review").* The
    first review's account of the Jacobi view ("the REPEAT credit", "matches the COLD
    prediction", "loses its whole saving", "one optional REPEAT-credit key") held for call
    totals only. The phase marks show two KL effects whose errors cancel at the total ("Cause"
    above), so MC-3 is amended as MC-3a (two optional keys at most, fitted against the phase
    marks, three more pass conditions and a fresh narrow-KL hold-out). The v1.18 hold-out is
    judged on two sets in MC-3 step 4 (all cases, and the cases whose GEMM view matches no
    fit case). Also applied: the exact n1 change list, which overlaps share N with a fit
    sample (4 of 7), the anchors' baseline next to "16–21 points", and the run-to-run
    figure of the Jacobi error.
- **M-d2 (§4.3) has not run: the binding ordering holds** (§5.9 "M-d2 ordering": M-d2 waits
  until the MC-3 path closes, and MC-3 has not started). A workflow task of 2026-10-07 asked
  for M-d2; no M-d2 system was run on the GPU. The independent reviewer upheld this (§5.10
  MD2-0, binding): MC-3a comes first. MC-3 may close without a calibration change only as
  MC-3a's recorded outcome or by Eric's explicit decision. Instead, before any M-d2
  measurement:
  - *Protocol and driver pre-registered,* first at 3254b15, re-registered at daf3583 after
    the review (approve-with-fixes, §5.10), and **re-registered at 35eeb46** after the
    re-review (approve-with-fixes, §5.10 "Re-review"), which is normative:
    `bench/accuracy/md2_time.py`, sha256 `90fa7f92a143…` (in full in §4.3), whose
    docstring is the protocol (§4.3 "M-d2", "Pre-registered protocol"). It has four modes:
    `predict` (CPU), `run` (GPU, one fresh process per system and data set), `judge` (CPU)
    and `selftest` (CPU, the decision and provenance rules, 95 checks; the CPU-only ctest
    `md2_time_selftest`).
    - `judge` decides per system *and data set* by the pre-registered rules (a)–(d) (§4.3):
      records that are contaminated, or whose bit-exact check is not true at every level,
      are inadmissible; a bit-exact failure on any record in the file stops M-d2; one
      re-run at most; the first admissible record decides. 3254b15's judge keyed its
      verdict by system only, so one data set recorded twice passed.
    - One normative records file (immerdink-eval `results/c1/m_d2/md2.jsonl`); every M-d2
      record in any file is an attempt and is reported. The driver is frozen from the
      first M-d2 record on. After rule (d), STOPPED or a refused file, nothing continues
      without a recorded reviewer verdict or Eric's decision (§4.3 "Records, freeze and
      continuation").
    - `judge` refuses a records file holding an M-d2 record that fails a provenance check:
      - the registered calibration file;
      - the loaded library's shipped (build-time embedded) calibration equal to that file,
        as canonical text and as bit-identical predictions;
      - the registered bar and p90 class reproduced on the loaded host library
        (`registered_check`);
      - a clean tree, and a `--md2-cleared` that cites the commit (an ancestor of the
        record's) and the NOTES-C1 section recording MC-3's closure;
      - the judging driver's own sha256;
      - the fixed protocol parameters;
      - the bars' configuration at every level (g = N, GENERATE, in-line, RN);
      - one library per file (`lib_sha256`, `host_lib_sha256`, host API).
    - `run` refuses the four M-d2 systems unless `--md2-cleared` cites the closure, and
      with any `--lib` but `build/libimmerdink.so`. Before any GPU work, it also refuses a
      record that would fail those checks, or that rules (a)–(c) forbid.
    - The protocol decisions MD2-0 to MD2-8 and the reviewer's verdicts are in §5.10: all
      upheld, MD2-3, MD2-6 and MD2-7 with the first review's fixes, and MD2-8 amended by
      the re-review (the driver frozen from the first M-d2 record on).
  - *The bars reproduce* from the shipped calibration (`d31f62ca…`). This is `predict` at
    35eeb46 on `libimmerdink_host.so` `466acdb4…`; CPU, one thread, under memcap, FREE.
    Its tables and systems are identical to daf3583's and 3254b15's. The library's
    shipped calibration equals the registered file, as canonical text and as predictions
    on the six systems.
    - S14/F14 = 0.7501 (PPL view), 0.6635 (DF) and 0.6728 (4096³), i.e. the registered
      0.750 / 0.664 / 0.673. Rank's choice and g = N agree, and `PrecisionPlan` picks
      g = N at every level. The Jacobi view gives 0.7865.
    - Bands: [0.6375, 0.8000], [0.5644, 0.7636] and [0.5721, 0.7740]; the Jacobi view's
      [0.550, 1.022] is reported only.
  - *Tooling check only, not M-d2 evidence:* the driver ran on two shapes that are not M-d2
    systems: 512³ with A invariant, and a PPL-like 100 × 576 × 576 with B invariant. Both
    data sets, 5 reps, under the GPU lock. It ran at 3254b15, at daf3583 and again at
    35eeb46 (41 s of wall time, each lock hold about 10 s).
    - At 35eeb46, packed and in-line agree bit for bit at every level. Every run was
      queued ahead (0 of 10 gated S14/F14 runs not queued ahead, per record). Worst
      spreads 0.39–2.28 %, and every record is admissible.
    - Measured S14/F14 against [M]: 0.859 and 0.859 against 0.859 on 512³; 0.724 and
      0.713 against 0.681 on the PPL-like shape (+6.3 % and +4.7 %). These ratios check
      the tooling. They are not model evidence.
    - The rules end to end, on relabelled copies of these records (CPU):
      - one data set recorded twice gives INCOMPLETE (3254b15's judge said PASS);
      - a contaminated record is inadmissible;
      - two inadmissible attempts give INCOMPLETE, and no further attempt without a
        recorded verdict;
      - a bit-exact failure stops M-d2, on a smoke record in the file too;
      - a null bit-exact value at one level is inadmissible, not a stop;
      - a dirty tree, another driver, other protocol parameters, two libraries, no
        `registered_check` or `calib_check`, and an `--md2-cleared` without a citation
        or with a commit that is no ancestor are refused.

      With no GPU visible, `run` refused each case the rules forbid and wrote nothing. A
      positive control that meets every check got past the gate and stopped at the first
      CUDA call ("no CUDA-capable device"), writing nothing.
    - Records: immerdink-eval `results/c1/m_d2/` (untracked; `rereg2/` for 35eeb46,
      `reregister/` for daf3583); summary `findings/c1-m-d2.md`. The relabelled synthetic
      records of the judge checks (real smoke records relabelled as M-d2 systems) were moved
      out of `m_d2/` at the final review, to `results/c1/m_d2_judge_checks/{rereg2,
      reregister}/` (files and sha256 unchanged), so that no glob over `m_d2/` counts them
      as M-d2 attempts (MD2-8).
  - *Caller note [M, model-only, outside hold-out]* (§5.10 MD2-4, §2.4). The PPL view's bar
    is an in-line ratio. With the invariant B (680 MB) packed per phase, the schedule gain
    is 1.14× (P14/PF14 = 0.876, above the 0.8 cap), against 1.33× in-line: the four packs
    cost 1.72–2.41 ms each, against packed calls of 0.67–1.59 ms.
    - Packing B is the larger lever. At the same schedule, P14/S14 = 0.509 (about 1.97×);
      at fixed N = 14, PF14/F14 = 0.436. P14/F14 = 0.382 combines both effects.
    - DF and 4096³ barely change packed (P14/PF14 = 0.667, 0.678).
    - The bars are unchanged, and the packed runs stay reported only. M-d2's measured
      P14/PF14 replaces these figures once it runs.

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

**Binding decisions of the independent design reviewer** (decided by the independent reviewer,
2026-10-07). A completeness critic then found 17 gaps, and the previous version left 8 items
open. The reviewer decided all of them, plus the stage-5 gate, pre-registered gate rules, the
order of work and a definition of done. They override the earlier text wherever the two differ.
§5.3 records each as IR1–IR29 with its rationale, and the stage texts of §4 carry them:
- **IR1** (gap 1): M-a runs on full operands up to m = n = 4096. It computes the guard
  quantities exactly, at O(mk), with a σ engine bit-identical to the oracle, and samples 64 × 64
  outputs. Tier S tunes d*; Tier L and the real operands are the gate evidence. The budget is a
  hard 90 CPU-min, with a pre-declared cut order; m = 16 is dropped (§4.3).
- **IR2** (gap 2): `PrecisionPlan` composes the per-role pack cost in Python. It makes one rank
  call with reuse = 1 and re-keys the candidates by contract_s + Σ pack_s(role, N)/reuse(role).
  Reuse is per role ('call' / 'iteration' / 'phase' / int). No ABI change (§4.2).
- **IR3** (gap 3): §1.3 is rewritten, since relaxed Krylov methods loosen precision
  (`monotone=False`). M-d1 adds report-only block Davidson and block CG cases (§1.3, §4.3).
- **IR4** (gap 4): §1.2 item 8 is corrected; no case anywhere is at N = 5, 6, 7, 9 or 13. M-c
  covers N ∈ {4, 5, 6, 7, 9, 10, 12, 13} plus anchors at 8 and 14 on 8 shapes. Until it passes,
  figures at N ∉ {8, 10, 14, 16} are [M, model-only, outside hold-out] (§1.2, §4.3).
- **IR5** (gap 5): stage 3 first gets L3c and an FP summation lemma γ_k in Lean, before L7; it is
  re-sized S → M. The code does not wait for the Lean; "C2 done" does (§4.4).
- **IR6** (gap 6): K-weights for an operand X with several partners. The clamp uses union
  extrema, and the guard is pairwise, G = min_p G_p. In `oracle.kweights` from stage 2 (§4.5).
- **IR7** (gap 7): the ρ recipe ships as the tested helper `rho_from_output`, for β = 0 and
  α = 2^e only. M-d1 measures its next-iteration miss rate (§4.2, §4.3).
- **IR8** (gap 8): pack-mode masks per contraction. `budget` keeps rank's meaning, and a new
  phase status BUDGET is added. "Packing never increases ws" becomes certificate I3, not ABI
  (§4.1, §4.2).
- **IR9** (gap 9): the design-phase scripts and records move to immerdink-eval, with a sha256
  manifest (stage 0). M-b re-runs every cited [M] figure on this worktree's library (Provenance,
  §4.1, §4.3).
- **IR10** (gap 10): the stage-0 GPU round measures the first plan and contract per N, and tests
  cold capture. A plan-time preload is added only if needed (§4.1).
- **IR11** (gap 11): output-selective refinement is recorded as §3.1 B6, a caller recipe with no
  API. M-a estimates its value (§3.1, §4.3).
- **IR12** (gap 12): §2.7's cost wording is corrected. Stage 3 follows the M-a summary, and its
  §7 rows cite M-a, or are worded "provisional until M-a" (§2.7, §4.4).
- **IR13** (gap 13): a capture-safe guard path through `IMD_KW_ZERO_ON_FAIL`. The guard read and
  the choice of N happen outside capture (§4.5).
- **IR14** (gap 14): rebuilding this worktree's libraries, with an API-13 check, is the first
  step of stage 0. Nothing else in stages 0–2 runs before it (§4.1).
- **IR15** (gap 15): `levels` may end with a DIRECT sentinel. The prefix-plane option is recorded
  as the comparator a slice revival must beat. M-b measures the share-group raise cost. One §7
  line covers larger moduli (§3.3, §4.2, §4.3, §4.4).
- **IR16** (gap 16): batched fixtures (real R7 with core orbitals, and a synthetic one), with a
  pre-registered Q12 trigger (§4.3, Q12).
- **IR17** (gap 17): F3's second half ("no byte beyond `pack_bytes`") becomes ABI only after the
  GPU test `pack_write_extent` passes (§4.1).
- **IR18** (open item 1): the ROADMAP edits of Q6 and Q7 are approved on Eric's behalf (below).
- **IR19** (open item 2): no ehrenfest dump in this cycle. Real operands come from our own psi4
  script in immerdink-eval, and the contractions R1–R7 are taken from the consumer's ccsd.py
  (§4.3, Q14).
- **IR20** (open item 3): the optional StreamSpec model is not adopted; R1 stands (§4.1).
- **IR21** (open item 4): stage 6 delivers the weights to the kernels through a Q9-index map q(k̄)
  in the vec region of plans created weighted. No new region (§4.7).
- **IR22** (open item 5): superseded by IR14.
- **IR23** (open item 6): every inferred item gets a named closing measurement, ordered before
  the claim (§4.1, §4.3, §4.5).
- **IR24** (open item 7): share groups are ranked jointly over every level ≥ the group floor
  (§4.2).
- **IR25** (open item 8): binding run limits for the stage-4/5 TLA+ models (§4.5).
- **IR26**: stage 5's gate is not met. Stage 5 is formally deferred with triggers T1–T3, and its
  cost bar is restated without `ncu` (§4.6). C1's definition of done does not need it.
- **IR27**: pre-registered gate rules for stages 4 and 6 (§4.5, §4.7).
- **IR28**: the order of work, as round 1 (CPU) and round 2 (GPU) (§4.11).
- **IR29**: a definition of done for C1 and C2 (§4.11).

**To report to Eric** (none of these blocks stages 0–3):
- **Q6 and Q7: approved on Eric's behalf by the independent reviewer (IR18).** Eric may overrule.
  - ROADMAP C1's decision column becomes "C1 = (a) per-call N (plan + pack per N) + host-side
    (b) (PrecisionPlan, min_moduli/rank, rigorous ρ-from-output helper); (c) Ozaki-I slices
    parked with triggers (§3.3); TLA+ for slices deferred to the slice trigger".
  - ROADMAP C2 is retitled "K-equilibration (power-of-two K-weights, guarded, staged and
    gated) + K-segmentation recipe; accurate mode re-rejected (§2.8)".
  - Both land with the DESIGN v1.19 merge.
- **Q14: no ehrenfest dump is requested in this cycle (IR19).** The real operands come from our
  own psi4 script. If psi4 cannot run, the Eric-approved dump is the named fallback.
- **Stage 5 is formally deferred (IR26).** It has triggers T1–T3, and T3 is Eric asking for the
  device certificate.
- **At the close-out (2026-10-07): C2 is done, C1 is open** (Status, "Status at the C1/C2
  close-out"). C1 continues with MC-3a (a performance-model feature and refit, GPU) and then
  M-d2. MC-3 can close without a calibration change only as MC-3a's recorded outcome or by
  Eric's explicit decision (§5.10 MD2-0).
- **IR26's premise was wrong (close-out review, 2026-10-07).** GPU performance counters are
  open to all users on this machine, as HANDOFF and the ROADMAP say, not admin-only. Stage 5's
  cost bar keeps its content (A/B timing at stock clocks, the compiler's register count, the
  occupancy query) because it is a time bar and these measure time directly. `ncu` is not part
  of it but not forbidden: as a diagnostic, only with `--clock-control none` (its default,
  `boost`, locks the clocks), under the GPU lock (§4.6, "Correction"). Nothing on the machine
  was changed.

**Decision summary.** §3 gives the reasons, §4 the stages.

| item | decision |
|---|---|
| C1(a) per-call N | **Plan per N + pack per N** on host API 13, with published footprint monotonicity (totals, plus the write extent once `pack_write_extent` passes, IR17), internal layout certificates, and a re-pack-in-place protocol (stage 0, which starts with the library rebuild, IR14). A single plan serving a range of N is **parked**: it gains nothing per call. |
| C1(b) auto-N | **Host-side**, using the existing `imd_err_min_moduli` / rank, plus a thin `PrecisionPlan` in its own Python module and a tested rigorous ρ-from-output helper (stage 1; IR2, IR7, IR8, IR24). Device statistics and an a-posteriori certificate (lemma L6) are **formally deferred**: the gate, a consumer with elementwise targets, is not met, and triggers T1–T3 are recorded (stage 5, IR26). Choosing N on the device is rejected. |
| C1(c) Ozaki-I slices | **Parked with triggers**; the numerics ideas and the prefix-plane comparator are recorded (§3.3). |
| C2 | **K-equilibration**: power-of-two weights on the contracted index (lemma L5), with a multi-partner rule (IR6). First as caller-side pre-scaling with an `imd_kweights` helper (stage 4). The helper has a mandatory FP64-exact clamp and an exact integer guard that is never worse in the §3.8 bound (L5c). It reports the gain G, which converts the plain operands' statistics into a rigorous request for the weighted call (L5d). Library-internal weights and weighted packs come later (stage 6). Both stages are gated, by pre-registered rules (IR27), on a CPU study at realistic sizes and on real psi4 operands (stage 2; IR1, IR19). **K-segmentation** becomes a documented caller recipe with lemma L7 (stage 3). **Accurate mode stays rejected**, for the corrected reason in §2.8; the never-worse partner-envelope rule is recorded in Appendix B. |

**Tags.**
- [V]: verified in code (file:line) or in the docs.
- [C]: computed on the CPU in this design phase. This is either exact integer emulation, bit-exact
  against `spec/oracle.py`, or the host model library's own output.
- [M]: performance-model prediction, not a measurement. Shipped v1.18 calibration, host API 13,
  DEFAULT (= SUSTAINED), COLD, GENERATE, g = N unless stated.
- [E]: an estimate built from [M] numbers and stated assumptions.
- [I]: inference, not checked.

**Provenance.**
- Design-phase scratch directory: `/home/ef/.claude/jobs/ca28e7d2/tmp/c1/` (ephemeral). **Copied
  to immerdink-eval in stage 0** (IR9): the scripts to `scripts/c1/design-phase/`, the records to
  `results/c1/design-phase/`, each with its sha256 in `findings/c1-design-phase-manifest.md`. The
  file names below are relative to those directories.
  - Scripts: emulator `emu.py`; scaling variants `exp_acc.py`, `exp_hyb.py`, `exp_lowN.py`;
    K-weight study `c2gate.py`, `c2gate2.py`; K-segments `c2cmp.py`; envelope rule `c2bcs/`;
    slices `slice_eval/`; the review's size check `judge/keq_m.py`.
  - Records: `acc.jsonl`, `lowN.jsonl`, `hyb.jsonl`, `run_all.out`, `run2.out`, `run3.out`,
    `perf_summary.md`, `c2cost.md`, `predict_cc.out`.
  - Second-round review scripts: R-api's host-model spot checks `review_api/spot*.py`; R-num's
    exact L5c/L6 checks `review/chk_lemmas.py` (and `review/chk_perf.py`).
- **Copy to immerdink-eval** (IR9; stage 0 step (b); **done 2026-10-07**, every copy compared
  with `cmp`). Scratch is ephemeral, and the provenance rule is paper-grade:
  - the scripts (emu.py, exp_*.py, c2gate*.py, c2cmp*.py, c2bcs/, slice_eval/, judge/,
    review*/, perf*.py, nsweep.py, phases.py, predict_cc*.py) go to
    `scripts/c1/design-phase/` (tracked);
  - the records (*.jsonl, *.out, *.md summaries) go to `results/c1/design-phase/` (untracked,
    never `git add`);
  - a manifest goes to `findings/c1-design-phase-manifest.md` (tracked). It lists each file's
    sha256, the source worktree commit (cbdf230), the sha256 of the agent-ac829dae library, and
    host API 13.

  DESIGN and these notes then cite the eval paths, not scratch. The copy happens before any
  DESIGN v1.19 text cites a [C] or [M] figure.
- Every run checked first that ehrenfest was FREE, then used one thread under
  `tools/memcap.sh` and took at most a minute. R-num's [M] re-runs were not completed, because
  ehrenfest became BUSY; R-num checked their arithmetic for internal consistency instead. M-b
  re-runs every [M] figure DESIGN will cite on this worktree's rebuilt library. A figure that
  differs by > 0.5 % is replaced, and the change is noted (IR9). **Done (M-b, 2026-10-07;
  Status):** every stored design-phase [M] record reproduces byte for byte on this worktree's
  library, so R-num's re-runs are complete; the one cited [M] figure that differed was a text
  error (§2.3's re-pack share), corrected there.
- Stage 2 (§4.3) moves the study into `bench/accuracy/` and the raw records into immerdink-eval
  `results/c1/`.
- [M] numbers come from `.claude/worktrees/agent-ac829dae3600d2888/build/libimmerdink_host.so`
  (cbdf230, host API 13). Its `src/host`, `include` and `python` match this worktree's
  (`diff -rq`). This worktree's own `build/libimmerdink_host.so` was stale in the design phase:
  host API 5, with no `imd_perf_*` or `imd_err_*` symbols (R-num confirmed this with `nm`).
  **Rebuilding it was the first step of stage 0** (IR14), and the rebuilt library is
  byte-identical to the agent-ac829dae one (sha256 under "Implementation status"). Stages 0 and
  1 already need the API-13 symbols, and every later record names this worktree's library.
- Tag suffixes: "[C, review]", "[M, review]" and "[E, review]" mark numbers a second-round
  reviewer computed or estimated, under the same rules. A tag without a suffix is this draft's
  own. "[V, independent reviewer]" marks a fact the independent design reviewer verified for its
  decisions (§5.3).
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
   - **N coverage** (the N column of each bench file; rows per N in parentheses where few)
     [V, independent reviewer, re-counted]:
     - calibration (`bench/sweep_calib.txt`): N = 4 (2), 8, 11 (2), 14–17 and 20;
     - hold-out (`bench/sweep_perf_validate.txt`): N = 8, 10 (2), 14 and 15–20;
     - cache sweep (`bench/sweep_cache.txt`): N = 8, 10, 12 (1), 14 and 16.

     **No case anywhere is at N = 5, 6, 7, 9 or 13.** Several things therefore rest on
     unvalidated predictions until M-c (§4.3, IR4) passes: the default levels (6, 8, 10, 12, 14),
     the §2.4 schedules (N = 6 and 9), and the 1024³ fit's K4 wave steps at N = 9 and 13 (§2.3).
9. **RN acceptance.** κ ≥ T(N) gives NOT_SUPPORTED at plan time (analysis.cpp:322) and at pack
   time (pack.cpp:213). N = 2 needs k ≤ 32041; N = 3 needs k ≤ 8.2e6 (params `rn_feasible_kmax`).
10. **K5's per-output code** holds |z| as L32(N) limbs plus its sign after `crt_finish`
    (contract.cu:515), just before `round_scaled` (contract.cu:516). That is where any
    per-output statistic goes.

### 1.3 Consumer profile

**Generic iterative methods** (rewritten by IR3).
- **Fixed-point (Jacobi/DIIS), Newton-type and residual-driven iterations**, iterative
  refinement included, tighten precision as the residual falls. That means a few monotone phases,
  switched at iteration boundaries where the caller already crosses to the host. This is
  `monotone=True`, the `PrecisionPlan` default (§4.2).
- **Relaxed (inexact) Krylov methods may loosen the matvec precision** as the residual shrinks
  [literature, not checked here: Bouras–Frayssé; Simoncini–Szyld; van den Eshof–Sleijpen]. They
  use `PrecisionPlan` with `monotone=False` and a decreasing schedule, and the docs state that
  use. `monotone=True` stays the default.
- M-d1 (§4.3) reports a block Davidson case (tightening ramp) and a block CG case (tightening and
  relaxed). Both are report-only and document caller guidance; they are not C1 gates.
- **Caller guidance from M-d1** (2026-10-07; CPU evidence on one fixture each, report-only, not
  a C1 gate; accepted with corrections by the independent reviewer, §5.6 S2-9; immerdink-eval
  `findings/c1-evidence.md`; carried into DESIGN §6g.12 at the close-out, which is the
  caller-facing text):
  - *Fixed-point iterations and DIIS.* The forcing-rule ramps match fixed N = 14: the same
    iterations, and the final residual within 2×. With DIIS, reset the history at each
    precision switch. That costs no iteration and improves the final residual, 1.5e−16
    against 2.1e−15. The gain comes from restarting a DIIS that stalls: plain DIIS stalls near
    1–2e−15 with or without switches (the ramps without the reset end at 1.18e−15 to
    2.09e−15; fixed N = 14 and the exact-arithmetic reference also end at 2.09e−15), so the
    stall is not the switch's (close-out review, 2026-10-07).
  - *Subspace methods (block Davidson).* At each N switch, collapse the subspace to the Ritz
    block and recompute its σ = HV. Otherwise the old low-N products stay in the subspace and
    the ramp stalls. Even with the recompute, the ramp took 16 iterations against 11 at fixed
    N = 14 (+5), and that on an easy, diagonally dominant case (coupling ‖2BBᵀ‖ = 1 Ha
    against a diagonal of 0.35–61 Ha). So §4.3's expectation (at most one more iteration) is
    not met, and a ramp is not recommended for Davidson unless the per-iteration saving
    outweighs that overhead.
  - *Krylov methods (CG): start tight and relax.* Rules that start at low N and tighten (the
    fixed-point forcing rule, or a relative η‖r_k‖ rule) stall CG at N = 5–6. A relaxed rule,
    matvec tolerance min(1e−2, tol/‖r_k‖), converges (1e−7 in 274 iterations against 251 at
    fixed 14, with N falling from 12 to 6), but it leaves a residual gap: it attains 7.1e−8,
    against 3.3e−10 at fixed 14 and 2.0e−10 with the FP64 matvec (cond 1e6). The relaxation
    must therefore be scaled by the expected iteration count, e.g. tol/(m_max·‖r_k‖), as
    inexact-Krylov theory has it; the M-d1 rule had no such factor.
- ACCURACY §3 shows a contracting iteration does not accumulate emulation error: drift is set in
  the first few iterations, and the steady state is about one contraction's error/(1 − ρ). So for
  a fixed-point iteration, the last phase's N sets the converged accuracy.

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
  random structure. M-a (§4.3) re-runs the C2 study on full operands up to m = n = 4096, with
  sampled outputs (IR1).
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
- **Re-run by M-b** (2026-10-07, this worktree's library, sha256 `466acdb4…`, host API 13;
  immerdink-eval `findings/c1-m-b.md`): the scripts' stored outputs reproduce byte for byte, and
  every figure of §2.3 and §2.4 recomputed from fresh queries prints as cited except the
  re-pack share below (a text error, corrected) and five table cells that were mis-rounded:
  the unchanged predictions 24.348 (1024³, N = 6), 82.848 (4096³, N = 4), 47.449 (8192³,
  N = 10) and 30.849 TFLOPS-eq (DF, N = 12) and 18.8945 ms (QZ, N = 16) lie just below a tie
  and were printed 24.4, 82.9, 47.5, 30.9 and 18.90. They are corrected in place below (M-b
  review; ≤ 0.21 %, so IR9's 0.5 % rule did not require it).

TFLOPS-eq (time):

| shape | N = 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| 1024³ | 30.2 (71 µs) | 24.3 (88) | 21.8 (99) | 18.5 (116) | 17.1 (125) | 14.3 (150) | 13.7 (157) | 11.3 (190) |
| 4096³ | 82.8 (1.66 ms) | 61.2 (2.25) | 48.1 (2.86) | 39.8 (3.46) | 33.9 (4.05) | 29.2 (4.70) | 26.0 (5.30) | 20.9 (6.58) |
| 8192³ | 106.1 (10.4 ms) | 75.4 (14.6) | 58.1 (18.9) | 47.4 (23.2) | 40.1 (27.4) | 34.5 (31.9) | 30.4 (36.1) | 24.4 (45.0) |
| DF acQ,bdQ→acbd 64⁴, Q = 2032 | – | 56.7 (1.20 ms) | 44.0 (1.55) | 36.3 (1.88) | 30.8 (2.21) | 26.3 (2.59) | 23.4 (2.92) | 18.6 (3.66) |
| QZ kiQ,acQ→kiac (o 21, v 489, Q 1122) | 32.3 (7.32 ms) | 25.9 (9.15) | 21.3 (11.09) | 18.3 (12.96) | 16.0 (14.82) | 13.9 (17.02) | 12.5 (18.89) | – |

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
- **Linear fits in µs:** 41 + 7.7·N (1024³, with K4 wave steps at N = 9, 13 and 17; no case
  covers 9 or 13, §1.2 item 8),
  363 + 310·N (4096³), 1441 + 2176·N (8192³), 120 + 177·N (DF).
  - The fixed part (FP64 reads in KF/KL, K0, K5's base cost) caps the gain at low N: N = 4 is
    only 1.35× faster than N = 6 at 4096³.
  - A chooser must therefore query the model, not assume time is linear in N.
- **K5's share of the call at N = 14** is 8.6 % (4096³), 5.1 % (8192³) and 15.6 % (DF) [M]. It
  was measured at 16.8 % on sq4096 (NOTES-D1:106: 0.887 of 5.276 ms).
- **Re-pack cost:** `imd_pack` of a 4096² operand is 313 / 414 µs at N = 8 / 14 [M] (measured
  0.437 ms at N = 14, NOTES-D1:107), against a call of 2.86 / 4.70 ms. That is 9–11 % of one
  call (10.9 % at N = 8, 8.8 % at N = 14; M-b, was "about 11 %", which holds at N = 8 only),
  paid once per switch for each invariant operand.
- **No calibration or hold-out case covers N = 5, 6, 7, 9 or 13.** N = 4, 11 and 12 have one or
  two rows each (§1.2 item 8). [KT26] also reports INT8 throughput that depends on the data at
  small N, through the fraction of zeros after scaling. M-c in stage 2 checks these predictions
  (IR4). Until it passes, DESIGN tags every figure at N ∉ {8, 10, 14, 16} as [M, model-only,
  outside hold-out].

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

These use g = N predictions. `PrecisionPlan`'s own choices (`configs()` over the levels 6, 8,
9, 10, 14, in-line, GENERATE) are g = N at every level of every shape here, so they give the
same ratios to three digits; with both vec modes (RESIDENT everywhere) the ratios move by at
most 0.004 (the Jacobi view, 0.7865 → 0.7826) [M; M-b, immerdink-eval
`results/c1/m_b/schedules_pp.jsonl`].

- Re-packing one invariant 4096² operand at each of the three switches costs about 1.1 ms, about
  2 % of the 50.6 ms schedule [M].
- So C1's value depends on the final precision the consumer needs: about 1.4–1.5× when the last
  phase needs N = 14, about 1.1× when it needs N ≈ 9.
- **It also depends on the shape.** The rows above are large calls. On the Jacobi view
  1600×100×1600 the same schedule is predicted at 0.79× of fixed N = 14 (1.27×), and on an
  o = 10, v = 96 PPL view (100 × 9216 × 9216) at 0.75× (1.33×) [M, review]. Many-small-plans
  workloads gain about 10–25 % (§2.3).
- **A caller who packs the PPL view's invariant B per phase gains less from the schedule**
  [M, model-only, outside hold-out]: 1.14× (P14/PF14 = 0.876) against 1.33× in-line, because
  each per-phase pack of the 680 MB B costs 1.72–2.41 ms.
  - Packing B is the larger lever: at the same schedule P14/S14 = 0.509 (about 1.97×), and
    at fixed N = 14 PF14/F14 = 0.436.
  - DF and 4096³ barely change (P14/PF14 = 0.667, 0.678).
  - M-d2's measured P14/PF14 replaces these figures once it runs (Status "M-d2", §5.10
    MD2-4).
- **All of it comes from running at a lower N**, which per-N plans already allow (§3, A1 vs A2).
- **These rest on predictions at N = 6 and 9**, and the default levels rest on N = 6 and 12. No
  calibration or hold-out case covers 6 or 9, and 12 has a single cache-sweep row (§1.2 item 8).
  M-c (§4.3) must validate them. Until it passes, DESIGN marks the default levels and every §2.4
  or §4.10 schedule gain that uses N ∉ {8, 10, 14, 16} as [M, model-only, outside hold-out]
  (IR4).

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
  count neither the helper nor the pre-scaling (§4.5). Dense data gain nothing. K-eq's N = 9 and
  12 lie outside {8, 10, 14, 16}, so DESIGN tags these rows [M, model-only, outside hold-out]
  as well until M-c passes (IR4, applied conservatively). M-a recomputes them with every cost
  term (§4.3).

  | target | N, fast → K-eq | shape | TFLOPS-eq (time) | speedup, upper bound |
  |---|---|---|---|---|
  | 1e−10 | 19 → 12 (k = 3600) | 4096³ | 21.9 (6.27 ms) → 33.9 (4.05 ms) | 1.55× |
  | 1e−10 | 19 → 12 (k = 3600) | DF | 19.5 (3.49 ms) → 30.8 (2.21 ms) | 1.58× |
  | 1e−10 | 18 → 12 (k = 600) | 4096³ | | 1.47× |
  | 1e−6 | 16 → 9 | 4096³ | 26.0 (5.30 ms) → 43.5 (3.16 ms) | 1.68× |

  Re-run by M-b: unchanged. DF N = 12 is 30.849 TFLOPS-eq, printed 30.9 before the M-b
  review, now 30.8. The recurring costs below reproduce at 0.9 TB/s (the design phase's
  bandwidth assumption); the last one is 1.19504×, printed 1.19 before the M-b review, now
  1.20.
- **Recurring costs** for operands rebuilt every iteration (t2, dressed integrals; §1.3) [E]:
  - Pre-scaling costs one FP64 read and write, 16 B per element, per changed operand per call,
    unless it is fused into the operand's producer kernel. That is about 0.3 ms per 4096²
    operand [E, review].
  - The helper costs one read of A and B without the guard and two with it, about 0.3 / 0.6 ms at
    4096³. It recurs only if it is re-run on each new pair, for example to keep the guard
    current.
  - At 4096³, 1e−10, the 1.55× becomes about 1.44× with one unfused changing operand and about
    1.35× with two [E, review]. Re-running the guarded helper every call as well gives about
    1.27× and 1.20× [E].
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

  *M-b* (immerdink-eval `findings/c1-m-b.md`). The review's method is not recorded. Taken as
  (S − 1) × (K0 + K5 + the W-plane writes N·m·n̂ B + the β = 1 read of C, 8·m·n B) at
  0.9 TB/s, with the [M] phases at N = 14, it gives the S = 2 row for 4096³ and 8192³ (+17.4,
  +10.3 %), +31.6 % for DF, and the two net speedups (1.075×, 1.280×); its S = 4 row is
  +52.2 / +94.8 / +30.8 %, so the table's S = 4 row is 3× the rounded S = 2 one. The [E] stands
  as an estimate. The model's own composition of the S calls (S calls of k/S, every call after
  the first with β = 1: the [M] method M-a and DESIGN §3.8 use) is lower:

  | S | 4096³ | DF | 8192³ |
  |---|---|---|---|
  | 2 | +13 % | +24 % | +8 % |
  | 4 | +40 % | +72 % | +24 % |

  [M, composition], N = 14. With it, S = 2 at N = 14 is 1.15× (DF) to 1.31× (8192³) faster
  than fast mode at N = 20, and S = 4 on DF is still a net loss (0.82×) [M, model-only,
  outside hold-out: N = 20 is outside {8, 10, 14, 16}]. The [M] composition is the reference
  figure for any decision; the [E] stays as the first-round DRAM-roofline cross-check. Both are
  replaced only when a GPU measurement of the recipe exists (§5.8 MB-1).
- **Numerics.** Segmentation adds S − 1 FP64 roundings, so the result is no longer correctly
  rounded once, and its ELEMENTWISE bound is provably worse at high N (L7).
- **Against K-equilibration on decay_opp:** K-eq gives about 6 moduli at about 0 per-call cost
  for invariant operands or pre-scaling fused into the producer; about 6–13 % per call per
  changed operand otherwise (§2.6 [E]; corrected by IR12). K-segmentation gives 3–5 at
  +10–93 % [E] (+8–72 % [M, composition], M-b). K-segmentation needs only contiguous
  magnitude bands, not separable profiles, so it stays as a recipe.

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
| B2 | Rigorous ρ from the previous output (§4.2) | **adopt (stage 1), as the tested helper `precision.rho_from_output`** (IR7) | No new C library code: one pass over C plus the caller's row and column norms (a pass over A and B, unless the caller has them), and the existing NORMWISE bound. β = 0 and α = 2^e only. Checked against `cert_rho_recipe`, so the capability is rigorous, not documentation alone. |
| B3 | K5 statistics plus an a-posteriori certificate (L6), with one pure host function `imd_stats_cond` that turns the statistics into `cond` (P1 S2, reduced) | **formally deferred, triggers T1–T3** (stage 5, IR26) | Fuses B2 into K5. It needs no norms at all (L6 uses ‖a·2^σ‖ ≤ d), which saves the pass over A and B as well as the pass over C, and it adds a rigorous per-call elementwise certificate. Gate not met: no consumer issues elementwise targets (§4.6). |
| B4 | Choosing N on the device within a call | reject | N fixes the launch sequence, the K5 instantiation, W, the workspace and σ itself. Kernel scope forbids the sync. |
| B5 | Device-side selection feeding a caller-owned conditional (SWITCH) graph node (P1 S5) | park | No consumer runs captured adaptive iterations. |
| B6 | Output-selective refinement: run at N_lo, then recompute only the rows (or columns) whose ρ̂ exceeds the target's threshold, with a higher-N plan over a row sub-view or over rows the caller gathers (IR11) | **caller recipe, no API** | With ELEMENTWISE targets a few max-ρ outputs set N for the whole call, so the gain can be large on outlier data. It needs nothing new from the library: views exist, and stage 1's `rho_from_output` locates the rows. M-a estimates its value (§4.3). A device version would need a per-row max-r output (b·m bytes) in stage 5. |
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
| D7 | **K-equilibration** (P1) | **adopt, staged and gated** (oracle rule with multi-partner mode in stage 2; stages 4, 6 by pre-registered rules, IR27) | About 6 moduli on separable anti-correlated data. The library adds no per-call cost; the caller's pre-scaling and the helper do, for operands rebuilt every iteration (§2.6). Once w is fixed it does not depend on the partner; an operand with several partners gets one w from the union extrema (IR6). Exact (L5). It avoids regressions in the bound only with the guard (§2.6). |
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
- **The cheaper alternatives a revival must beat** (IR15):
  - *The prefix-plane option* [I, analytic]. It uses the first N planes of an N_max pack, with an
    asymmetric partner budget T_B ≈ H(N)/T(N_max). Its accuracy is about that of a symmetric
    N_eq ≈ 2N − N_max. It needs an asymmetric scaling rule in KF/KL, not a new pack format.
    Unlike A6 it re-scales the partner, so the CRT over the prefix stays exact [I].
  - *The share-group raise cost*: a group run at its max N against each member at its own floor
    with separate packs. M-b measures it on the DF-CCSD contraction set (§4.3). *Measured
    (M-b, [M], model-only and outside the hold-out where a level is not in {8, 10, 14, 16};
    immerdink-eval `findings/c1-m-b.md`):* on ccsd.py's pairwise DF contractions at the psi4
    sizes, with the forcing rule's equal ABSOLUTE targets and the real norm products, the
    members of a group share a floor in 11 of 12 (molecule, target) cases on `PrecisionPlan`'s
    default levels (6, 8, 10, 12, 14), so the raise costs nothing; the twelfth
    (benzene/cc-pVDZ, 1e−8) costs +16.5 % per iteration (20.45 against 23.83 TFLOPS-eq, 4.020
    against 3.450 ms), hh's floor reaching R2 and L2 through R5. That figure belongs to the
    default ΔN = 2 grid: on the integer levels 6–14 and on §2.4's 6/8/9/10/14 the same 11
    cases cost nothing and the twelfth, with hh's floor at 9, costs +8.1 % (22.07 against
    23.85 TFLOPS-eq, 3.726 against 3.448 ms; +8.7 % against the best separate). With one
    member's floor a level apart (ΔN = 2) the raise is −32 % to +25 % of the separate-packs
    time per iteration: negative where the separate variant's second pack of a dressed factor
    (re-packed every iteration) outweighs the raised calls, largest where the raised member
    dominates the group (R2's v⁴ output); from N = 6 to 14, up to +102 %.
  - *What a slice revival could save* is not the raise cost. The raise (shared against
    separate packs) is a difference of two overheads. A pack serving every N saves at most the
    best configuration without one (shared, separate packs, or best separate, where a lone
    member may stay in-line) minus the ideal: each member at its own floor, every grouped role
    reading its group's one pack, priced as an ordinary pack at the group's top floor, with no
    slice-mode per-call penalty. *Measured (M-b review, same records, [M, model-only, outside
    hold-out]), as a share of the best configuration's time per iteration:*
    - The 12 forcing-rule cases: 0 (to rounding), on each of the three grids. In the twelfth,
      separate packs are 0.014 ms (0.42 %) above the ideal (0.0066 ms, 0.19 %, with each pack
      priced at the shared plan's raised N instead), but the best separate, with hh's and R5's
      Boo roles in-line, is already 0.006 ms below it.
    - The 280 synthetic floor patterns: 0–9.1 % one level apart and 0–16.0 % over all (the
      most: benzene/cc-pVTZ Bvv with R5 or L2 at 8 and the other at 14, where the raise is
      +1.1 %); in 67 of them it exceeds max(raise, 0) by more than 0.5 points.
    - It is largest where a dressed factor's pack (re-packed every iteration) is a large share
      of small calls, which is where the raise is cheapest or negative (Bvv at cc-pVTZ:
      5.9–9.1 % one level apart, raise −13 to −11 %). It is small where the raise is largest
      (the Bvv group at cc-pVDZ, which R2 dominates: at most 3.2 %, raise up to +25 % one level
      apart), since separate packs already avoid most of the raise there.
  - *Limits of these figures:* (a) R1, R2, hh and Lld1 read one factor through both roles, and
    their A and B pack keys are equal at every level 6–14 (`m_b.py`'s key check). A
    `PrecisionPlan` share group takes one role per contraction, so their B role is modelled
    in-line: the figures are for that configuration, not the consumer's best (open item,
    Status, M-b). (b) The norm products are the bare (t1 = 0) psi4 factors'; Bvo takes Bov's,
    and the dressed Boo, Bvv and Bvo norms differ by O(t1). (c) The level grid, above. (d) R2 is
    eq 11's first pairwise step in the order M-a's R2 fixed; ring, p2, sub and add (t2 or u with
    Bov twice) are left out because M-a fixed no pairwise order for them. A Bov·Bov first step
    would have Lld1's extents and norm product, so Lld1's floor (not measured).

---

## 4. The chosen design, staged

| stage | size | gate | ABI | delivers |
|---|---|---|---|---|
| 0. Precision phases on host API 13 | S | — | none | first the library rebuild with the API-13 check (IR14), then the provenance manifest (IR9); C1(a): published monotone footprints (F3's second half only after `pack_write_extent`, IR17), layout certificates I1–I3, re-pack rule, Python guard (no TLA+ change, R1, IR20) |
| 1. `PrecisionPlan` and recipes | S | — | none | C1(b) on the host: per-role pack-cost composition, pack-mode masks and BUDGET, joint share groups, the DIRECT sentinel (IR2, IR8, IR24, IR15); the tested `rho_from_output` (`cert_rho_recipe`, IR7); schedule example |
| 2. Evidence | S (CPU) + M (GPU) | — | none | M-a at full size with sampled outputs, real psi4 operands, and multi-partner and batched fixtures, with `oracle.kweights` (IR1, IR6, IR16, IR19); M-b re-runs of the [M] figures; M-c small-N hold-out (IR4); scheduled iterations (convergence on the CPU, time on the GPU) |
| 3. L7 and the K-segmentation recipe | M (was S; IR5) | after the M-a summary (IR12, IR28) | none | L3c and γ_k in Lean, then L7 in PROOFS + Lean; `cert_kseg_composition`, `oracle.contract_segments`; DESIGN §7 rows citing M-a |
| 4. `imd_kweights` + caller-side equilibration | S–M | pre-registered rule (IR27, §4.5): **failed, closed** 2026-10-07, revival trigger in §4.5 | host API +1, not built | C2, step 1, with L5 in PROOFS (+ Lean), its certificates and `GuardSound` |
| 5. K5 statistics + certificate | S–M | **not met: formally deferred**, triggers T1–T3 (IR26, §4.6) | host API +1, if built | C1(b) on the device, with L6 in PROOFS (+ Lean) and its certificates |
| 6. Library-internal weights + weighted packs | M | pre-registered rule (IR27, §4.7); layout decided (IR21): **deferred** 2026-10-07, trigger in §4.7 | host API +1, not built | C2, step 2 |

Each lemma is proved in the stage that first relies on it (R2, §5.2). Its hand proof is
recorded in §4.4 now. Stage 3 adds the Lean prerequisite L3c (IR5). §4.11 gives the order of
work and the definition of done (IR28, IR29).

DESIGN v1.19 gets the following:
- §6g "Adaptive precision", with the stage 0–1 text, and the stage-5 API only if a trigger fires;
- a §3.8 addendum (L7, stage 3; L6 when stage 5 ships) and §3.9 "K-weights" (stages 4 and 6);
- updated §6a and §6b.7 contract text;
- rewritten §7 rows.

### 4.1 Stage 0 — precision phases on host API 13

**Order** (IR28). Nothing else in stages 0–2 runs before step (a) passes (IR14).
- **(a) Rebuild** this worktree's `libimmerdink_host.so` and `libimmerdink`. Run
  `tools/test.sh` with `IMMERDINK_BUILD_JOBS=4`, under memcap, after a FREE busy check. Assert
  that the library reports host API 13 through its version query, and that `nm` shows
  `imd_perf_rank`, `imd_perf_predict_pack` and `imd_err_min_moduli`. Record the library's sha256
  in the provenance manifest. The current build is host API 5 (Provenance).
- **(b) Provenance copy and manifest** (IR9; Provenance in Status), before any DESIGN v1.19 text
  cites a [C] or [M] figure.
- **(c) DESIGN §6g stage-0 text:** F1, F2, and F3's first half with the contract read extent
  (IR17); C-ws, C-pack and the re-pack rule.
- **(d)** `cert_footprint_monotone` (I1–I3) and `host_footprint_monotone`.
- **(e)** `PackSpec.with_moduli`, `PackedBuffer` with the `python_device` CPU part, and
  `python_overhead`.

The stage-0 GPU tests run in round 2 (§4.11).

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
  include/immerdink/immerdink.h:74-75). A packed contract reads only [0, `pack_bytes`) of a
  packed buffer: that is the buffer's ALIASING extent (immerdink.h:107), and CompatSpec's
  `NoOOBRead` (spec/PackedOperand.tla:353) models it.
  - **F3's second half, "`imd_pack` writes no byte beyond `pack_bytes`", becomes published ABI
    only after the GPU test `pack_write_extent` passes** (IR17). Until then the stage-0 DESIGN
    text publishes only the first half and the read extent. Under §6a an ABI promise cannot be
    withdrawn, so it needs evidence on every pack path, not one shape's guard zones.

**Internal facts, not ABI** **[cert: cert_footprint_monotone]**. They are true of today's
formulas, but no caller needs them, and a future format (§4.7's kw region after res, a slice
format, a range plan) may break them. Under §6a a published promise could not be withdrawn,
so they stay certificates that a format change must revisit:
- **I1.** In `imd_pack_layout`, the sig and flag offsets and bytes, the res offset and
  `pack_ws_bytes` do not depend on N; res(N′) lies inside res(N_max) (the drafted TLA+
  `LayoutPrefix`, moved here because `pack_layout` is static, pack.cpp:35-57).
- **I2.** `imd_plan_bytes` device bytes do not depend on N or g. Host bytes do depend on them,
  through the K4 shape keys.
- **I3.** Packing more roles never increases ws(N, g) (IR8). A packed role drops sA/fA/RA or
  sB/fB/RB and adds no region (src/host/workspace.cpp:65-68). Per R6 it stays a certificate,
  because a later format could break it. `PrecisionPlan.footprints()` reports the maximum over the
  allowed pack modes × levels from the queries, so no caller relies on I3.

*Proof.* Every §4a region is either independent of (N, g) or has N or g as a factor
(workspace.cpp:49-56). align₂₅₆ is monotone, and the total is a sum of aligned regions in a fixed
order. In the pack, res is affine in N (pack.cpp:44-47) and the earlier regions do not contain N.
I3: packing only removes regions from the sum. F3's first half is §6b.4. Its second half holds if
each pack kernel writes only regions of the layout, which is how they are built [I]; the GPU test
`pack_write_extent` checks it on every pack path before it becomes ABI (IR17). ∎

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
  N′ ≤ N_max. This follows from F2, F3's first half and the read extent, so it can be published
  before `pack_write_extent` passes: `imd_pack` at N′ writes all of [0, `pack_bytes(N′)`), and a
  contract at N′ reads only that range. The bytes above it keep stale planes that nothing reads.
  The offsets of I1 are not needed.
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
- **Optional model: not adopted** (IR20; decided by the independent reviewer). R1 stands, and
  stage 0 has no TLA+ change. The model is built only if the N-range plan is revived, as §3.2
  already requires, or if Eric asks for it. As recorded: in StreamSpec, make each cell a pair
  ⟨version, key⟩, with N in the key (R-num). Add an op c3 after post with a RAW edge post → c3,
  and bind each contract to its plan's key (R-api). The invariant is then "every contract reads
  cells whose last writer's key equals its plan's key, given the caller rule".
  `neg_repack_stale` (c3 keeps the old plan) violates it, and so does dropping the post → c3
  edge. Optionally, add a CompatSpec model at N′ with the buffer sized for N + 1. Building it
  would spend TLC time with no new assurance: the ordering half is modelled already, and the
  stale-handle half is §6b.10 #3's documented limit.

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
  - `host_footprint_monotone`: F1, F2 and the internal facts I1–I3 for `imd_workspace_size`,
    `_packed`, `imd_pack_layout` and `imd_plan_bytes` against params. The grid is
    m, n ∈ {1, 15, 16, 17, 255, 256, 257}, k ∈ {2, 96, 97, 1024, 1025, 131008, 131009},
    b ∈ {1, 3}, N = 2..19, g = 1..N, both vec modes, packed roles ∅/A/B/AB, both k_align
    policies, plus 10⁴ random shapes.
  - The `python_device` CPU part: `PackedBuffer` refusal through a fake library.
- GPU (round 2, §4.11; queued for a quiet machine, under `_wip/gpu.lock`):
  - `packed_repack_n`:
    1. Pack X at N = 14 into a buffer sized for N = 14 and contract.
    2. Re-pack at N = 8 into the same buffer and contract with the N = 8 plan.
    3. Both results must be bit-exact against the oracle.
    4. The stale N = 14 plan with a `PackedBuffer` must raise `PackMismatchError`.
    5. Guard zones show that `imd_pack` at N = 8 writes exactly [0, `pack_bytes(8)`), a smoke
       check only; the F3 evidence is `pack_write_extent`.
  - **`pack_write_extent`** (IR17), the evidence for F3's second half. It covers:
    - every pack emitter path (KF, KL, K2/K3);
    - r ∈ {1, 15, 16, 17, 33};
    - both k_align policies;
    - N ∈ {2, 3, 8, 14, 20};
    - RN and TRUNC.

    Each pack is written into an exactly sized window inside a sentinel-filled buffer, and every
    byte outside [0, `pack_bytes`) must keep the sentinel. One compute-sanitizer memcheck run
    with exact-size allocations and `--padding` is the second check. There is no TLA+
    write-footprint invariant: it would only restate the layout of a static formula, which is
    params territory (I1).
  - `workspace_shared_n`: one workspace sized for (14, g = 0); plans N ∈ {5, 8, 14} called
    alternately on one stream; all bit-exact; the LD_PRELOAD shim counts no allocation.
  - `graph_phases`: one graph per phase (N = 6 and 10), **sharing one workspace**, replayed
    alternately, bit-exact. It captures a **cold** phase, whose N was never launched in the
    process, as well as a warm one (IR10).
  - **First call at a new N** (IR10): in a fresh process per N ∈ {6, 8, 10, 14}, the first
    `imd_plan_create`, the first contract and the steady state. This switch cost is reported
    next to the re-pack cost, TFLOPS-eq first, time alongside.
  - **Plan-time preload, only if needed** (IR10). If cold capture fails, or the first replay
    costs > 2× the steady replay, `imd_plan_create` gets a preload: `cudaFuncGetAttributes` on
    every kernel the plan will launch (its K5<N>, its K4 configurations, its KF/KL/K1 variants).
    It is host-side, runs before capture, changes no ABI, and is documented in §6g. The test is
    then re-run. *Why:* NOTES-M4:220-222 records module loading on the first plan, and K5<N>
    instantiations get no attribute call (src/k4/k4_cfg.cuh:80 covers K4 only), so lazy loading
    could land inside a capture, which kernel scope and capture safety forbid [I].

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
- **Lazy module loading** at an N never launched in the process could land inside a capture.
  `graph_phases` tests cold capture, and the preload above is the fix if it fails (IR10).

### 4.2 Stage 1 — `PrecisionPlan` and recipes

**Placement.** `PrecisionPlan` lives in its own module, `python/immerdink/precision.py`, not in
`planner.py`. §6e.4 promises that planner.py computes nothing in Python (DESIGN.md:4204), and
`PrecisionPlan` does compute: levels, the monotone floor, share groups and the per-role pack-cost
composition (IR2). It calls only `Planner.min_moduli`, `Planner.rank`, `Planner.predict_pack`,
and the workspace and pack-layout queries; no GPU, no policy. §6e.4's "computes nothing" rule
covers planner.py only.

```python
# python/immerdink/precision.py
class PrecisionPlan:
    def __init__(self, planner, contractions, *, levels=(6, 8, 10, 12, 14), rounding="RN",
                 mode="EMULATE", share=(), packs=None, reuse=None, budget=0, monotone=True)
        # planner: a planner.Planner (calibration, regime and cache come from it, not duplicated)
        # contractions: {name: (spec, extents[, strides][, calls_per_iteration])}
        # levels: sorted allowed N (3-5 levels recommended); may end with the sentinel "DIRECT"
        # share: [(group, [names])] - contractions that read one packed operand use one N
        # packs: {name: mode or set of modes}, mode "NONE" | "A" | "B" | "AB": the allowed
        #        pack modes, ranked at every level, so the packing may differ by N (IR8)
        # reuse: {name: {"A" | "B": "call" | "iteration" | "phase" | int}} per packed role;
        #        default "iteration" (the pack is remade every iteration) (IR2)
        # budget: immerdink-owned bytes per contraction, as in rank; 0 = none (IR8)
        # monotone: True for tightening iterations; False for relaxed Krylov (§1.3)
    def configs(self) -> dict      # {name: {N: Candidate}} at every level, up front
    def phase(self, targets, *, iterations=None, fail_budget=None) -> Phase
        # targets: {name: planner.Accuracy}; iterations: the phase length
    def footprints(self) -> Footprints  # per name: max workspace over the allowed pack modes x
                                        # levels, the value per level, and per-level totals;
                                        # per packed operand: pack_bytes per level and max
    def reset(self)                     # forget the monotone floor: a new solve, geometry or restart
class Phase:   # lazy view
    N: dict; g: dict; vec_mode: dict; status: dict   # status: also MISSED, BUDGET or DIRECT
    packed: dict                        # name -> "NONE" | "A" | "B" | "AB" (IR2)
    pack_bytes: dict                    # name -> {role: pack_bytes} of the packed roles
    reuse: dict; assumed_reuse: dict    # the effective per-role reuse; where 'phase' fell back
    time_s: float; tflops_eq: float     # calls-weighted; TFLOPS-eq = sum 2bmnk / time
    err_bound: dict; err_typical: dict
```

There is no `noise_floor`: a sum of errors across contractions in possibly different metrics
means nothing.

**Reuse, per role** (IR2):
- 'call' = 1;
- 'iteration' = calls_per_iteration;
- 'phase' = calls_per_iteration × the phase's iterations, i.e. the `iterations` argument of
  `phase()`. If it is None, 'phase' falls back to 'iteration' (conservative), and
  `Phase.assumed_reuse` records the fallback. An invariant operand is re-packed at each phase
  switch, so 'phase' is its value.
- A role shared by a share group divides its pack cost by the total calls of all members that
  read that pack.

**Pack-cost composition** (IR2; decided by the independent reviewer). Rank takes one int32 reuse
for every packed operand, but a contraction's two roles can have different reuse. So
`precision.py` composes the ranking itself:
- One `Planner.rank` call per contraction per phase, over the levels ≥ the floor and the allowed
  pack-mode mask, with reuse = 1, requesting all candidates.
- Each candidate is re-keyed as
  time = contract_s + Σ_{role ∈ candidate.packed} pack_s(role, N) / reuse(role),
  with pack_s(role, N) = `Planner.predict_pack` of `imd_pack_desc_for(role)` at N, memoized per
  (contraction, role, N). The interval bounds time_lo/time_hi are composed the same way.
- Feasibility (status, budget, accuracy) is taken from the candidate unchanged; none of it depends
  on reuse.
- Order: feasible by the composed time, then rank's tie order (total_bytes, N, g, vec_mode,
  packed).

The re-keying is exact. time_s = contract_s + pack_s/reuse is the only reuse-dependent field of a
candidate, and contract_s, pack_s and the packed bits are exposed
(include/immerdink/immerdink_host.h:763-770). This keeps "stages 0–3: no ABI change" (§4.9).
`PrecisionPlan` runs once per phase switch, off the hot path, and ehrenfest's hot-path rank calls
are untouched. A per-role reuse in `imd_perf_rank_request_t` is **not** adopted. It is revisited
only if a hot-path consumer needs per-role reuse inside one C rank call.

**Budget and pack-mode masks** (IR8; no ABI change):
- `budget` keeps rank's meaning: immerdink-owned bytes per contraction (workspace + plan +
  packs + pack workspace). The docs state that the caller subtracts its own resident tensors,
  including the FP64 source of every packed operand, which must stay resident (§6b.10 #4,
  DESIGN.md:2211-2213). FP64 sources are needed whether an operand is packed or in-line, so they
  shift feasibility, not the ranking among modes. There is no extra parameter, so one budget
  meaning holds across the API.
- With a mask, the benzene-5Z case (an N = 14 pack of 17.7 GB plus the 9.97 GB FP64 source,
  over 24 GB) falls back to in-line at N = 14 and packs at lower N.
- A share group's shared role is fixed packed; masks apply to the other roles only.
- Joint multi-contraction budgets are caller policy. `footprints()` reports per-level totals so
  the caller can check them.

**`configs()`** ranks each contraction once per level (all g, the given vec modes, the allowed
pack modes, the composed reuse and budget) and returns the best candidate at each level. The
caller creates every one of these plans before the iteration loop, and uses
`PlanCache.freeze()` for capture (§4.1 Risks). With `k4_tune` ON, that multiplies the plan-time
GPU autotuning by the number of levels. A fixed-N rank needs no accuracy: the error estimate
depends only on (N, rounding, k), so accuracy makes every candidate at one N feasible or every
one infeasible.

**`phase()` rule,** for each contraction c:
1. n_c = `min_moduli(acc_c, n_lo = levels[0], n_hi = the top emulated level)`. A lowering error
   raises. The RN κ ≥ T levels are +∞ in the estimate, so they are skipped automatically. If no
   emulated level meets the target:
   - with `levels` ending in "DIRECT" (IR15), DIRECT is evaluated through
     `min_moduli` / `imd_err_estimate` with mode DIRECT: status DIRECT if it meets the target,
     MISSED otherwise. DIRECT is not in the default levels, so there is no silent switch to
     native FP64, which runs at 1:64 on the 4090;
   - otherwise the status is MISSED, with N = the top emulated level.
2. f_c = the smallest level ≥ n_c.
3. With `monotone`, f_c = max(f_c, the previous phase's **final** N_c), i.e. its N after the
   share-group choice and the rank choice. DIRECT is the top of the monotone order. `reset()`
   clears it.
4. **A contraction outside any share group** uses the composed ranking above over the levels ≥ f_c
   and its allowed pack modes, with the accuracy set, the given vec modes and the budget. Its best
   feasible candidate is the phase's choice. This equals the fastest composed `configs()[c][N]`
   over N ≥ f_c, which an implementation may use as its memo. It holds even where time is not
   linear in N (§2.3), and it replaces both the draft's `rank([N_c])` and §6f.3's
   `rank([n, n+1])`, which is the two-level case.
5. **Share groups are ranked jointly** (IR24; changed by the independent reviewer). The group
   floor is the max of its members' f_c. For each level N ≥ the group floor, group time =
   Σ_members calls × (the member's best composed time at N), with the shared pack's cost counted
   once per pack through the members' summed reuse. The N with the least group time is chosen
   for every member. Time is usually monotone in N, so this normally returns the group floor, but
   not when the packing and reuse terms differ by level. *[Stage-1 review, §5.5 S1-5, binding:]*
   a member whose floor is DIRECT reads no pack, so it leaves the group for that phase and is
   settled alone; the other members are ranked jointly over the emulated levels ≥ their max f_c.
6. **BUDGET** (IR8). If no candidate at any level ≥ the floor fits the budget, N is the floor
   level, the status is BUDGET, and the reported candidate is the plan-time SUCCESS candidate with
   the least total_bytes at that level. MISSED takes precedence over BUDGET. *[Stage-1 review,
   §5.5 S1-3:]* BUDGET only if that candidate meets the target; otherwise MISSED.
7. **TYPICAL targets.** p in §6f.2 is the failure probability *per call* (DESIGN §6f.2,
   [review] p). With `iterations` and `fail_budget` P given, `phase()` passes
   p = P / (iterations · Σ_c calls_per_iteration_c) to every TYPICAL target of the phase: a
   union bound over the phase's calls. Without them it uses each Accuracy's own `fail_prob`.

The cost is one `min_moduli` (about 1.7 µs), one rank over at most |levels| N values, and the
memoized `predict_pack` calls, per contraction per phase switch. A share group adds
|levels| × members re-keyed candidates. That is off the hot path, and memoizable through `Memo`.

**Not in the library:** the forcing rule, i.e. residual → target. It ships as an example in
`bench/accuracy/schedule.py` and the docs:
- after iteration t, give each contraction c an ABSOLUTE target η·r_t·w_c (Σ w_c = 1), with
  `norm_product_c` refreshed per phase;
- raise N only at an iteration boundary;
- use BOUND for the final phase and TYPICAL for intermediate phases, as the user chooses, with
  p = P/C as in step 7;
- for relaxed Krylov methods, use `monotone=False` with a decreasing schedule (matvec tolerance
  ∝ tol/‖r_k‖), §1.3.

**Rigorous ρ from the previous output** (ELEMENTWISE targets) **[cert: cert_rho_recipe]**. It
ships as a tested helper in `python/immerdink/precision.py` (IR7), not as documentation alone:

```python
rho_from_output(C, row_norm_ub, col_norm_ub, *, N, rounding, k, alpha=1.0, beta=0.0,
                l1_ratio=None) -> (rho_hat, certified_mask)
```

- It implements the recipe below exactly: U evaluated as compose() does, times (1 + 2^−40), then
  the nextafter chain.
- **Restrictions.** β must be 0, and α a power of two 2^e, since D = ldexp(C, −e) is exact for
  normal C. Anything else raises ValueError. An output whose C is not normal is uncertified.
- For β = 1 accumulations, as in CC residual builders, the docs say: take ρ from a probe call
  with β = 0 into a scratch buffer, once per phase. No α ≠ 2^e composition ships, because D
  cannot be recovered exactly from RN(α·D) (§3.7: E = RN(α·D); DESIGN.md:432).
- *Inputs.* A call at N with α = 2^e and β = 0, so its output C gives D = ldexp(C, −e). Upper
  bounds n_ij ≥ ‖a_i‖‖b_j‖ from the caller's row and column 2-norms, computed so that they really
  are upper bounds (§6f.1's existing obligation). *[Stage-1 reviews, §5.5 S1-7, binding:]*
  each bound lies along its own axis, a row bound axis −2 and a column bound axis −1. For a
  2-D C: a scalar, a 1-D row bound of exactly length m and a 1-D column bound of exactly
  length n, a row bound (m, 1) and a column bound (1, n). For a batched C (C.ndim ≥ 3; second
  re-review, MF4): a scalar, or a bound with exactly C.ndim dims, each batch dim C's or 1, a
  row bound with shape[−1] = 1 and a column bound with shape[−2] = 1 ((…, m, 1), (…, 1, n),
  (1, …, m, 1), (b, …, 1, 1)); a 1-D or 2-D bound on a batched C raises, with the keepdims
  spelling in the message (e.g. `r[None, :, None]` for one shared across batches). Anything
  else raises. A 0-D or 1-D C broadcasts its bounds elementwise.
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
  about 9 % of a call when every norm is recomputed (M-b: 24·4096² B, 3.2 + 6.3 = 9.5 % at
  0.9 TB/s and 8.5 % at the 1.008 TB/s peak, against the [M] call of 4.701 ms). Norms of an
  invariant operand are computed once per phase. The ρ that sets N is needed once per phase
  decision, so this cost amortizes over the phase.
- σ-derived norms, ‖a_i‖ ≤ d·2^{−σ_i} (L1b), are valid for nonzero rows only. A zero row has
  σ := 0, so its bound d·2^0 is vacuous and can inflate the maximum by up to about T. So the
  recipe uses real norms unless the caller knows its zero rows.
- The per-output ρ̂ also gives an a-posteriori elementwise bound for that call, and it locates
  the rows for output-selective refinement (§3.1 B6).
- A stage 5, if a trigger fires (§4.6), would fuse this recipe into K5 and **need no norms at
  all**: L6 uses ‖a·2^σ‖ ≤ d, which the scaling rule guarantees. That is the hook's real
  advantage over the recipe.

**Tests.** A new CPU test `host_precision` of the module, next to `host_planner`:
- `PrecisionPlan.phase` equals a brute-force composition from `predict`, `predict_pack` and the
  footprint queries, at every level, g, vec mode and pack mode (IR2), and equals the
  `configs()` lookup;
- per-role reuse, including the 'phase' → 'iteration' fallback and `assumed_reuse`, and a shared
  role's cost divided by the members' summed calls;
- share groups: the joint choice equals a brute force over the levels ≥ the group floor (IR24),
  and members get equal N;
- N is monotone over phases until `reset()`, and the levels are respected; DIRECT only when
  no emulated level meets the target, and only if listed (IR15);
- BUDGET: the floor level and the least-total_bytes SUCCESS candidate; MISSED takes precedence
  (IR8);
- TYPICAL targets get p = P/C when `iterations` and `fail_budget` are passed;
- footprints equal the maximum of the C queries over the allowed pack modes × levels;
- MISSED and κ ≥ T levels are handled;
- `rho_from_output` against params' `cert_rho_recipe` in exact rationals, on random rows and on
  rows with |D| just above U·n; ValueError for β ≠ 0 or α ≠ 2^e (IR7).

### 4.3 Stage 2 — evidence

**Prerequisite.** Stage 0 step (a) (§4.1): the rebuilt library with the API-13 check (IR14).
Every stage-2 record names this worktree's library (commit, sha256, host API).

**M-a (CPU, round 1): the combined C2 study.** It merges P1 S0, P3 stage 0 and P4 stage 3a. The
protocol is the independent reviewer's (IR1), and it replaces the previous "sizes and seeds"
(m = n ∈ {16, 128, 512}). The Q13 gate needs evidence at the ≥ 2048³ class, and §2.6 shows that
random-structure gains vanish as m grows, while every guard quantity costs only O(mk).
- **Harness:** `bench/accuracy/c2_gate.py`, oracle arithmetic only. It compares immerdink's own
  scalings, so it belongs in the product repo. Raw records go to immerdink-eval
  `results/c1/m_a/` as JSONL (untracked); the summary goes to `findings/c1-evidence.md` (it
  covers M-d1 too; the name `c1-m-a.md` planned here was not used).
- **`oracle.kweights`** (spec/oracle.py) is added here, not in stage 4, with its multi-partner
  mode (§4.5, IR6). M-a needs one exact reference for the §4.5 rule, the clamp, the L5c guard
  and the gain G, and until the Q13 gate passes that reference is C2's deliverable (Q8). Only its
  self-tests come with it; its certificates, PROOFS L5 and `GuardSound` wait for stage 4 (R2,
  §5.2).
- **Engine:**
  1. *Full-operand quantities*, exact, over **all** rows and columns:
     - c_A/m_A/c_B/m_B per shared K index, Δ, the median, and w and the clamp counts per scheme;
     - σ_i, σ̃_i, τ_j and τ̃_j at N_ref ∈ {8, 14}, and from these gain_row_min, gain_col_min, G
       and the guard. Both N_ref are reported, and a run whose guard differs between them is
       flagged;
     - the worst pair W = min_i log₂(‖a_i‖/‖ã_i‖) + min_j log₂(‖b_j‖/‖b̃_j‖). It is separable,
       so it costs O(mk) with no products. FP64 norms are allowed for W, which is labelled a
       diagnostic;
     - percentiles of σ̃ − σ.
  2. *σ engine.* (S, γ) do not depend on N; only u does. A vectorized engine is allowed only if
     it is bit-identical to `oracle.scale_vector`. The recommended form is an FP interval
     pre-pass on S with a rigorous error bound, plus an exact Python fallback for any row whose u
     is ambiguous. The existing per-element Python S loop (bench/accuracy/common.py:139-154)
     would take minutes per operand at k = 16384 [E]. **Self-test before any record is written:**
     ≥ 2000 rows (random, near the u boundary, subnormal, single-entry, zero) at
     N ∈ {3, 6, 8, 14, 20}, RN and TRUNC, 0 mismatches.
  3. *Row-blocked processing*: ≤ 256 rows or columns per block, peak RSS ≤ 3 GB per run.
  4. *Sampled outputs*: 64 rows × 64 columns per run.
     - Rows: up to 4 argmin_i(σ̃_i − σ_i) at N_ref = 14, ∪ up to 4
       argmin_i log₂(‖a_i‖/‖ã_i‖), ∪ seeded random rows up to 64. Columns are chosen the same
       way. So the sample is adversarial exactly where min G is set.
     - On the sample: exact x, and the emulated D of every scheme at every N = 4..20 under RN,
       and under TRUNC at N ∈ {8, 14}. Limb BLAS with K chunked by 4096, so every partial sum is
       < 2^53.
     - The CRT precondition is asserted on every sampled output of every scheme. For fast and
       K-eq it holds on the full operands by L1c.
- **Schemes:**
  - *Tier S:* fast; K-eq with deadband d ∈ {0, 2, 3, 5}, each with its guard report; K-seg
    S ∈ {2, 4}; K-eq(d*) + K-seg S = 2; the envelope rule (Appendix B), the reference for the
    best partner-aware σ/τ without a GEMM. Exact-integer SC25, the reference for "accurate mode
    as published", runs at m = n = 128 only: it needs a full m×k×n GEMM. It and `acc_exact` are
    reference schemes already rejected (§2.5).
  - *Tier L and the real operands:* fast, K-eq d = 0, K-eq d* (guarded), K-seg S = 2,
    K-eq(d*) + K-seg S = 2, and the envelope rule.
- **Deadband d*** (settles Q11). It is chosen at Tier S, m = n = 512, over 5 seeds: the d with
  the fewest guard rejections on the random-structure fixtures (outliers, φ = 2 and 4, mixed),
  subject to the median gain on decay_opp and opp_nonsep staying within 0.25 moduli of the best
  d. Ties go to 3.
- **Fixtures and sizes** (m = 16 is dropped):
  - *Tier S* (deadband tuning, continuity with §2.6), 5 seeds: m = n = 128 with
    k ∈ {600, 3600} and full outputs; m = n = 512 with k ∈ {3600, 16384} and sampled outputs.
    Fixtures: uniform, φ ∈ {0.5, 2, 4}, decay_same, decay_opp, opp_nonsep, outliers, cancel,
    same-sign, local_df, mixed, and the synthetic CC family.
  - *Tier L* (gate evidence): m = n = 2048 with k ∈ {3600, 16384} and 3 seeds; m = n = 4096
    with k = 3600 and 2 seeds. Fixtures: outliers, φ = 2, φ = 4, decay_opp, opp_nonsep, uniform
    (control), and the synthetic CC contractions F_DF, F_PPL and F_ring.
  - *Synthetic CC family*, documented in the fixture docstring, with its modelling assumptions
    tagged [I]:
    - occupied ε: o/7 core orbitals at −11 ± 0.5 Ha, valence uniform in [−1.5, −0.3] Ha;
    - virtual ε: log-spaced over [0.05, 50] Ha;
    - DF factors B_{Q,pq} = s_Q·g(ε_p)g(ε_q)·N(0,1), with s_Q decaying over 8 decades along Q
      and g(ε) = (1 + |ε|)^{−1/2};
    - MP2 amplitudes t = (ia|jb)/(ε_i + ε_j − ε_a − ε_b);
    - contractions F_DF (K = Q), F_PPL (t2 × W over K = cd) and F_ring (K = ld).
  - *Real operands*: R1–R7, below (IR19).
  - *Multi-partner* (IR6): the real Bov with its Q-partners {Bov, Bvo}, plus a synthetic X with
    3 partners (decay_opp-like, φ = 4, uniform).
  - *Batched* (IR16): the real R7, iaQ,ibQ->iab with batch i over **all** occupied orbitals,
    core included; and a synthetic batch whose λ < b/4 slices decay along K at a different rate
    from the rest. Each reports the gain of a shared w (Q12 a) against per-λ w (Q12 b, from
    `oracle.kweights` per slice), with the guard for both. Pre-registered trigger: Q12 (b) is
    reconsidered only if per-λ weights save ≥ 1 modulus beyond shared weights on a real fixture
    at the same guard status. Otherwise Q12 (a) stands, and the escape clause is closed.
- **Real operands** (IR19; decided by the independent reviewer). No ehrenfest dump is requested
  or performed in this cycle. ehrenfest saves no tensors (checked read-only), and its integrals
  come from a psi4 sidecar at runtime, which would mean running ehrenfest code.
  - *Generator:* a new script, immerdink-eval `scripts/c1/psi4_operands.py`. It is our code: it
    must not import or run anything from `/home/ef/repos/ehrenfest-ipdps`. It uses the psi4
    already installed in the `main` conda env, read-only; nothing is installed into that env.
  - *Run conditions:* CPU-only, `psi4.set_num_threads(1)` and OMP/MKL/OPENBLAS = 1,
    `set_memory` ≤ 4 GB, under `tools/memcap.sh`, after a FREE busy check, `timeout 600` per
    molecule, ≤ 15 CPU-min in total, inside the M-a budget.
  - *Molecules:* benzene/cc-pVDZ (for PPL), benzene/cc-pVTZ, and water/cc-pV5Z (the 5Z virtual
    spectrum of the consumer's target basis). Each run: DF-SCF; MO-basis RIFIT factors
    B^Q_{oo, ov, vv}; orbital energies; MP2 amplitudes t = (ia|jb)/D. No frozen core.
  - *Storage:* `.npz` files in immerdink-eval `results/c1/real/` (untracked). `c2_gate.py`
    loads them by path argument and never depends on psi4.
  - *Contractions*, taken read-only from the consumer's own ehrenfest
    examples/chemistry/coupledCluster/ccsd/ccsd.py:
    - R1 aiQ,bjQ->abij (K = Q, TZ);
    - R2 acQ,bdQ->acbd and R2b ijcd,acbd->abij (K = cd, DZ);
    - R3 ilad,ldkc->aikc (K = ld, TZ);
    - R4 kjbc,kiac->abij (K = kc, TZ);
    - R5 kiQ,acQ->kiac (K = Q, TZ);
    - R6 the multi-partner Bov over Q;
    - R7 the batched iaQ,ibQ->iab.

    Each is of the ≥ 2048³ class (2mnk ≥ 1.7e10) except R7, which is a fixture for Q12 only.
  - *Status for Q13:* these count as real, consumer-confirmed operands: real molecules, on the
    consumer's own contraction list. Recorded assumptions [I]: MP2 amplitudes stand in for
    converged t2, and bare integrals for t1-dressed ones (the t1 dressing is small).
  - *Fallback:* if psi4 cannot run in that env, the stage-4/6 gates are formally deferred with
    the trigger "real operands available". The Eric-approved ehrenfest dump is then the named
    fallback route.
- **Reported:**
  - per run, as JSONL in immerdink-eval `results/c1/m_a/`: commit, library sha256, host API,
    CPU-seconds, peak RSS;
  - per fixture, as a function of m ∈ {128, 512, 2048, 4096}: the guard acceptance rate, G
    (median and min), the worst-pair bits, and the moduli saved at equal sampled
    max-elementwise (1e−6 / 1e−8 / 1e−10 / 1e−12) and max-normwise (1e−8 / 1e−12 / 1e−15) error;
  - **realized regressions among guard-accepted runs** (the guard bounds the error bound, not the
    error): sampled outputs whose weighted error exceeds fast's at the same N by > 2×, as a count
    and the max ratio;
  - stale-weight rates (weights from seed s applied to seed s + 1);
  - the L5d moduli, i.e. what `min_moduli` picks for the weighted call from the plain
    statistics, against the measured minimum;
  - the **B6 estimate** (IR11), on outliers, opp_nonsep, φ = 4 and the real operands: the fraction
    of rows with max_j ρ_ij above the threshold, and the [M] time t(N_lo, full) + t(N_hi, rows)
    against t(N_hi, full). It uses full outputs where m·n ≤ 512², and FP64 x on the real
    operands (an estimate, labelled);
  - [M] TFLOPS-eq first, time alongside, from `imd_perf_predict` of this worktree's rebuilt
    library, with every cost term: for K-seg, the S-fold K0/K5/W writes and the β = 1 read of C;
    for K-eq, the helper (one or two operand reads) and the pre-scaling (16 B per element per
    changed operand per call unless fused), as in §2.6.
- **Gate admissibility.** Only Tier L and real-operand runs feed Q13 (the stage-4 rule, §4.5,
  IR27). Tier S tunes d* and shows the trend.
- **CPU budget (hard):**
  - every run single-threaded (OMP/OPENBLAS/MKL = 1), under `tools/memcap.sh`, after a FREE busy
    check in its own call, with `timeout 600` inside memcap (`tools/memcap.sh timeout 600 cmd`).
    memcap runs the command in a systemd scope, and a timeout around memcap.sh does not stop
    it (§5.6 S2-7);
  - the whole of M-a ≤ 90 CPU-min: self-tests + pilot ≤ 10, Tier S ≤ 25, Tier L ≤ 25, real
    operands (generation included) ≤ 30;
  - pilot first: outliers, 4096², k = 3600, seed 0. The total is projected from it;
  - if the projection exceeds 90, cut in this pre-declared order: (1) Tier L 4096 seeds 2 → 1;
    (2) Tier S m = 512, k = 16384 kept only for decay_opp, opp_nonsep, outliers and φ = 4;
    (3) Tier L 2048, k = 16384 kept only for decay_opp, opp_nonsep and synthetic CC;
  - never cut the real operands, the guard, or m = 4096;
  - the per-tier allotments trigger cuts too (§5.6 S2-6): a tier whose projection exceeds its
    allotment applies the earliest pre-declared cut that reduces that tier, under the same
    "never cut" rule, and the projection that triggered it is recorded. In the 2026-10-07 run
    Tier S projected about 26 min against its 25, so cut (2) was applied.

  The machine was BUSY during the review, so these costs are estimates. The pilot plus the
  pre-declared cut order keeps the 90-minute bound enforceable without post-hoc choices.

**M-b (CPU, round 1): model trajectories.** Per-phase TFLOPS-eq and time for square and
CC-shaped cases, schedule totals against fixed N, and the re-pack cost per switch
(`predict_pack`). §2.3–§2.4 already hold most of this. In addition:
- **Re-run every [M] figure that DESIGN will cite** with this worktree's rebuilt library (IR9).
  A figure that differs by > 0.5 % is replaced, and the change is noted. This also completes
  R-num's unfinished [M] re-runs (Provenance).
- **The share-group raise cost** (IR15) on the DF-CCSD contraction set (ehrenfest ccsd.py,
  read-only): the group at its max N against each member at its own floor with separate packs,
  [M], TFLOPS-eq and time.

*Outcome (2026-10-07; immerdink-eval `findings/c1-m-b.md`, records `results/c1/m_b/`).*
- *Re-runs:* the design-phase scripts, run unchanged on this worktree's library (sha256
  `466acdb4…`, host API 13), reproduce every stored [M] record byte for byte, and so do stage
  2's (M-a's `perf` blocks, `c2_summary.md`, `b6_split_cost.txt`). Of 191 cited figures
  recomputed from fresh queries, 169 print as cited, 7 differ in the last digit only
  (≤ 0.42 %), and 6 differ by more: §2.3's re-pack share (a text error, corrected: 9–11 %,
  not "about 11 %"), four of §2.7's first-round [E] K-segmentation costs and §4.2's "about 9 %"
  [E] (kept as estimates and annotated in §2.7 and §4.2). §2.7 also gets the [M] composition
  of the S calls. The 7 last-digit cases were mis-roundings of unchanged predictions, just
  below a tie; corrected in place after the M-b review (§2.3, §2.6; §5.8 MB-2).
- *M-d2's bars* (below), unrounded: 0.750 (PPL view, best g), 0.664 (DF), 0.673 (4096³);
  0.786 on the Jacobi view. The best-g and g = N predictions coincide on these shapes
  (`PrecisionPlan` picks g = N at every level), so the bars hold whichever M-d2 runs.
- *The share-group raise cost:* §3.3. The contraction set is ccsd.py's pairwise DF
  contractions (R1, R2 at cc-pVDZ only, hh, R5, both L_aikc and both L_ldkc terms) at the M-a
  psi4 sizes, with four groups: Bvv {R2, R5, L2}, Boo {hh, R5}, Bvo {R1, L1} (t1-dressed,
  re-packed every iteration) and Bov {L1, Lld1, Lld2} (bare, re-packed per phase). L2 reads Boo
  with another pack key than hh and R5 (`imd_pack_layout` digests), so it is not in the Boo
  group. The joint ranking chose the group floor in all 292 scenarios. The review added the
  bound on what a slice revival could save (§3.3), the forcing cases on two more level grids
  and the pack keys at every level 6–14 (§5.8).

**M-c (GPU, round 2): small-N hold-out** (IR4; decided by the independent reviewer). It runs
**strictly before M-d2**.
- N ∈ {4, 5, 6, 7, 9, 10, 12, 13}, plus N ∈ {8, 14} as anchors on the same shapes, so that a
  shape error separates from an N error.
- Shapes: sq1024, sq2048, sq4096, DF acQ,bdQ 64⁴ with Q = 2032, QZ kiQ,acQ (o 21, v 489,
  Q 1122), skinny 256²×65536, the PPL view 100×9216×9216 and the Jacobi view 1600×100×1600.
  A script checks by extents that every shape is absent from `bench/sweep_calib.txt`; if one is
  not, the nearest unlisted size is used (e.g. 2000, 4000). sq2048 is in the calibration set
  today [V], so it becomes such a size.
- Data: dense normal, and decay_opp-like, to cover [KT26]'s data-dependent INT8 throughput.
- About 160 cases, ≤ 15 min of GPU time, under `_wip/gpu.lock`.
- Judged per size class against §6c.5 (median and p90 of the total), and reported per N.
- If a target is missed: per-class residuals first, then N-specific calibration cases under the
  existing format, with optional keys only (Q24; no new features).
- **Until M-c passes**, DESIGN marks the default levels (6, 8, 10, 12, 14) and every §2.4 or
  §4.10 schedule gain that uses N ∉ {8, 10, 14, 16} as [M, model-only, outside hold-out].
- *Outcome (2026-10-07): failed; the Q24 refit was not adopted* (Status, "M-c"; binding
  verdicts §5.9).
  - The < 100 µs median is 15.8 % (target 15 %; repeat 16.1 %). Two effects set it, each
    necessary: the two sq1024 calls at N = 4 (KF path) are 16–21 points (repeat 15–16) more
    optimistic than the mean of sq1024's anchors' errors (8.2 / 15.4 points against the
    N = 8 anchor alone), and they are the median; the Jacobi view (26–29 % fast at N ≤ 10,
    the anchor N = 8 included; −14 to −15 % at N = 12–14) puts 12 calls above them. The
    Jacobi error is two KL effects whose errors cancel at the call total: KL A keeps only
    27–50 % of its modeled REPEAT credit, and KL B, the narrow pack, takes ×1.5–2.0 its
    prediction in every cache state (Status, "Cause"). The other classes are met over all
    calls.
  - The marks stay; there is no partial pass. The path to a pass is the MC-3 feature as
    amended (§5.9 MC-3, MC-3a): DESIGN §6c text from a per-phase account first, at most two
    optional keys, a refit against the phase marks, a fresh narrow-KL hold-out, then the
    full M-c case list measured again.

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
- **M-d1 (CPU, round 1): convergence.** The bit-exact emulator (checked against
  `oracle.contract`) runs the iterations, so no GPU is needed. Systems at reduced size where
  needed (the PPL-like system at v ≈ 40), single-threaded, under memcap, when ehrenfest is FREE.
  - Metrics: iterations to reach tol ∈ {1e−7, 1e−10}; the final residual in long double; drift
    against a double-double reference iteration (CPU Dot2 contractions) and against the N = 20
    trajectory; for DIIS, iteration counts with and without the reset at each switch.
  - **Success:** the ramp reaches the fixed-N_final residual within 2× and with at most one more
    iteration, on every system, DIIS included. *Outcome (2026-10-07): met on every system and
    ramp* (Status, stage 2).
  - **Report-only cases** (IR3). They are not C1 gates; they document caller guidance. Block
    forms give the emulated contraction a GEMM shape, which is what immerdink is for; a plain
    GEMV gains nothing from INT8 emulation. A's σ is computed once per N.
    - *Block Davidson:* the lowest 4 roots of a symmetric n = 2048 matrix,
      diag(Δε_ia-like) + DF-like coupling, block 8, subspace ≤ 48 with restart. σ = H·V is an
      emulated 2048×2048×8 GEMM. Schedules: fixed N ∈ {8, 14}, and a tightening ramp over the
      levels {6, 8, 10, 12, 14} with **absolute** targets η·‖r‖. Reported: iterations to
      residual 1e−7 and 1e−10, and the eigenvalue error against FP64 `eigh`. Expected: the ramp
      needs ≤ fixed-N_final + 1 iterations. *Outcome (2026-10-07): not met;* +5 even with a
      recompute at each switch (§1.3).
    - *Block CG:* SPD A = BBᵀ/k + μI from DF-like factors, n = 2048, cond ≈ 1e6, 16 right-hand
      sides, with an emulated GEMM matvec. Schedules: fixed N ∈ {8, 10, 14}; monotone
      tightening; relaxed decreasing (matvec tolerance ∝ tol/‖r_k‖). Reported: iterations to a
      true residual of 1e−7 and 1e−10, recomputed exactly, and the attainable accuracy.
  - **ρ-miss rate** (IR7). A schedule variant with ELEMENTWISE targets, whose cond comes from
    the previous iteration's ρ̂ (`rho_from_output`), records per iteration whether the current
    call's realized max elementwise error missed the target. The miss rate is reported per
    system: it is the number a caller needs to trust the lagged ρ. *Reading (§5.6 S2-8):* the
    rate is over the calls whose lagged request was MET at some level ≤ 20, against the
    previous call's certified outputs; MISSED (capped) requests are reported apart, and the
    levels go up to 20 for this variant only.
  - **Budget:** M-d1 ≤ 30 CPU-min in total, each run ≤ 10 min, single-threaded, under memcap,
    after a busy check.
- **M-d2 (GPU, round 2): time, on realistic shapes.** It runs only after M-c (IR23). If M-c
  triggers a recalibration, the predicted ratios and bands below are re-derived and
  pre-registered before M-d2 runs. *After M-c failed (binding, §5.9 "M-d2 ordering"):* M-d2
  waits until the MC-3 path closes. If the calibration changes, the bars are re-derived and
  committed before M-d2; if MC-3 closes without a change, M-d2 runs on the pre-registered
  bars unchanged (0.750 / 0.664 / 0.673, ± the §6c.5 p90 of the class, and ≤ 0.8). M-c's
  indicative measured ratios never set or adjust the bars. Systems: the o = 10, v = 96 PPL view
  (100 × 9216 × 9216; predicted schedule/fixed-14 = 0.75 [M, review]), DF (0.66) and 4096³
  (0.67). The Jacobi view (predicted 0.79) is reported but not gated.
  - Metrics: measured TFLOPS-eq and time per phase against the predictions; re-pack time at the
    switches.
  - **Success, per system:** the measured schedule/fixed-14 ratio lies inside that system's own
    prediction band, i.e. its predicted ratio, stated next to the bar, ± the §6c.5 p90 error of
    its call class; and it is ≤ 0.8 on the PPL view, DF and 4096³. The gain against a fixed
    N = 8–9 is reported even if small.
  - **Pre-registered protocol** (2026-10-07, before any M-d2 measurement). Not run: it waits
    for the MC-3 path (Status "M-d2").
    - *Normative driver:* `bench/accuracy/md2_time.py` at **35eeb46**, sha256
      `90fa7f92a1439f4056253aa69d378fadda4cee3751b2ec5ce0b221ca262f5b9a`; its docstring is
      normative. It replaces daf3583 (sha256 `b4cb9326…`), which the re-review found to
      leave forking paths open outside one records file, and 3254b15 (sha256
      `95ed2f6c…`), whose judge the first review found wrong (§5.10).
    - *Decisions:* §5.10 MD2-0 to MD2-8, reviewed twice (approve-with-fixes both times; the
      verdicts are binding).
    - *If the calibration changes (MC-3a):* re-run M-b's [M] figures and `predict` (which
      then exits 1). Update the bars, the band classes and `REGISTERED_CALIB` in the
      driver, and commit before any M-d2 run. That re-registers the driver: a new commit
      and sha256 here (allowed before the first M-d2 record; below).
    - *Required at the next re-registration, before any M-d2 run, whether or not the
      calibration changes* (final review of C1/C2, binding): `run`'s bit-exact check (rule
      (a)) fills C with a NaN bit pattern before the in-line reference call and before the
      packed call, and counts a level as bit-exact only if the in-line call wrote every
      element (no pattern left) and the packed result equals it. At 35eeb46 C is not reset
      between the two calls, so a packed call that writes nothing would compare equal
      (`tests/test_phases.py` guards the same case: "the reference is not vacuous"). Not
      applied now: changing the driver re-registers it, and the review placed the fix in the
      next re-registration; M-d2 cannot run before one (the ordering above).
    - *Records, freeze and continuation* (binding; the re-review's must-fix, §5.10 MD2-8 as
      amended; mirrored in the driver's docstring):
      - *One normative records file:* immerdink-eval `results/c1/m_d2/md2.jsonl`. Every
        M-d2 record `run` ever writes, in any file, is an attempt under rules (b) and (c)
        below and is reported. Starting another file resets nothing.
      - *The driver is frozen from the first M-d2 record on.* Before it, re-registration
        is a commit plus this section and the findings naming it, as now. From the first
        M-d2 record on, the driver is re-registered only with an independent-review
        verdict recorded in §5.10. Records made under an earlier driver stay reported and
        binding, judged by the driver that made them (`git show
        <commit>:bench/accuracy/md2_time.py`, run in a checkout of that commit, e.g. `git
        worktree add`).
      - *Nothing continues automatically* after rule (d) (two inadmissible attempts) or
        M-d2 STOPPED. A further attempt, a new records file, or a restart after a library
        fix needs a reviewer verdict or Eric's decision, recorded here. That record names
        the cause or the fix and the new file, and every earlier record stays reported.
      - *A refused file is a procedural breach, not a reset.* If `judge` refuses a records
        file holding M-d2 records (a provenance failure, below), the breach is recorded
        here: which record, and how it got there. Continuing in a new file needs a
        recorded reviewer verdict or Eric's decision.
    - *The library:* `build/libimmerdink.so` (and its `libimmerdink_host.so`), built at the
      registered commit before the first M-d2 run; the findings name its sha256 before
      that run. `run` refuses the M-d2 systems with any other `--lib`.
    - *`--md2-cleared`* must cite the commit and the NOTES-C1 section that record MC-3's
      closure (MC-3a's recorded outcome, or Eric's decision; §5.10 MD2-0), e.g.
      "MC-3a's outcome recorded at <commit>, NOTES-C1 §5.n". `run` and `judge` check that
      the reason names a "NOTES-C1 §n.m" section and a commit that is the record's commit
      or an ancestor of it, so each record points at that evidence.
    - *Configuration:* `PrecisionPlan.configs()` over the levels 6, 8, 9, 10, 14, in-line,
      GENERATE, RN: M-b's view, which gives g = N at every level here. The invariant
      operand is B (the integrals) for the PPL view and DF, and A for 4096³ and for the
      Jacobi view's W.
    - *Runs*, 16 calls each, back to back on one stream through `Plan.contract_ptr`:
      - S14 = 5×N6 + 4×N8 + 4×N10 + 3×N14 and F14 = 16×N14: the gated pair;
      - S9 = 6×N6 + 10×N9, F9 and F8: reported (the gain against fixed N = 8–9);
      - P14, S14 with the invariant operand packed into one buffer before each phase (the
        initial pack and the three re-packs), and PF14, one pack at 14 and then 16 packed
        calls: reported (re-pack times against `imd_perf_predict_pack`).

      Before timing, packed and in-line results are checked bit for bit at every level.
    - *Timing:* GPU events, after a 1 GiB memset primer so that the host enqueues each run
      ahead of the GPU (`queued_ahead`).
      - One fresh process per system and data set, under `tools/run_quiet.sh`: the quiet
        gate, the GPU lock and memcap, FREE only, stock clocks (§6c.5's regime, as in M-c).
      - A 3 s warm-up of the system's own N = 14 call, then every level once.
      - 15 reps in rotated order. If the S14 or F14 spread exceeds 5 %, a second attempt
        runs and the attempt with the smaller worst spread is kept.
    - *Data:* normal ((u − ½)·exp(0.5 g)) and decay_opp (`c2_fixtures`). Both are gated.
    - *Statistic:* r = the median of the S14 run totals / the median of the F14 run totals.
      It must lie inside bar × (1 ∓ p90) and be ≤ 0.8. Every level of the three gated
      systems is predicted ≥ 1 ms, so p90 = 15 %; a system whose levels span classes takes
      the largest p90 among them (the Jacobi view: 30 %).
      - The bands are [0.6375, 0.8000] (PPL view), [0.5644, 0.7636] (DF) and
        [0.5721, 0.7740] (4096³); the Jacobi view's [0.550, 1.022] is reported only.
      - A system passes when both data sets are inside; M-d2 passes when all three
        systems pass.
    - *Admissibility, re-runs and the decision* (the reviewer's binding rule, §5.10 MD2-6;
      per system and data set; `selftest` checks it):
      - (a) A record with `contaminated` not false, or whose bit-exact check is not true at
        every level (a missing or non-boolean value included), is inadmissible: neither
        PASS nor FAIL. A bit-exact failure (false at any level) is a library defect and
        stops M-d2 (M-d2 STOPPED) until it is fixed and recorded. This holds for a
        bit-exact failure on *any* record made with the M-d2 library, smoke records
        included, in any file; `judge` and `run` apply it to every record in the file.
      - (b) At most one re-run, in a fresh process after a FREE busy check in its own call,
        appended to the same records file.
      - (c) The first admissible record in file order decides, and every record is
        reported. A later record never replaces it, and a record past the second is
        excess, never considered.
      - (d) If both attempts are inadmissible, the pair is INCOMPLETE, and M-d2 is not
        decided until the cause is recorded here; nothing continues automatically (above).

      A system fails when either data set is outside. M-d2 fails when any system fails,
      and is INCOMPLETE while any pair is undecided.
    - *Provenance:* `judge` refuses a records file holding an M-d2 record that fails any of
      the following, and `run` refuses to make one, before any GPU work:
      - the registered calibration file (`d31f62ca…`);
      - the loaded library's shipped calibration (embedded at build time) equal to that
        file, as canonical text and as bit-identical predictions (`calib_check`);
      - `registered_check`: the loaded host library reproduces the system's registered bar
        (rank's choice and g = N, to 3 decimals) and its p90 class, as `predict` checks.
        `calib_check` compares two calibrations on the same model code, so this catches a
        model-code change under an unchanged calibration;
      - a clean tree, and an `--md2-cleared` that cites the closure (above);
      - the judging driver's own sha256;
      - the protocol's fixed parameters (15 reps, 2 attempts, 3 s warm-up, 1 GiB primer,
        seed 1, 5 %);
      - the bars' configuration at every level (N moduli, g = N, GENERATE, in-line, RN,
        EMULATE);
      - per file, one library: the M-d2 records share one (`lib_sha256`,
        `host_lib_sha256`, `host_api`).

      `run` also refuses a re-run of a decided pair, a third record, and any M-d2 system
      after a bit-exact failure in the file. `md2_time.py selftest` (95 checks, the bars,
      bands, runs and protocol constants pinned as literals) is the CPU-only ctest
      `md2_time_selftest`.
    - *Budget:* about 3 min of GPU in all for 8 processes, each well under 1 min per lock
      hold, with a busy check before each chunk (≤ 5 min).

**GPU rules.** Run under `_wip/gpu.lock` (the tools take it), behind the quiet gate, under
memcap, and only when the ehrenfest busy check says FREE (silent mode otherwise). Report
TFLOPS-eq first, with time alongside.

### 4.4 Stage 3 — L7 and the K-segmentation recipe (CPU); the lemma texts L5–L7

**Stage 3 delivers** L7 in PROOFS with its Lean proof `kseg_compose`, `cert_kseg_composition`,
`oracle.contract_segments`, the K-segmentation caller recipe (§2.7, with L7's caveats), and the
DESIGN §7 rows below.

**Implemented 2026-10-07** (Status; design questions §5.7): DESIGN §3.8 "Segmented
accumulation" and §7, PROOFS L7 (L7a–L7d), the Lean below, and the code deliverables. The text
of this section is the design it was built from.

**Order and size** (IR12, IR28, IR5). Stage 3 starts after the M-a summary exists, and its §7
rows cite M-a. It is re-sized from S to M for the Lean prerequisite below.

**Lean prerequisite: L3c** (IR5; decided by the independent reviewer). proofs/NOTES.md maps only
L3a and L3b. There is no IEEE model; rounding enters as a hypothesis. L7 needs both L3c and
γ_{S−1}. So before L7's `kseg_compose`, Lean gets:
- L3c's ν form;
- the ELEMENTWISE and ABSOLUTE compositions;
- the final-rounding step in the normal range, under the same style of hypothesis as L3b's
  existing |D − z·2^{−σ−τ}| ≤ u|D|;
- an FP recursive-summation lemma γ_k: the per-step hypothesis |fl(a + b) − (a + b)| ≤ u|a + b|,
  and induction to γ_{S−1}.

All of it lives in proofs/Proofs/Error.lean or a new Summation.lean, with no new axioms, and is
audited by `tools/check_proofs.sh`. proofs/NOTES.md maps L3c and L7 when they land.
- The stage-3 code deliverables do not wait on the Lean: `oracle.contract_segments`,
  `cert_kseg_composition` and the recipe text.
- The Lean must be complete before C2 is declared done (§4.11).
- L5d and L6 Lean (stages 4 and 5, if built) build on this L3c.

**The other lemmas are proved where they are first relied on** (R2, §5.2):
- L5, its certificates, `GuardSound` and Lean in stage 4;
- L6, its certificates and Lean in stage 5, which is deferred (§4.6, IR26).

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
  - **Several partners** (§4.5, IR6). With X pre-scaled once and partners Y_1..Y_P, the guard
    is per pair: G_p = gain_min(X) + gain_min(Y_p), and the reported G = min_p G_p. L5c is
    pairwise, so G ≥ 3 makes it hold for every partner. L5d's transform with the min G is valid
    for every partner, because the transform is monotone in G.
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

**L6 (a-posteriori certificate)** **[stage 5, deferred (IR26); cert: cert_apost, cert_stats_constants,
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
  the drafted `LayoutPrefix`: equal sig/flag offsets across N, res(N′) inside res(N_max)), I2,
  and I3 (packing more roles never increases ws(N, g); IR8).
- Stage 1: `cert_rho_recipe`. The §4.2 FP evaluation gives ρ̂ ≥ ρ in exact rationals, on random
  rows and on rows with |D| just above U·n. `host_precision` checks the shipped
  `rho_from_output` helper against it (IR7).
- Stage 3: `cert_kseg_composition`. L7 against exact rationals, with rows from every M5 family
  plus constant rows; S ∈ {1, 2, 3, 4}; N ∈ {3, 4, 5, 8, 14, 20}; RN and TRUNC; per-segment ν;
  every metric; plus the ν counterexample (a = (1, ε, …)).
- Stage 4:
  - `cert_kw_rule`: the §4.5 integer rule against an independent brute force on random Δ
    vectors (ties, even and odd counts, deadband edges, floor of negative values, both clamps,
    the Q9 index order with repeated and one-sided labels), including the multi-partner union
    extrema and clamps, and the rejection of partners whose shared-K label set differs (IR6).
  - `cert_kw_fp64_exact`: clamped weights make every scaled entry exact in FP64, on edge
    exponents including subnormals. R-num re-derived the four clamp bounds.
  - `cert_kw_guard`: L5c on random and adversarial rows in exact rationals:
    ‖ã_i‖‖b̃_j‖ < 4s²·2^{−G}·‖a_i‖‖b_j‖ for every pair at the reported G, and the strict
    inequality whenever G ≥ 3. It also checks L5d's FP evaluation: cond′ and norm_product′ at
    least, and t′ at most, the exact values. It does not search for a margin-2 witness; L5c
    states analytically why margin 2 is not enough for the proof.
- Stage 5 (deferred, IR26; built only on a trigger):
  - `cert_stats_constants`: X and B̄ are exact ceilings; their widths fit L32(N) for RN; the
    TRUNC "uncertified" flag; r ≤ 155; k ≤ 2^40.
  - `cert_stats_bin`: the K5 bin rule (bitlen(X) − bitlen(Y), clamped at 0, then one exact
    shift-compare adjust) equals min{r ≥ 0 : Y·2^r ≥ X}, at the edges Y = 1, Y = X, X a power
    of two, and Y = X ± 1, for every (N, k) edge.
  - `cert_apost`: L6 against `emulate_element` on random, constant, adversarial and weighted
    rows (ρ ≤ 2^r, error ≤ certificate for in-range outputs, out-of-range ones flagged), with
    counterexamples for dropping kδ² or using ⌊√k⌋.
- Stage 6: `cert_kw_shift_range` and reruns; `cert_workspace` and `cert_plan_bytes` for the
  weighted vec sizes of the q map (§4.7, IR21).

**Oracle** (spec/oracle.py), by stage:
- Stage 2: `kweights(A, B, plan, params)`, returning (w, report) in exact integers, with the
  report's G (§4.3 M-a); and its multi-partner mode (one X, partners Y_1..Y_P, G = min_p G_p;
  P = 1 is the single-partner case), which rejects partners whose shared-K label set differs
  (§4.5, IR6).
- Stage 3: `contract_segments(A, B, C, N, rounding, bounds)`, a literal composition of
  `contract` with β = 1. (As implemented: each band dispatched as the library dispatches
  it, DIRECT for a K-empty band of k_s = 1; §5.7 S3-2.)
- Stage 5 (deferred): `apost_bin(z, sigma, tau, plan)`, and `stats(view, ...)`, the integer
  fields of `imd_stats_t`.
- Self-tests cover the CRT assertion on every fixture, invariance of w under joint K
  permutation, and determinism.

**DESIGN §7 rows** (immerdink data only). They are written after the M-a summary and cite it
(IR12): DESIGN §7 must not assert dominance on synthetic small-m data before the realistic-size
study.
- "Accurate mode": still rejected. By an estimate from the §3.8 bound, confirmed by the data of
  §2.5–§2.6 (not a theorem), partner-aware σ/τ shifts realize about G/7.8 moduli out of a G-bit
  headroom. They saved ≤ 0.3 moduli on dense data and about 3 on separable anti-correlated data,
  need one more operand pass (losing KF/KL group-1 fusion) and break D1. K-equilibration turns
  the same headroom into about G/3.9 moduli, with no pass inside the library (§2.8). The row
  adds M-a's Tier L and real-operand figures. The envelope rule (Appendix B) is recorded as the
  proven partner-aware option.
  - **If any stage-3 text is written before M-a reports**, this row is worded "provisional until
    M-a". It then keeps only the reasons that hold regardless of M-a: it breaks D1 for σ, needs
    the extra operand pass, loses KF/KL group-1 fusion, and gets about G/7.8 against G/3.9
    moduli per bit, tagged as an estimate.
- "More bits per modulus via larger moduli (e.g. Karatsuba-split residues)": rejected, more
  INT8 GEMMs per bit on sm_89 [I] (IR15). arXiv:2603.10634 is cited only if the builder can
  verify its title and claim; otherwise the row carries no citation.
- New rows: N-range plan (A2), multi-N and RO packs (A3, A4), device-chosen N (B4), slices (C1,
  parked, with §3.3's triggers), K5 statistics and the device certificate (B3, deferred with
  triggers T1–T3, §4.6). Every deferral names its trigger, here or in the ROADMAP (IR29).

### 4.5 Stage 4 — `imd_kweights` and caller-side equilibration (C2, step 1)

**Gate** (pre-registered by the independent reviewer, IR27; applied mechanically in round 2,
before M-a's results are known). Stage 4 passes iff all of the following hold:
- **(i)** On at least one real-operand contraction R1–R6 (≥ 2048³ class, §4.3):
  - the guard = 1;
  - **and** either **(b)**: the predicted net TFLOPS-eq gain is ≥ 5 % at equal target, in either
    metric. The elementwise targets are {1e−8, 1e−10}. The normwise targets, {1e−10, 1e−12},
    are relative to the **plain** norms through L5d's t′. Every cost is included: the helper,
    amortized over a 16-iteration run for invariant operands and per call for operands rebuilt
    every iteration; unfused pre-scaling at 16 B per element per changed operand per call; and
    the memory of scaled copies;
  - **or (c)**: a target ≤ 1e−8 elementwise that fast mode cannot reach at N ≤ 20 becomes
    reachable at N ≤ 20.
- **(ii)** No Tier-L or real fixture shows realized regressions > 2× among guard-accepted runs.
- **(iii)** The FP64-exact clamp is never violated.

If the gate fails, stage 4 is closed with the revival trigger "a real-operand family meeting
(i)", and `oracle.kweights` and M-a remain as the C2 deliverable.

**Applied, 2026-10-07: the gate FAILS, and stage 4 is closed** with the revival trigger "a
real-operand family meeting (i)" (computed by `bench/accuracy/c2_summary.py`; immerdink-eval
`findings/c1-evidence.md`; §5.6 S2-3).
- **(i) not met.** No real R1–R6 contraction of the ≥ 2048³ class has guard = 1; G ≤ 0 on
  every one. R1, R2, R6 and R7 contract one DF factor with itself or its transpose, so Δ ≡ 0
  and w ≡ 0 by construction. R2b, R3, R4 and R5 have nearly aligned K profiles (|w| ≤ 4,
  G ∈ [−4, 0]). With the caller's costs, K-eq reaches 2–47 % lower TFLOPS-eq than fast mode
  at the normwise gate targets (+2 % to +89 % time; N = 9–12) [M, model-only, outside
  hold-out] and 1–36 % lower at the ew8 ones (+1 % to +57 % time; N = 13–16) [M,
  model-only, outside hold-out at N = 13 and 15], and (c) cannot apply: point-group zeros (ρ up to 1e20) leave max-elementwise
  1e−8 unreachable for every scheme at N ≤ 20. Water R1 and R5 (2mnk = 7.2e8) are below the
  class and not gate-admissible.
- **(ii) not met as pre-registered.** Every guard-accepted Tier-L fixture has sampled outputs
  above 2× fast's error at the same N: 2–19 per fixture at d*, max ratio 247 (opp_nonsep
  2048 × 16384). At d* they all occur at N = 4–5, on outputs whose fast-mode error is itself
  of order 1 (high-ρ outputs; fast mode as a whole already meets normwise 1e−8 at N = 4 on
  decay_opp). At d = 0 they also occur at N = 6, 12 and 14 (TRUNC). The noise-robust
  measures (2) and (3) below flag nothing; they are diagnostics this round, and the outcome
  does not depend on (ii), since (i) fails.
- **(iii) met:** no FP64-clamp violation in any run.
- **When:** at the end of round 1, although IR28 places the gate decisions in round 2. That
  is admissible because the failing part of (i), guard = 1 on a real R1–R6 contraction, is a
  CPU-only quantity that no round-2 measurement can change. M-e needs a stage-4 build and is
  not run.

**Re-registered for any revival** (decided by the independent reviewer before any new
evidence, §5.6 S2-1, S2-2, S2-12, S2-14, S2-16; binding on a revival):
- **(ii)** passes iff, on every guard-accepted Tier-L or real fixture, both hold:
  - (2): no sampled output whose median ratio (K-eq error / fast error at the same N, over
    the N where fast's error exceeds 4u|x|, at least three of them) exceeds 2;
  - (3): no N at which fast's sampled max error is below 1 and the scheme's exceeds 2× fast's,
    in either metric.

  The per-output count (1) above is still reported, as a diagnostic: at equal accuracy it
  fires on independent rounding noise (for two half-normal errors whose scales differ 2×,
  about 16 % of outputs exceed 2×).
  - (ii) is judged on the scheme the revival would ship (K-eq at d*). If the revived caller
    recipe allows reusing a w on changed data (stale weights) instead of re-running the
    helper per changed operand, as the recipe below prescribes, the stale-weight runs count
    as gate-(ii) runs too (S2-14).
- **Gate (b)'s memory of scaled copies** is reported for the recipe actually proposed (which
  operands are held pre-scaled, per caller-cost scenario), with the upper estimate
  8(mk + nk) bytes (both operands pre-scaled and kept beside the plain ones) alongside
  (S2-16).
- **Elementwise criteria (b) and (c)** are judged on the outputs with ρ_ij ≤ 1e8 (exact x,
  plain norms): DESIGN §6f.1's "outputs that must meet the target". The all-outputs reading is
  reported alongside. On this round's data the outcome is the same under both readings.
- **Q12** is re-tested on a real batched contraction that is not a self-contraction, before
  the C API is fixed (R7 cannot discriminate: Δ ≡ 0 in every batch).

**Cost figures at the gate** (IR23). The gate uses bandwidth-based [E] figures for the
pre-scaling (16 B per element) and the helper (1–2 operand reads), at the DRAM bandwidth implied
by the shipped calibration, and states them next to the gate. M-e measures them after any
stage-4 build. If M-e's net gain on the gate's own case falls below the 5 % bar, stage 4 is not
merged to dev unless criterion (c) holds there, and stage 6 is not started.

**The idea.** The caller stores pre-scaled operands, ã = a·2^w along K and b̃ = b·2^{−w}, and
contracts them with ordinary plans and packs. The library's numerics do not change; C is
bit-exact against `oracle.contract(ã, b̃, ...)`. immerdink supplies an exact, deterministic,
capture-safe helper that picks w, reports the L5c guard and the gain G, and a host function
that turns plain statistics into the weighted call's request (L5d).

**Index order** (Q9, both reviewers' rules merged). w is indexed over the **shared K labels**:
the plan's contracted labels that both A and B carry.
- Labels are ordered by their first occurrence in X's descriptor mode order, first label
  fastest (column-major). X is the operand scaled by 2^w: A in a single plan, and the
  pre-scaled operand of the multi-partner rule below (IR6). A repeated (diagonal) label counts
  once, at its first occurrence.
- The length k_s is the product of the shared labels' extents. k_s = k when every K label is
  shared. An extent-1 label contributes a factor 1 and changes nothing.
- A K label carried by only one operand (DESIGN §5 step 6: stride 0 in the other) is not
  indexed: w is constant along it, i.e. it gets w ≡ 0. Scaling it on one side would need the
  other operand materialized along a label it does not carry.
- The order does not depend on the partner or on C's strides, unlike the plan's canonical k̄
  (§5 step 8). Stage 6 maps it to k̄ through the q map (§4.7, IR21).

**Several partners** (IR6; decided by the independent reviewer). In DF-CCSD the invariant Bov is
contracted over Q with both Bov and Bvo (ehrenfest ccsd.py, read-only: ldQ,kcQ->ldkc and
aiQ,kcQ->aikc). The rule below therefore defines w on one operand X, pre-scaled once by 2^w along
the shared K labels. Its partners Y_1..Y_P each receive 2^{−w}, whatever their role in each plan;
defining w on X keeps role swaps across plans well defined.
- Every partner must carry exactly the same shared-K label set, in X's descriptor mode order (the
  Q9 order taken from X). `oracle.kweights` rejects partners whose shared-K set differs.
- Partners over different K labels (e.g. t2 over cd in PPL and over ld in ring) need separate
  weight vectors and separate scaled copies: the memory risk below.
- The single-partner case is P = 1, with X = A and Y_1 = B.
- Share groups are the partners of a shared packed X: one N (a pack per N) and one w. There is
  no further interaction.

**Rule** (integers only; one vector shared by every batch λ, Q12):
1. **Exponent extrema.** For each shared index h: c_X(h) and m_X(h) are the max and min of
   ilogb|x| over the finite nonzero entries of X at h (all λ, its free indices and one-sided K
   labels). c_Y(h) and m_Y(h) are the max and min over the finite nonzero entries of **every**
   partner at h (all λ, free indices and one-sided K labels): the union extrema. An index with
   no such entry on one side is undefined. With P = 1 these are c_A, m_A, c_B, m_B.
2. **Δ and median.** Δ_h = c_Y(h) − c_X(h) where both are defined; Δ ∈ [−2097, 2097]. med = the
   lower median of the defined Δ, from a one-CTA histogram.
3. **Raw weight.** w0_h = 0 if Δ_h is undefined or |Δ_h − med| ≤ deadband; otherwise
   ⌊(Δ_h − med)/2⌋.
4. **Clamp, always.** w_h = w0_h clamped to [−max_abs, max_abs] (max_abs ≤ 127) **and** to the
   interval where scaling both operands is exact in FP64. The FP64-exact clamp is
   unconditional (R-num must-fix 1); there is no flag to turn it off. A side with no finite
   nonzero entry at h imposes no bound.
   - w > 0 (X scaled up, partners down): w ≤ 1023 − c_X(h) (no overflow) and
     w ≤ m_Y(h) + 1022 (no partner entry drops below the normal range).
   - w < 0 (X down, partners up): w ≥ −1022 − m_X(h) and w ≥ c_Y(h) − 1023.
   - The union extrema keep the clamp exact for every partner.

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
   guard = 1 iff G ≥ 3 (L5c). Rows and columns are those of the caller's A and B, independent
   of lowering's operand swap (§5.6 S2-5). If A has no nonzero finite row or B no nonzero finite
   column, G is undefined. Then guard = 1 vacuously, since every product is zero or flagged,
   and gain = INT32_MIN; the plain request applies unchanged.
   - *Several partners:* the guard is per pair, G_p = gain_min(X) + gain_min(Y_p), each computed
     at pair p's plan (N, k, rounding). The reported G = min_p G_p, and guard = 1 iff G ≥ 3. L5c
     is pairwise, so it then holds for every partner. `imd_err_request_kweighted` with the min G
     is valid for every partner, because the transform is monotone in G.
   - *w ≡ 0* (`num_weighted` = 0, e.g. aligned profiles, or Δ ≡ 0 in a self-contraction): then
     G = 0, so guard = 0, with no special case (decided, §5.6 S2-4): guard = 1 stays the one
     claim of L5c. The weighted call IS the plain call, so the caller skips the pre-scaling and
     uses the plain request with its own ℓ1 ratios. It does not call
     `imd_err_request_kweighted` with gain 0, which would tighten cond 4× for nothing. The
     same goes into the stage-4 header.

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
  int32_t  flags;                /* IMD_KW_GUARD = 1; IMD_KW_ZERO_ON_FAIL = 2 (only with
                                    IMD_KW_GUARD, else INVALID_VALUE: on guard failure w := 0);
                                    other bits: INVALID_VALUE. The FP64-exact clamp is always
                                    applied (no flag).                                        */
  int64_t  reserved;             /* 0 */
} imd_kweights_params_t;
typedef struct {                 /* 32 B, written by the call into caller device memory      */
  int32_t guard;                 /* 1: G >= 3, so L5c holds for every nonzero row/column pair;
                                    0: not; -1: not computed                                  */
  int32_t gain;                  /* G = gain_row_min + gain_col_min (L5c/d); INT32_MIN if not
                                    computed (never 0: G = 0 is a real value)                  */
  int32_t gain_row_min, gain_col_min;  /* min (sigma~ - sigma) over nonzero finite rows of A / columns of B,
                                    for the pair that sets G; A and B are the caller's operands
                                    as passed to that pair's plan, independent of lowering's
                                    operand swap (the same for row- and column-major C);
                                    INT32_MIN whenever gain is, even if one side is defined
                                    (§5.6 S2-5)                                                */
  int32_t median;                /* med, INT32_MIN if no index has both exponents            */
  int32_t num_weighted, max_abs_w, clamped;  /* of the WRITTEN w: #{w != 0}, max |w|,
                                    #{w != w0}; ZERO_ON_FAIL with guard = 0: 0, 0, #{w0 != 0},
                                    while guard, gain, gain_*_min and median keep their
                                    computed values (§5.6 S2-5)                                */
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

- **Single-partner shape.** The signature above is the P = 1 case. The multi-partner C API shape
  (partners as an array of plans and B pointers, one report with the min G) is finalized at the
  stage-4 gate (IR6); the rule and `oracle.kweights` already define it.
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
- **Capture, end to end** (IR13). The helper may run inside a capture. The guard read (a D2H),
  the weighted request (`imd_err_request_kweighted`) and the choice of N happen once per
  magnitude profile, outside capture, by the caller.
  - `IMD_KW_ZERO_ON_FAIL` (valid only with `IMD_KW_GUARD`): on guard failure the helper writes
    w ≡ 0, and sets the report's `clamped` and `num_weighted` to match. The report then
    describes the written w (`clamped` = #{w0 ≠ 0}, `num_weighted` = 0, `max_abs_w` = 0), while
    `guard`, `gain`, `gain_row_min`, `gain_col_min` and `median` keep their computed values: the
    caller sees guard = 0 and uses the plain request (confirmed, §5.6 S2-5). `oracle.kweights`'
    detail fields `clamped_max_abs` and `clamped_fp64`, which are not in the C struct, describe
    the clamps of the pre-zeroing w.
  - **The fully captured pattern:** the helper with ZERO_ON_FAIL → the caller's scaling kernel
    applies w → a contract at an N chosen beforehand from the **plain** request with
    `l1_ratio` = 0 (ν = √k). That request is rigorous in both outcomes: by L5c when the guard
    holds, and on plain operands when w ≡ 0. It gives up only the L5d tightening inside captured
    iterations.
- **Cost** [E]: one read of A and B without the guard, two with it. At 4096³ that is about 0.3 /
  0.6 ms, about 6 / 13 % of an N = 14 call. It is paid once per magnitude profile, or per call
  if it is re-run on operands that change every iteration (§2.6). The gate states
  bandwidth-based figures, and M-e measures them (IR23).
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
   with `min_moduli` / `PrecisionPlan` as usual. If `num_weighted` = 0, skip steps 2–3: the
   plain operands and request apply unchanged.
4. Contract and pack the scaled tensors as usual.

The caller owns consistency: every partner of a pre-scaled packed operand must carry the same w,
the one the multi-partner rule chooses over all of them (IR6). A mismatch is a silently wrong
product, documented as a caller rule. The reported G is valid for the pairs the helper ran on.
For new data it is a prediction, unless the helper is re-run.

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
  certificates of §4.4. `oracle.kweights` already exists from stage 2. L5d's Lean builds on
  stage 3's L3c (IR5).
- **TLA+ run limits** (IR25; binding when stages 4 and 5 are built). A previous TLC run
  OOM-crashed the machine, so:
  - every run under `tools/memcap.sh`, ≤ 2 workers, -Xmx ≤ 8 GB, ≤ 30 min per configuration;
  - first a 1-worker smoke configuration at the smallest constants, with TLC's state count and
    memory recorded;
  - toy constants then sized to fit: CTAs ≤ 2, threads per CTA ≤ 2, calls ≤ 2, toy k ≤ 3, toy
    widths as in OzakiScaling;
  - the chosen constants, the distinct-state count, the run time and the peak RSS recorded in
    each .cfg header.

  `GuardSound`'s toy-exhaustive enumeration is bounded the same way, and the bounds are stated in
  PROOFS next to L5c.

**Tests.**
- `host_kweights`: workspace bytes and k_s against params, the Q9 order (repeated labels,
  one-sided K labels, extent-1 labels), and validation (struct_size, unknown flags,
  ZERO_ON_FAIL without GUARD, deadband < 0, max_abs ∉ 1..127, reserved ≠ 0, a nonzero tail
  beyond the known size).
- `host_accuracy`: `imd_err_request_kweighted` against params' exact transform for every metric,
  with ℓ1 ratios reset, and its INVALID_VALUE cases.
- GPU `kweights_helper`: w and the report bit-equal to the oracle on the M5 families, random Δ,
  ties, all-zero K indices, non-finite entries, all-equal Δ, subnormals at the clamp edges, k = 2,
  batches; with ZERO_ON_FAIL, w ≡ 0 on a failing guard. Also determinism (repeats, two
  streams), capture/replay, and compute-sanitizer memcheck, racecheck and initcheck. Once the
  multi-partner C API is fixed at the gate, the same against `oracle.kweights`' multi-partner
  mode.
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
- It confirms the stage-4 gate case (IR23, IR27). Below the 5 % bar on that case, stage 4 is not
  merged to dev unless criterion (c) holds there, and stage 6 is not started.

**Risks.**
- The value depends on the data: none on dense data.
- Weights that go stale on random-structure data cost up to 2 moduli. The guard detects this only
  when it is re-run on the current pair.
- Pre-scaling a tensor that is also contracted over other indices (e.g. t2 in PPL over cd and in
  ring over ld) needs a separate weight vector and a scaled copy per K-label set, which costs
  memory (IR6). Stage 6 removes the copies.

### 4.6 Stage 5 — K5 statistics and the a-posteriori certificate (C1(b) on the device)

**Gate: a consumer with elementwise targets (Q16). Not met; stage 5 is formally deferred**
(IR26; decided by the independent reviewer on read-only evidence):
- ehrenfest FOUNDATIONS.md D8 (lines 756-788): precision is user-directed; accuracy and stability
  analysis is out of scope; auto-choosing precision is a non-goal; the built-in policy is
  fp32 → fp64 on a residual threshold.
- ehrenfest docs/design/hardwareAcceleration.md:22: the Ozaki choice uses a 2× speed threshold,
  not an accuracy target.
- BENZENE_5Z_READINESS.md: the Ozaki work has been frozen since 2026-09-06.
- ccsd.py issues no accuracy requests.

So no consumer has elementwise targets. The stage-1 helper is rigorous and is needed only once
per phase decision, so its ~9 % cost amortizes away. Stage 5 would put a hook into K5, which runs
at 82–96 % of its bound (ROADMAP A4), for no current user, and would add an ABI, a TLA+ module,
Lean L6 and three certificates. Deferring with explicit triggers decides the gate on evidence,
and nothing that regresses ships.

**Triggers:**
- **T1:** a consumer issues ELEMENTWISE accuracy requests per phase, and the `rho_from_output`
  recipe's measured cost (the norm pass plus the pass over C) is ≥ 3 % of that consumer's run;
- **T2:** a consumer needs a rigorous per-call elementwise certificate on every call (verified or
  validated numerics);
- **T3:** Eric asks for the device certificate, e.g. for the paper track.

On a trigger, stage 5 is built as designed below (R3, R7, Q17–Q21), with one correction recorded
now: **the cost bar cannot use `ncu`.** GPU performance counters are admin-only on this machine,
and that must not change. The bar is restated as compiler resource usage (`-Xptxas -v` /
`cuobjdump --dump-resource-usage`), occupancy from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and A/B timing.

*Correction (close-out review, 2026-10-07; the binding text above is kept as written).* The
premise is wrong: GPU performance counters are open to all users on this machine, as HANDOFF
("Machine notes") and the ROADMAP say. Checked read-only: `RmProfilingAdminOnly: 0` in
`/proc/driver/nvidia/params`, and `/etc/modprobe.d/nvidia-profiling.conf` sets
`NVreg_RestrictProfilingToAdminUsers=0`. The bar's content stands as restated above: A/B
timing at stock clocks under the GPU lock, the register count from the compiler (`-Xptxas -v`
/ `cuobjdump --dump-resource-usage`) and occupancy from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The reason is that the bar is a time bar and
these measure it directly. `ncu`'s serialized replay does not reproduce production timing
(NOTES-K5-bimodal §4: K5 launches took 0.445–0.608 ms under replay, against 0.44 / 0.85 ms in
production). So `ncu` is not part of the bar, but it is not forbidden. As a diagnostic it runs
only with `--clock-control none` under the GPU lock, as NOTES-K5-bimodal ran it, after a FREE
busy check and in holds of at most 5 minutes. Never with the default: `ncu` 2026.3.1 in the
immerdink env defaults to `--clock-control boost`, which locks the clocks (immerdink-eval
`results/c1/closeout/review/ncu_clock_control.txt`). Nothing on the machine changes, and
nothing needs sudo. *(Revised after the second review. The first version of this note, at
c345582, kept `ncu` out for two wrong reasons: a default of `--clock-control base`, and its
replay holding the shared GPU, which A/B timing does too.)* The other places that state the
premise (the stage-5 tests below, Q20's IR26 amendment, IR26 in §5.3, Appendix A item 52)
point here.

**C1's definition of done does not need stage 5.** C1(b), auto-N from the M5 elementwise rule, is delivered
host-side by stage 1: `min_moduli`/rank, `PrecisionPlan`, and the rigorous `rho_from_output`
helper, whose per-output ρ̂ also yields an a-posteriori bound for that call.

The design below is kept for a trigger.

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
  template variant and the optional calibration key `k5.stats_op` (Q20). Registers come from
  the compiler (`-Xptxas -v` / `cuobjdump --dump-resource-usage`) and occupancy from
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, never from `ncu` (IR26).

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
  - The cost bar (A/B timing), with the register count from the compiler and the occupancy from
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, for `stats` NULL in every variant and at
    every N. No `ncu`: GPU performance counters are admin-only on this machine (IR26).
    *Corrected 2026-10-07: the counters are open to all users. `ncu` stays out of the bar
    because the bar measures time and `ncu`'s replay does not reproduce it; as a diagnostic it
    runs only with `--clock-control none`, under the GPU lock (§4.6, "Correction").*
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

**Gate** (pre-registered by the independent reviewer, IR27). Stage 6 starts iff:
- stage 4 has shipped;
- M-e confirms the stage-4 gate case (net ≥ 5 %, or (c));
- and a real fixture shows that library-internal weights remove a cost stage 4 cannot avoid:
  ≥ 5 % from pre-scaling or scaled copies on a recurring contraction.

Otherwise stage 6 is deferred with that trigger. The layout follows the decision below.

**Applied, 2026-10-07: deferred.** Stage 4 has not shipped: its gate failed and it is closed
(§4.5). Trigger: a revived stage 4 that ships and passes M-e on its gate case, plus a real
fixture on which library-internal weights remove ≥ 5 % of pre-scaling or scaled-copy cost on a
recurring contraction (the three conditions above).

**Decided: from caller-order weights to the kernels' k̄** (R-api should-fix 7; IR21, decided by
the independent reviewer: candidate (i), in its map form). Stage 6 stays gated; this only
unblocks its design.
- w arrives in the Q9 order (shared K labels, X's mode order). The scaling and pack kernels
  index K in the plan's canonical k̄, which depends on the partner and on C's strides (§5
  step 8), and they need w per k̄ position, constant along one-sided labels.
- **The q map.** A plan created as weighted gets one more K scatter vector in the vec region:
  q(k̄) = Σ_{shared K labels l} coord_l(k̄)·qstride_l, the Q9 index of each k̄ position.
  - It is mixed-radix over the shared labels in X's mode order, with qstride = 0 for one-sided
    labels, so w ≡ 0 along them.
  - It uses 8-byte entries like the other vectors, 8k bytes in all.
  - K0 generates it under GENERATE, and it is resident under RESIDENT, exactly like the existing
    A/B K offsets.
- **Use.** The scaling and pack kernels gather w[q(k̄)] from the caller's Q9-ordered vector
  (in-line), or read the pack's kw region (packed). `imd_pack_ex` writes kw in the pack's
  adopted K order through the same map. There is no per-call conversion kernel, and
  `IMD_REGION_COUNT` does not change.
- **Unweighted plans keep byte-identical layouts**, because the map exists only for plans
  created weighted. `imd_options_t` is frozen at 24 B, so that needs a new `imd_plan_create_ex`
  with a `struct_size`-versioned descriptor (flag `IMD_PLAN_KWEIGHTED`, zero-tail rule, §4.9).
- **Rejected:**
  - (ii) a new region. It changes `IMD_REGION_COUNT`: `imd_workspace_layout` returns that many
    regions (include/immerdink/immerdink_host.h:149-151), and tests/test_host_c.c:59 checks it.
    This is the same problem Appendix A #18 raises for P3's regions.
  - (iii) a caller-supplied k̄ order. That is Q9 (b), rejected because the order depends on the
    partner and on C.
- *Why the map.* The Q9 index of a k̄ position is a linear function of the K-label coordinates.
  That is the same kind of object as the existing per-k operand offsets, so it reuses K0's
  generator and the RESIDENT/GENERATE machinery, and changes no region count.
- **Spec and params plan:** `cert_workspace` and `cert_plan_bytes` for the weighted vec sizes;
  IndexMap "the q map equals the Q9 index", with negatives `neg_kw_map_onesided` and
  `neg_kw_map_order`; Pipeline "kw read through the map is read-only".

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
  same size), in the Q9 order and gathered through the q map above. If stage 5 has not shipped
  by then, stage 6 introduces `imd_call_t` and `imd_contract_ex` with §4.6's layout, and a
  non-NULL `stats` returns NOT_SUPPORTED until stage 5 ships.
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
- The `kw` region holds k̂ bytes, 256-aligned, after res, in the pack's adopted K order, written
  through the q map (IR21).
- Unweighted packs keep their bytes, formats 1/2 and keys.
- Plans: `imd_plan_create_ex` with the `IMD_PLAN_KWEIGHTED` descriptor flag (IR21); unweighted
  plans keep byte-identical layouts.

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
- Pipeline: kw is a read-only input of the scaling and pack kernels, read through the q map;
  `neg_kw_helper_race` (the helper's output is read on another stream without an event).
- IndexMap (IR21): the q map equals the Q9 index; negatives `neg_kw_map_onesided` (a one-sided
  label with qstride ≠ 0) and `neg_kw_map_order` (the map follows k̄'s label order instead of
  X's).
- Certificates `cert_kw_shift_range`, `cert_pack_words` (weighted), `cert_error_estimate`
  (weighted rows), `cert_workspace`/`cert_pack_layout` for weighted packs, and
  `cert_workspace`/`cert_plan_bytes` for the weighted vec sizes (IR21).
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
| PackedOperand.tla | 0 | **none** (R1, IR20): ordering is already modelled, the stale handle is §6b.10 #3's documented limit, and the layout prefix is a params certificate. The optional ⟨version, key⟩ cells, a c3 after post with RAW post → c3, and `neg_repack_stale` (§4.1) are **not adopted**: built only on a range-plan revival (§3.2) or if Eric asks |
| DeviceReduce.tla (new) | 4, 5 (if built) | helper reductions (`neg_kw_no_zero`, `neg_kw_rmw`, `neg_kw_median_early`); statistics (`pos_stats_two_calls`, `neg_stats_flush_before_barrier`, `neg_stats_global_rmw`, `neg_stats_lib_zeroes`). Under the IR25 run limits (§4.5) |
| HostAPI.tla | 5 (deferred) | `pos_stats_shared` |
| IndexMap.tla | 5 (deferred), 6 | stats argument checks (`neg_stats_alias`) (5); the q map equals the Q9 index (+ `neg_kw_map_onesided`, `neg_kw_map_order`) (6, IR21) |
| OzakiScaling.tla | 4, 5, 6 | `GuardSound`, toy-exhaustive, bounded per IR25 (4); `CertSound` (+ `neg_cert_no_kdelta2`, `neg_cert_floor_kappa`) (5, deferred); weighted JointSpec (+ `neg_kw_one_side`, `neg_kw_same_sign`, `neg_kw_range`) (6) |
| PackedOperand.tla | 6 | `pos_kw_from_pack`, `neg_kw_partner_own`, `neg_kw_both_packed` |
| Pipeline.tla | 6 | kw read-only input, read through the q map; `neg_kw_helper_race` |
| OzakiCRT, CRTRealization, K4Align | — | no change: integer back end independent of how the scaling is chosen (OzakiCRT `CRTExact`) |
| PROOFS / Lean | 3, 4, 5 | L3c and the FP summation lemma γ_k, then L7 `kseg_compose` (3, IR5); L5 `kweight_cancel`, `kweight_guard` (4); L6 `apost_cert` (5, deferred) |
| params.py | 0, 1, 3, 4, 5, 6 | `cert_footprint_monotone` with I3 (0); `cert_rho_recipe` (1); `cert_kseg_composition` (3); `cert_kw_rule` (multi-partner), `cert_kw_fp64_exact`, `cert_kw_guard` (4); `cert_stats_constants`, `cert_stats_bin`, `cert_apost` (5, deferred); `cert_kw_shift_range`, `cert_workspace`/`cert_plan_bytes` for weighted vec sizes, and reruns (6) |
| oracle.py | 2, 3, 5, 6 | `kweights` with multi-partner mode (2); `contract_segments` (3); `apost_bin`, `stats` (5, deferred); the weighted quantizer (6) |

Every TLC/Apalache run goes through `tools/memcap.sh` with bounded parallelism, and stage 4/5
models follow IR25's limits (§4.5).

### 4.9 Versioning

- **Stages 0–3: no ABI change.** `IMD_HOST_API_VERSION` stays 13. The F1–F3 guarantees are
  documented "from API 13 on"; they describe the current formulas and change no meaning. They
  cover only the totals and the write extent (§4.1). F3's second half ("no byte beyond
  `pack_bytes`") is published only after `pack_write_extent` passes (IR17). The offset,
  device-byte and packing facts I1–I3 are certificates, not ABI, so a later format may change
  them. Stage 1's `PrecisionPlan` composition and `rho_from_output` are Python only (IR2, IR7).
- **Stages 4, 5 and 6** each take the next host API number in the order they ship (14 for the
  first). No number is reserved in advance, so the drafts' competing "API 14" claims are void.
  Stage 5 is deferred (IR26), so it takes a number only if a trigger fires.
- **New meanings only through new function names or validated words.** New functions
  (`imd_stats_cond`, `imd_err_request_kweighted`, `imd_pack_*_ex`, `imd_plan_create_ex`), or
  validated flag and reserved words that older libraries reject when nonzero (`imd_call_t`
  reserved fields, `IMD_KW_ZERO_ON_FAIL` in `imd_kweights_params_t.flags`). Never through an
  appended tail field that an older library ignores silently. `imd_err_request_t` does not
  change in any stage. Its `reserved` word stays free (R3).
- **Zero-tail rule for every new `struct_size`-versioned input** (`imd_call_t`,
  `imd_kweights_params_t`, `imd_pack_desc_t` v2, stage 6's plan descriptor for
  `imd_plan_create_ex`; R-api should-fix 4). The library reads
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
- **It rests on unvalidated predictions at N = 5, 6, 7, 9 and 13**, which no calibration or
  hold-out case covers (§1.2 item 8; M-c, IR4), and on unmeasured iteration counts under ramps,
  DIIS included (M-d1). Until M-c passes, DESIGN tags the schedule gains above, and every other
  gain at N ∉ {8, 10, 14, 16}, [M, model-only, outside hold-out].
- **The C2 evidence is synthetic and small,** one seed, and without the guard. Random-structure
  gains already vanished at m = n = 512; the structured gains must be confirmed at Tier L and on
  the real psi4 operands (stage 2; IR1, IR19) before stages 4 and 6.
- **C2's speedups are upper bounds.** The helper and the pre-scaling recur for operands rebuilt
  every iteration, and they cut 1.55× to about 1.2–1.44× at 4096³ unless the pre-scaling is fused
  (§2.6).
- **Weights can go stale,** and caller-side pre-scaling may need copies, one per K-label set of
  a multi-partner operand (stage 4, IR6).
- **The re-pack and pre-scaling consistency rules are caller obligations** that the library
  cannot check (stage 0, stage 4).
- **The statistics overhead** in K5 is unmeasured; stage 5 is deferred (IR26).
- **The real operands are stand-ins** [I]: MP2 amplitudes for converged t2, and bare integrals
  for t1-dressed ones (IR19).

### 4.11 Order of work and definition of done

Decided by the independent reviewer, 2026-10-07 (IR28, IR29). The order follows the
dependencies:
- the rebuild gates everything that calls the library;
- `oracle.kweights` gates `c2_gate.py`;
- M-a gates stage 3's §7 text and the C2 gates;
- M-c gates M-d2 and every claim at N ∉ {8, 10, 14, 16} (§1.2 item 8).

**Round 1 (CPU; the GPU only for the tools' existing CPU tests).**
- **Stage 0, in order** (§4.1): (a) the rebuild and the API-13 check; (b) the provenance copy and
  manifest; (c) the DESIGN §6g stage-0 text (F1, F2 and F3's first half, C-ws, C-pack, the
  re-pack rule); (d) `cert_footprint_monotone` (I1–I3) and `host_footprint_monotone`; (e)
  `PackSpec.with_moduli`, `PackedBuffer` with the `python_device` CPU part, and
  `python_overhead`.
- **Then two tracks in parallel.** They touch disjoint files (`python/immerdink/precision.py`
  against `spec/oracle.py`, `bench/accuracy` and immerdink-eval), so they can run concurrently,
  each single-threaded and busy-gated.
  - **Track A, stage 1** (§4.2): `precision.py` with the IR2 composition, pack-mode masks and
    BUDGET (IR8), joint share groups (IR24), the DIRECT sentinel (IR15), `rho_from_output` with
    `cert_rho_recipe` (IR7), and `host_precision`; `bench/accuracy/schedule.py`; the §1.3 text
    in the docs (IR3).
  - **Track B, stage-2 CPU evidence** (§4.3):
    1. `oracle.kweights` with the multi-partner mode and self-tests (IR6);
    2. the psi4 operand generator (IR19);
    3. `c2_gate.py`: engine self-test → pilot → Tier S (d*) → Tier L → real operands (IR1),
       plus the B6 estimate (IR11);
    4. M-b: the re-runs of the [M] figures and the share-group raise cost (IR9, IR15);
    5. M-d1: Jacobi, DIIS with and without reset, PPL-like, Davidson and block CG (IR3), and the
       ρ-miss rate (IR7).
- **Then stage 3, after the M-a summary** (§4.4): L3c in Lean → L7 + `kseg_compose` (IR5),
  `cert_kseg_composition`, `oracle.contract_segments`, the K-seg recipe, and the §7 rows citing
  M-a (IR12).

**Round 2 (GPU, under `_wip/gpu.lock`, behind the quiet gate, under memcap, FREE only).**
- The stage-0 GPU tests: `packed_repack_n`, `workspace_shared_n`, `graph_phases` with cold
  capture, the first-call measurement (IR10), and `pack_write_extent` with memcheck (IR17).
- M-c, N ∈ {4, 5, 6, 7, 9, 10, 12, 13} plus the anchors (IR4), **strictly before** M-d2.
  *It ran and failed (2026-10-07, §5.9).* Its recalibration is now the MC-3 feature as
  amended (§5.9 MC-3 and MC-3a), in order: the DESIGN §6c text from a per-phase account of
  the allowed records; at most two optional keys (the REPEAT credit; the KL narrow-pack time),
  each with `host_perf` tests; a fresh narrow-KL hold-out, committed and checked disjoint
  before the refit; a refit fitted against the phase marks; the pass checks (the v1.18
  hold-out on both sets, `mc.jsonl` with and without the repeat, a fresh full M-c run, the
  narrow-KL hold-out, and the per-phase conditions); then M-b's [M] figures and M-d2's bars
  re-derived if the calibration changes. Still strictly before M-d2.
- M-d2 against its pre-registered bands, after the MC-3 path closes (§5.9 "M-d2 ordering").
  *Its protocol and driver are pre-registered (re-registered after two reviews, now at
  35eeb46, §4.3, §5.10), and it has not run.*
  Once the ordering is met, the library is built at the registered commit and its sha256
  named in the findings; then it is eight `md2_time.py run` processes into the one
  normative records file, and one `judge` (§4.3 "Records, freeze and continuation").
- Then the mechanical gate decisions: stage 4 by its pre-registered rule (§4.5); stage 5
  deferred (a trigger check only, §4.6); stage 6 by its rule (§4.7).

If stage 4 passes, its build (L5 + Lean, `GuardSound`, `DeviceReduce`, the helper, M-e) is
round 3.

**Definition of done** (IR29). Every gated item ends as built, closed or deferred with a trigger,
never as open.
- **C1 is done when:**
  - stages 0 and 1 have shipped with their tests and certificates;
  - the round-2 stage-0 GPU tests pass;
  - M-c has passed after a recalibration per Q24 or with the MC-3 feature(s) (amended by the
    M-c review, §5.9 MC-3, and read as "feature(s)" at the second review, MC-3a; was "M-c has
    passed (or the model was recalibrated per Q24 and then passed)". The first run,
    2026-10-07, failed);
  - M-d1 and M-d2 are reported with their success criteria evaluated;
  - the ROADMAP C1 row shows the Q6 decision column and "stage 5 deferred (triggers T1–T3)".
- **C2 is done when:**
  - the M-a summary is published;
  - stage 3 has shipped, including L3c and L7 in Lean;
  - the stage-4 gate has been decided by the pre-registered rule: built and M-e measured if it
    passed, closed with its trigger if not;
  - the stage-6 gate has been decided: built if it passed, deferred with its trigger otherwise;
  - the ROADMAP C2 row is retitled per Q7.
- Every deferral names its trigger in DESIGN §7 or the ROADMAP.

*Checked at the close-out, 2026-10-07* (Status, "Status at the C1/C2 close-out"): **C2 is
done**; **C1 is not, and stays open** (it is neither closed nor deferred; its record continues
with MC-3a), since M-c has not passed (it failed, and MC-3a has not started) and M-d2
has not run. Every other condition of both, and the trigger rule, is met. This does not break
"never as open" above: that rule covers the gated items, stages 4, 5 and 6, which have all
ended (closed, deferred, deferred, each with its trigger). C1 is required work that is not
finished, not a gate.

---

## 5. Design questions and their resolutions

Both reviewers answered every question.
- **R-num**: numerics, proofs and spec.
- **R-api**: API, pipeline, performance and integration.

Each entry gives the options, the resolution, who decided it, and the rationale. "Both" means
the two reviewers chose the same option; amendments from either are adopted unless the entry
says otherwise. Where they disagreed, the entry says so and decides on the merits. §5.2 records
the review decisions that fall outside the questions. §5.3 records the independent design
reviewer's binding decisions (IR1–IR29, 2026-10-07). Where one amends a question's resolution,
the entry carries an **Amended (IRn)** line, and the amendment is binding.

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
**Amended (IR8).** With pack-mode masks, `footprints()` reports the maximum over the allowed
modes × levels, and per-level totals for the caller's joint budget.

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
**Amended (IR20).** The optional combined model is not adopted. It is built only on a range-plan
revival (§3.2) or if Eric asks.

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
**Amended (IR2, IR8, IR15, IR24, IR7, IR3).**
- The per-role pack cost is composed in Python: one rank with reuse = 1, re-keyed by
  contract_s + Σ pack_s(role, N)/reuse(role). Reuse is per role.
- `packs` takes allowed pack-mode masks, with a BUDGET status.
- `levels` may end with "DIRECT".
- Share groups are ranked jointly over every level ≥ the group floor.
- The ρ recipe ships as the tested `rho_from_output`.
- The docs state `monotone=False` for relaxed Krylov.

Still no ABI change (§4.2).
*Rationale.* It only composes `min_moduli` and rank, which already exist, and it keeps policy
with the caller (ehrenfest D8: precision is user-directed). planner.py's contract is "nothing
computed in Python" (DESIGN.md:4204). A single rank call over the levels matches §6f.3's
guidance, and it holds even where time is not linear in N.

**Q5. What accuracy basis does `PrecisionPlan` use by default?**
(a) BOUND;
(b) TYPICAL;
(c) BOUND for the final phase and TYPICAL for intermediate phases, as passed by the caller.
**Resolved: (a) as the default, both; (c) documented** as an explicit per-phase opt-in, with
fail_prob = P/C over the phase's calls (§4.2 step 7). M-d measures both.
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
**Amended (IR18): approved on Eric's behalf by the independent reviewer**; it is reported to
him, and he may overrule. The column text is "C1 = (a) per-call N (plan + pack per N) +
host-side (b) (PrecisionPlan, min_moduli/rank, rigorous ρ-from-output helper); (c) Ozaki-I
slices parked with triggers (§3.3); TLA+ for slices deferred to the slice trigger". It lands with
the DESIGN v1.19 merge.
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
**Amended (IR18): approved on Eric's behalf by the independent reviewer**; he may overrule. The
title is "K-equilibration (power-of-two K-weights, guarded, staged and gated) + K-segmentation
recipe; accurate mode re-rejected (§2.8)". It lands with the DESIGN v1.19 merge.
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
unstable orders. Stage 6's conversion to k̄ needs its own layout design first (§4.7).
**Amended (IR6, IR21).** The order is taken from X, the operand pre-scaled by 2^w (A in a single
plan). Every partner must carry the same shared-K label set. Stage 6 reaches k̄ through the q map
of plans created weighted (§4.7).

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
**Amended (IR6, IR13).**
- With several partners, the guard is per pair and G = min_p G_p.
- `IMD_KW_ZERO_ON_FAIL` makes a failing guard write w ≡ 0, for a capture-safe path. In that
  path the N comes from the plain request with ν = √k.

**Q11. Which weight rule?**
(a) Deadband-median balanced, with the deadband a parameter re-tuned in stage 2 at
m = n ≥ 128;
(b) plain balanced;
(c) one-sided, from the reused operand only.
**Resolved: (a), both.** The deadband is re-tuned in M-a at m = n ≥ 128 with at least 5 seeds.
The guard is the actual barrier against regressions.
**Amended (IR1).** The selection rule for d* is fixed in advance. It is chosen at Tier S,
m = n = 512, over 5 seeds: the d ∈ {0, 2, 3, 5} with the fewest guard rejections on the
random-structure fixtures, subject to the median gain on decay_opp and opp_nonsep staying within
0.25 moduli of the best d. Ties go to 3.
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
cases (orbital energies, DF factors) comes from the contracted labels, not the batch [I].
**Amended (IR16).** The escape clause could not fire without a batched fixture, so M-a adds two:
the real R7 (iaQ,ibQ->iab over all occupied orbitals, core included) and a synthetic batch.
The trigger is pre-registered. (b) is reconsidered only if per-λ weights save ≥ 1 modulus beyond
shared weights on a real fixture at the same guard status. Otherwise (a) stands, and the escape
clause is closed.

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
**Amended (IR27, IR1, IR23).** The rule is pre-registered and applied mechanically (§4.5 for
stage 4, §4.7 for stage 6). It covers the targets, the cost terms, the regression bar (ii) and
the clamp bar (iii). Only Tier L and real-operand runs of M-a are admissible. M-e must confirm
the gate case before stage 4 merges to dev, unless (c) holds there.

**Q14. Where does real-operand evidence come from?**
(a) The synthetic integral × amplitude family now, plus a one-off ehrenfest FP64 dump of a few
DF-CCSD operand pairs;
(b) synthetic only.
**Resolved: (a), both.** The dump needs Eric's go-ahead. It runs CPU-only, single-threaded,
under memcap, outside ehrenfest's test windows (busy check, silent mode). Raw dumps and records
stay in immerdink-eval `results/c1/` or scratch, never in the product repo.
*Rationale.* The whole C2 value case rests on the claim that real anti-correlated profiles are
separable along K. Only real operands can confirm it, and the dump is cheap and read-only.
**Amended (IR19): no ehrenfest dump is requested or performed in this cycle.**
- The real operands come from our own script, immerdink-eval `scripts/c1/psi4_operands.py`. It
  uses the psi4 of the `main` conda env read-only, CPU-only and single-threaded, and imports
  nothing from ehrenfest.
- Molecules: benzene/cc-pVDZ and cc-pVTZ, and water/cc-pV5Z. The contractions R1–R7 are taken
  read-only from the consumer's ccsd.py (§4.3).
- They count as real, consumer-confirmed operands for Q13. MP2 amplitudes stand in for t2, and
  bare integrals for t1-dressed ones [I].
- If psi4 cannot run, the stage-4/6 gates are formally deferred with the trigger "real operands
  available", and the Eric-approved dump is the named fallback.

*Why:* ehrenfest has no saved tensors (checked read-only), and its integrals come from a psi4
sidecar at runtime, so a dump would mean running ehrenfest code. Our own generator yields the
same bare DF blocks, with real anti-correlation and separability structure. It touches no
ehrenfest code, GPU or session, and it removes the only Eric-blocking dependency of the C2 gates.

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
**Amended (IR7, IR26).** "The recipe covers the need" rested on documentation alone. It now ships
as the tested `rho_from_output` (β = 0, α = 2^e), checked against `cert_rho_recipe`. Its
next-iteration miss rate is measured in M-d1. The gate was decided on evidence: **not met**, and
stage 5 is formally deferred with triggers T1–T3 (§4.6).

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
**Amended (IR26).** Not with `ncu`, since GPU counters are admin-only on this machine. Registers
come from `-Xptxas -v` / `cuobjdump --dump-resource-usage`, occupancy from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and time from A/B runs. This applies only if a
trigger revives stage 5. *Corrected 2026-10-07: GPU counters are open to all users on this
machine. The amendment's measurements stand because the bar measures time and they measure it
directly, while `ncu`'s replay does not reproduce production timing; `ncu` is a diagnostic
only, with `--clock-control none`, under the GPU lock (§4.6, "Correction").*

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
**Resolved: (a), both.** M-c, run on shapes outside the calibration set, is a prerequisite for
publishing any schedule gain that uses an N no case covers (§2.4).
*Rationale.* No calibration, hold-out or cache case covers N = 5, 6, 7, 9 or 13 (§1.2 item 8),
while the schedules lean on N = 6 and 9 predictions, and [KT26] reports INT8 throughput at small N
that depends on the data. Validate before refitting: adding calibration cases first would spend
GPU time, and possibly add model features, without knowing whether they are needed.
**Amended (IR4).**
- The previous text ("about 20 calls, N = 4–7"; "the hold-out covers only N = 8, 10, 14 and 16")
  was inaccurate.
- M-c now covers N ∈ {4, 5, 6, 7, 9, 10, 12, 13}, plus the anchors 8 and 14 on the same shapes,
  on 8 shapes checked absent from the calibration set. That is about 160 cases, ≤ 15 min of GPU
  time, judged per size class and reported per N.
- A miss leads to per-class residuals, then N-specific calibration cases with optional keys
  only.
- Until M-c passes, figures at N ∉ {8, 10, 14, 16} are [M, model-only, outside hold-out].

**Q25. What about DIIS at precision switches?**
(a) Measure in M-d (with and without a history reset) and document the result for callers;
(b) leave it to the caller.
**Resolved: (a), both.** The convergence part runs on the CPU with the bit-exact emulator
(M-d1, R-api must-fix 4). The GPU is used only for timing on realistic shapes (M-d2).
*Rationale.* ACCURACY §3 shows only that a contracting Jacobi iteration does not accumulate
emulation error. A DIIS history that mixes vectors computed at two precisions is untested, and
it decides whether ramps keep the iteration count, which is M-d's success criterion. Iteration
counts do not depend on GPU timing.
**Amended (IR3, IR7, IR23).**
- M-d1 adds report-only block Davidson and block CG cases. CG includes a relaxed (decreasing)
  schedule.
- M-d1 also adds the ρ-miss rate of lagged ELEMENTWISE targets.
- M-d1 has a budget of ≤ 30 CPU-min.
- M-d2 runs strictly after M-c. If M-c recalibrates, its bands are re-derived and pre-registered
  first.

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
    recorded as optional, should Eric want one. **Not adopted (IR20):** it is built only on a
    range-plan revival or if Eric asks.
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

### 5.3 Binding decisions of the independent design reviewer (2026-10-07)

A completeness critic reviewed the previous version and found 17 gaps; that version also left
8 items unresolved. All of them, plus four further items, were decided by the independent
reviewer, 2026-10-07. The decisions are binding. Each entry gives the decision in brief, where
the full text lives, and the reviewer's rationale. The machine was BUSY during the review, so
its cost figures are estimates.

**On the critic's gaps.**
- **IR1. M-a at realistic sizes** (gap 1: M-a could not answer the Q13 gate at m ≤ 512).
  - *Decision:* adopt the full-size, sampled-output protocol of §4.3, which replaces "sizes and
    seeds"; m = 16 is dropped.
  - *Rationale:* Q13 needs ≥ 2048³-class evidence, and §2.6 shows that random-structure gains
    vanish (and worst-pair losses grow to −13.6 bits) as m rises. Every guard quantity is O(mk),
    and the worst pair is separable, so full operands are affordable at m = 4096 if only the
    errors are sampled. A sample that includes the rows and columns setting min G is
    adversarial where it matters. The existing Python S loop would take minutes per operand at
    k = 16384 [E], hence the bit-identical vectorized engine with an exact fallback. SC25 and
    `acc_exact` need full GEMMs, so they stay at small m; they are already rejected (§2.5). The
    pilot and the pre-declared cut order keep the 90-minute bound enforceable without post-hoc
    choices.
- **IR2. Per-role pack cost** (gap 2: `PrecisionPlan`'s per-role packing cost against rank's
  single int32 reuse).
  - *Decision:* option (a). Compose in `precision.py` (§4.2): one rank call with reuse = 1,
    candidates re-keyed by contract_s + Σ pack_s(role, N)/reuse(role), feasibility unchanged,
    and per-role reuse with `Phase.packed`, `pack_bytes`, `reuse` and `assumed_reuse`.
    `host_precision` checks it against a brute force. Option (b), a per-role reuse in
    `imd_perf_rank_request_t`, is not adopted; it is revisited only if a hot-path consumer needs
    per-role reuse inside one C rank call.
  - *Rationale:* candidates already expose contract_s, pack_s and the packed bits, and time_s is
    the only reuse-dependent field (include/immerdink/immerdink_host.h:763-770; DESIGN.md
    3410-3425), so re-keying is exact. It keeps "stages 0–3: no ABI change". §6e.4's "computes
    nothing" covers planner.py only. The helper runs once per phase switch, and ehrenfest's
    hot-path rank calls are untouched.
- **IR3. Relaxed Krylov** (gap 3: the consumer profile was wrong for relaxed Krylov, and M-d
  had no Krylov or Davidson case).
  - *Decision:* rewrite §1.3, with `monotone=False` documented for relaxed Krylov. M-d1 adds
    report-only block Davidson and block CG cases (§4.3), ≤ 30 CPU-min in total.
  - *Rationale:* §1.3 contradicted established inexact-Krylov theory, and the fix is cheap. Block
    forms give the emulated contraction a GEMM shape. Davidson is the consumer-adjacent
    eigensolver. Report-only keeps C1 from being gated on methods outside the narrowed profile.
- **IR4. Performance-model N coverage** (gap 4).
  - *Decision:* correct §1.2 item 8 from the counts in the bench files. M-c covers N ∈ {4, 5, 6,
    7, 9, 10, 12, 13} plus the anchors 8 and 14 on 8 shapes (§4.3). Until M-c passes, the
    default levels and every §2.4 or §4.10 schedule gain at N ∉ {8, 10, 14, 16} are
    [M, model-only, outside hold-out].
  - *Rationale:* the counts were reproduced from the N columns of `bench/sweep_calib.txt`,
    `sweep_perf_validate.txt` and `sweep_cache.txt`. The schedules use N = 6 and 9, the
    defaults 6 and 12, and the 1024³ fit has K4 wave steps at 9 and 13; all are unvalidated.
    Anchors cost little GPU time and make a miss attributable.
- **IR5. L3c in Lean** (gap 5: the Lean plan rested on an unformalized L3c).
  - *Decision:* stage 3 gets L3c, its compositions, the normal-range final rounding and an FP
    summation lemma γ_k in Lean before L7 (§4.4), and is re-sized S → M. The code deliverables do
    not wait on the Lean; "C2 done" does.
  - *Rationale:* proofs/NOTES.md maps only L3a and L3b, with rounding as a hypothesis; L7 needs
    L3c and γ_{S−1}. R2's "prove where first relied on" puts it in stage 3. Implementation does
    not block on Lean (the paper-track preference).
- **IR6. Several partners** (gap 6: K-weights for an operand with several partners).
  - *Decision:* the multi-partner rule of §4.5. w is defined on X; the partners must share X's
    K-label set; the clamp uses union extrema; the guard is per pair, with G = min_p G_p. It is
    in `oracle.kweights` from stage 2. The C API shape is finalized at the stage-4 gate.
  - *Rationale:* in DF-CCSD, Bov is contracted over Q with Bov and Bvo. "Every partner carries
    the same w" was unusable without a rule for choosing w. Union extrema keep the clamp exact
    for every partner, and the min over pairs keeps never-worse pairwise, as proved.
- **IR7. The ρ recipe as a helper** (gap 7: the rigorous-ρ recipe covered less than claimed).
  - *Decision:* ship `rho_from_output` in `precision.py`, for β = 0 and α = 2^e only, checked
    against `cert_rho_recipe` (§4.2). M-d1 measures the next-iteration miss rate.
  - *Rationale:* no shipped code implemented the certified FP evaluation. α ≠ 1 adds a rounding
    (DESIGN.md:432), so the restriction keeps the claim exact. The miss rate is the number a
    caller needs to trust the lagged ρ.
- **IR8. Memory pressure and budgets** (gap 8).
  - *Decision:* pack-mode masks; `budget` keeps rank's meaning, and the caller subtracts its own
    resident tensors; a phase status BUDGET; the shared role of a share group is fixed packed;
    "packing never increases ws" becomes certificate I3; joint budgets are caller policy (§4.1,
    §4.2). No ABI change.
  - *Rationale:* packed roles drop regions today (src/host/workspace.cpp:65-68), but per R6
    facts that later formats could break stay certificates. FP64 sources are needed in either
    mode, so subtracting them is exact. The mask lets the benzene-5Z case fall back to in-line at
    N = 14 and pack at lower N.
- **IR9. Provenance** (gap 9: design-phase evidence lived in ephemeral scratch).
  - *Decision:* stage 0 copies the scripts and records to immerdink-eval with a sha256
    manifest (Provenance, §4.1). M-b re-runs every cited [M] figure on this worktree's library
    and replaces any that differ by > 0.5 %.
  - *Rationale:* the project's provenance rule is paper-grade, and scratch is ephemeral. The
    layout follows the eval repo's conventions: scripts and findings tracked, results untracked.
    The re-runs close R-num's unfinished [M] re-runs.
- **IR10. First call at a new N** (gap 10: untested cold capture).
  - *Decision:* round 2 measures the first plan and contract per N, and `graph_phases` captures
    a cold phase. If either fails its bar, a plan-time preload is added, with no ABI change
    (§4.1).
  - *Rationale:* NOTES-M4:220-222 records module loading on the first plan, and K5<N> gets no
    attribute call. A preload is the minimal fix, made only if the test shows a problem.
- **IR11. Output-selective refinement** (gap 11).
  - *Decision:* record it as §3.1 B6, a caller recipe with no API. M-a estimates its value.
  - *Rationale:* with ELEMENTWISE targets, a few max-ρ outputs set N for the whole call. Views
    and the stage-1 helper suffice, so it is a recipe, not an ABI item.
- **IR12. §7 timing and §2.7's wording** (gap 12).
  - *Decision:* stage 3 is ordered after the M-a summary, and its §7 rows cite M-a, or are
    worded "provisional until M-a" (§4.4). §2.7's cost wording is corrected.
  - *Rationale:* §2.6 lists recurring costs of 6–13 % per call that §2.7 contradicted. DESIGN §7
    must not assert dominance on synthetic small-m data before the realistic-size study.
- **IR13. A capture-safe guard path** (gap 13), scoped to stage 4 if it is built.
  - *Decision:* `IMD_KW_ZERO_ON_FAIL` and the documented fully captured pattern (§4.5).
  - *Rationale:* feeding the reported gain into a host function needs a sync. The flag gives a
    rigorous capture-safe path at the cost of one validated bit, and gives up only L5d's
    tightening inside captured iterations.
- **IR14. Rebuild first** (gap 14: the library rebuild was ordered too late).
  - *Decision:* the rebuild, with the API-13 and `nm` checks and the sha256 recorded, is stage 0
    step (a). Nothing else in stages 0–2 runs before it (§4.1).
  - *Rationale:* `host_footprint_monotone`, `host_precision` and every [M] record need the
    API-13 symbols, and the current build is API 5.
- **IR15. DIRECT, prefix planes, raise cost, larger moduli** (gap 15).
  - *Decision:* (1) `levels` may end with the sentinel "DIRECT" (§4.2). (2) The prefix-plane
    option is recorded as the slice comparator (§3.3). (3) M-b measures the share-group raise
    cost (§4.3). (4) One §7 line rejects larger moduli [I], citing arXiv:2603.10634 only if
    verified (§4.4).
  - *Rationale:* the sentinel makes `min_moduli`'s existing DIRECT support usable without
    changing default behaviour. The prefix-plane option and the raise cost are exactly the
    cheaper alternatives a slice revival must beat. An unverified claim must not enter DESIGN as
    fact.
  - *M-b review (§5.8), a reading of (3), not a change:* the raise cost is measured as asked,
    but a revival must beat the best configuration without slices (the shared group paying the
    raise, separate packs, or in-line), and the most it could save is bounded in §3.3; the raise
    figures themselves are not that bar.
- **IR16. Batched fixtures** (gap 16: Q12's escape clause could not fire).
  - *Decision:* real R7 with core orbitals and a synthetic batch, with a pre-registered trigger
    (§4.3, Q12).
  - *Rationale:* core against valence occupied orbitals is the physically natural
    batch-dependent profile in CC. Without the fixture the clause could never fire.
- **IR17. F3 as ABI** (gap 17: F3 rested on thin evidence).
  - *Decision:* F3's second half becomes ABI only after `pack_write_extent` passes, on every
    pack path, with memcheck padding (§4.1). No TLA+ write-footprint invariant.
  - *Rationale:* an ABI promise cannot be withdrawn under §6a. A sentinel window plus memcheck
    padding is cheap and decisive. A TLA+ invariant would only restate a static formula.

**On the previously unresolved items.**
- **IR18** (open item 1). The ROADMAP edits of Q6 and Q7 are approved on Eric's behalf and
  reported to him; he may overrule (Status, Q6, Q7). Both reviewers and all three judges agree,
  and neither edit blocks stages 0–3.
- **IR19** (open item 2). No ehrenfest dump in this cycle. Real operands come from our own psi4
  generator (§4.3, Q14), with the dump as the named fallback.
- **IR20** (open item 3). The optional StreamSpec model is not adopted, and R1 stands (§4.1).
  The ordering half is modelled; the stale-handle half is §6b.10 #3's documented limit.
- **IR21** (open item 4). Stage 6's weights reach the kernels through the q map, candidate (i)
  in its map form, with `imd_plan_create_ex` for weighted plans; (ii) and (iii) are rejected
  (§4.7). The map is the same kind of object as the existing per-k offsets, and unweighted
  layouts stay byte-identical.
- **IR22** (open item 5). Superseded by IR14: stages 0 and 1 already need the API-13 symbols.
- **IR23** (open item 6). Every inferred item is closed by a named measurement, in an order
  where the check precedes the claim:
  - F3's second half by `pack_write_extent` (IR17);
  - the [M] figures by M-b's re-runs (IR9);
  - the §2.6/§4.2 [E] costs by bandwidth-based figures stated at the stage-4 gate, then by M-e
    (§4.5);
  - the M-d2 bars by running M-c strictly first, with the bands re-derived and pre-registered
    if M-c recalibrates (§4.3).
- **IR24** (open item 7). Share groups are ranked jointly over every level ≥ the group floor
  (§4.2). With IR2's composition this is exact and cheap, and it subsumes the max-floor
  simplification.
- **IR25** (open item 8). The binding TLA+ run limits for `GuardSound` and `DeviceReduce`
  (§4.5). A previous TLC run OOM-crashed the machine, so every new model is gated on a measured
  smoke run.

**Gates, order and completion.**
- **IR26. The stage-5 gate is not met.** Stage 5 is formally deferred with triggers T1–T3, and
  its cost bar is restated without `ncu` (§4.6). C1's definition of done does not need stage 5.
  The evidence is read-only: ehrenfest makes precision user-directed and issues no accuracy
  requests, and its Ozaki work is frozen. *(Its "admin-only counters" premise was wrong. The
  restated bar stands because it measures time directly; `ncu` is a diagnostic only, with
  `--clock-control none`: §4.6, "Correction", 2026-10-07.)*
- **IR27. Pre-registered gate rules for stages 4 and 6** (§4.5, §4.7), applied mechanically in
  the next round. Writing them down before M-a reports prevents post-hoc gate choices.
- **IR28. Order of work** (§4.11): round 1 (CPU) and round 2 (GPU), following the dependencies.
- **IR29. Definition of done** for C1 and C2 (§4.11). Completion is a finite checklist in
  which every gated item ends as built, closed or deferred with a trigger.

### 5.4 Stage 0: the implementation's design questions (independent reviewer, 2026-10-07)

The stage-0 implementation raised seven questions that the text above did not settle. Each was
decided conservatively and then put to an independent reviewer, whose verdicts are binding.
- **S0-1. What "every pack path" means for F3** (IR17 named "KF, KL, K2/K3"). *Verdict:
  confirmed, on condition that the radix > 1 cases pass.* `imd_pack` launches only `kp_prep`
  and one KL (contract.cu `enqueue_pack`); KF and K1 + K2/K3 write only the contract
  workspace, so IR17's list was a misnomer. Every pack path means every `imd_pack` launch shape
  for the write extent, plus packed contracts on every producer path, under memcheck, for the
  read extent. The reviewer recomputed the launch rule for the 88 original cases: they covered
  all 26 (along rows, tile height, values per thread, W) shapes, but not the radix > 1 row
  mapping a batch-fastest X takes. The fix added those cases (113 packs in all), and the suite
  now asserts its own coverage (DESIGN §6g.8). It passed under both the sentinels and memcheck,
  so the publication stands. Had it failed, the library would have been fixed to meet the
  promise; an ABI promise is not withdrawn.
- **S0-2. F1 for `imd_workspace_size_packed` with RN.** The packs' κ ≥ T(N) makes the smallest N
  NOT_SUPPORTED, so DESIGN §6a states F1 over the N where the query succeeds. *Verdict:
  accepted, and the up-set is part of the promise.* T(N) is strictly increasing on 2..20, and
  RN accepts k ≤ (T − 1)², so the N where the query succeeds form an up-set. C-ws needs that: if
  any phase's query succeeds, the query at N_max succeeds too. `host_footprint_monotone`
  asserts the up-set and the exact NOT_SUPPORTED boundary.
  - *Amended by the re-review (independent reviewer, 2026-10-07; binding): the up-set is
    withdrawn from the promise before merge, and the downward statement is promised instead.*
    The first verdict overlooked INVALID_VALUE for a size past uint64 (§5 row 5), which also
    grows with N and g: RA = g·b·m·k̂ and W = N·b·m·n̂ in the workspace, and res = N·b·r·k̂ in
    a pack. Measured on this build for "ik,kj->ij", m = 2^30, k = 2^31, n = 16: the plain query
    (TRUNC, g = 0) succeeds at N = 2..7 and returns INVALID_VALUE at 8..20. With A packed and
    RN it returns NOT_SUPPORTED at 2..4, SUCCESS at 5..7 and INVALID_VALUE at 8..20. So the N
    where the query succeeds form an interval, not an up-set. The κ-feasible N are still an
    up-set. What C-ws uses, and what DESIGN §6a F1 now promises, is the downward direction: if
    the query succeeds at (N_max, g_max), every (N, g) with 1 ≤ g ≤ N ≤ N_max and g ≤ g_max
    (g = 0 counting as N) either succeeds with at most that many bytes or is NOT_SUPPORTED by
    the packs' κ ≥ T(N) under RN. The same text is in the headers of `imd_workspace_size` and
    `imd_workspace_size_packed`. `host_footprint_monotone` now models every status from params
    (overflow, then κ). It asserts the pattern at each g (a NOT_SUPPORTED prefix, a SUCCESS
    interval, an INVALID_VALUE suffix) and the downward statement, step by step on every case
    and pair by pair on two shapes that pass uint64. The measurement above is one of those
    shapes.
- **S0-3. `PackedBuffer` refuses a buffer with no record** (never written by `Pack.pack`).
  *Verdict: accepted.* A buffer filled by `pack_ptr`, a copy or persisted bytes has no
  host-side provenance, so it goes through the raw path (DESIGN §6g.5).
- **S0-4. `PackMismatchError.reason` for that refusal.** *Verdict: keep None.* The attribute
  means the library's plan-time `IMD_PACK_MISMATCH_*` reason; the digest comparison is a
  host-side diagnostic of a different reliability, and the message names the difference
  (DESIGN §6g.5).
- **S0-5. `packed_repack_n` asserts that the stale plan with a raw buffer returns SUCCESS and a
  wrong C.** *Verdict: accepted.* DESIGN §6b.7 states that hazard as fact. If the library ever
  detected stale buffers, it would reverse §6b.10 #12, and the DESIGN text and the assertion
  would change together.
- **S0-6. `CUDA_MODULE_LOADING=LAZY` forced** in `test_phases.py` and (overridable with
  `IMD_BENCH_MODULE_LOADING`) in `bench/first_call.py`. *Verdict: accepted.* LAZY is the CUDA
  default and the worst case for capture: a function first loaded inside the capture is what
  IR10 asked to test.
- **S0-7. `host_footprint_monotone` covers N = 2..20** where §4.1's grid said 2..19. *Verdict:
  accepted.* N_MAX is 20; the superset covers the W = 3 boundary (N ≥ 17) and the whole
  published F1/F2 range.

### 5.5 Stage 1: the implementation's design questions (independent reviewer, 2026-10-07)

The stage-1 implementation raised nine questions that §4.2 did not settle. Each was decided
conservatively, in the spirit of §4.2, implemented and documented (DESIGN §6g.10–§6g.11), and
put to an independent reviewer, whose verdicts are binding. The review was approve-with-fixes:
two must-fix items (MF1, MF2) and nine should-fix items (SF1–SF9), all applied (DESIGN v1.19
change list, "Stage-1 review fixes"). A re-review of the fixes was approve-with-fixes again:
one must-fix item (MF3, under S1-7) and four should-fix items (partial batch dims of a norm
bound, S1-10 in the API docstrings and here, a docstring reflow), all applied ("Stage-1
re-review fixes"); its verdicts on the three questions the fixes raised are under S1-5, S1-7
and S1-10. A second re-review was approve-with-fixes too: one must-fix item (MF4, under S1-7,
binding) and three should-fix items (the MF3 sentence of DESIGN §6g.11, the S1-10 wording
of the docstrings, and the probe's forms with an exact-rational check in the test), all
applied ("Stage-1 second re-review fixes"). A third review of the MF4 fix approved it with
should-fix items only (a test of bounds shared over some batch axes, the exact conversion of
norm bounds, the GEMM view of a tensor output in DESIGN §6g.11), all applied ("Stage-1 third
review fixes"); its verdict on the question MF4 left open is S1-11.
- **S1-1. Parameters beyond §4.2's signature.** `vec_modes` (§4.2 ranks "the given vec modes"
  but names no parameter); `perf` (regime, cache, clock of rank and the pack predictions: the
  `Planner` carries only the calibration and the memo, not the regime or cache §4.2's comment
  attributes to it); a dict form of a contraction with `beta` (rank's β flag; CC residual
  builders accumulate); `configs(iterations)` (so a `"phase"` reuse prices the same in
  `configs()` as in the phase, and the phase's choice is a `configs()` plan); `mode=` of
  `rho_from_output` (a DIRECT call's U is γ_k, which exceeds the EMULATE bound at large N, so
  using the EMULATE U there would be unsound). *Verdict: accepted; `mode=` on condition of its
  evidence (SF3).* Each fills a real gap without touching planner.py or the ABI. The evidence
  is added: `cert_rho_recipe` (1′) checks the recipe on the native FMA chain in exact
  arithmetic with U = γ_k, and `host_precision` checks U ≥ params' exact bound for both
  dispatches at N = 2–20.
- **S1-2. Share-group members** are `(name, "A" | "B")`, the role reading the group's pack, or
  a bare name (equal N only). Groups that share a contraction are merged for N; one role is in
  at most one group. Whether the members' packs have one key is left to plan creation
  (PACK_MISMATCH), not checked by `PrecisionPlan`. A shared role's int reuse is the total
  calls per pack, and the members' kinds must agree. *Verdict: accepted.* PACK_MISMATCH is
  the library's authoritative one-key check; a Python copy would be a second, weaker source of
  truth. The docs now say that `Phase.pack_bytes` repeats a shared pack under every member
  and `footprints()` counts it once (SF9).
- **S1-3. Statuses.** The normal status is `MET`; precedence MISSED > BUDGET > DIRECT.
  `configs()` reports `OK` or `BUDGET`. In a share group where no level fits every member,
  the group takes its floor; members that fit there are MET, the others BUDGET. Ties between
  levels of a group go to the smaller N. *Verdict: accepted, amended.* `BUDGET` may be
  reported only when the reported candidate meets its target; a fallback whose candidate has
  the ACCURACY bit is `MISSED` (`_settle`'s guard). The tie rule (smaller N between levels of
  equal composed time) is documented for single contractions too (SF6).
- **S1-4. MISSED** contractions are ranked with their target, so the reported err fields are in
  its metric, and chosen by status and budget only (the accuracy bit ignored). *Verdict:
  accepted.*
- **S1-5. DIRECT** choices are in-line (no packs), g = 1, with the plan's num_moduli the top
  emulated level (a DIRECT plan ignores it); `Phase.N` reports `"DIRECT"`. Under the monotone
  floor DIRECT stays, and a target DIRECT misses is reported MISSED there. *Verdict: accepted
  for single contractions; amended for share groups (binding, MF1).* A DIRECT call reads no
  pack, and the one-N rule exists only for the shared pack, so a member at DIRECT leaves its
  group's N coupling for that phase: it is settled alone, and the other members are ranked
  jointly over the emulated levels ≥ their maximum floor. The implementation had dragged
  every member to DIRECT; the reviewer reproduced a member whose TYPICAL target is met at
  N = 12 and missed by γ_k reported `BUDGET` at native FP64. The brute force of
  `host_precision` encoded the same rule, so the fix changed both, and a scenario pins it
  (both member forms, a coupled third member, and a member that linked two groups).
  *Re-review (binding):* groups joined only through a DIRECT member (G = {x, y}, H = {x, z},
  x at DIRECT) are ranked separately. The one-N rule exists only because members read one
  shared pack, and with x at DIRECT G's pack is read by y alone and H's by z alone; tying y
  and z would add a constraint with no reason behind it. "The other members are ranked
  jointly" means the other members of each group, not of the merged component; members
  listed together in one group stay tied.
- **S1-6. Footprints.** Per level, the maximum over the allowed pack modes × vec modes at the
  largest g whose query succeeds (F1, ABI); the plan device bytes are included; the totals
  share no workspace; a shared pack counts once, keyed by its group. *Verdict: accepted,
  amended (SF4).* At the largest g the workspace follows from F1's downward statement, but the
  plan device bytes would rest on I2, a certificate and not ABI; `footprints()` now takes the
  maximum over every g that succeeds.
- **S1-7. `rho_from_output` details.** A C of exactly 2⁻¹⁰²² with e < 0 is uncertified (it
  may be a subnormal 2^e·D rounded up, so D is not recoverable); a ρ̂ that overflows is
  uncertified; α must be a positive power of two (−2^e would also be exact, but §4.2 says
  2^e); norm bounds broadcast against C (1-D vectors for a 2-D C) and are validated (NaN or
  negative: ValueError); uncertified entries are +∞. *Verdict: accepted, amended (binding,
  MF2).* The uncertified classes are required for soundness; −2^e may be added later with one
  certificate line if a consumer needs α = −1. The broadcast rule was unsound: for a batched C,
  or mixed 1-D and 2-D bounds with m = n, numpy paired `row_norm_ub[j]` with column j and
  certified ρ̂ ≈ 1 where ρ ≥ 100. Now, for C.ndim ≥ 2, a 1-D row bound is always axis −2 and
  a 1-D column bound axis −1, each of exactly that length. *Re-review: the length-1 detail
  accepted (binding), the rule extended (binding, MF3).* A length-1 1-D bound on a longer axis
  raises (a caller who wants one value for every row passes a scalar); a 0-D or 1-D C has no
  row or column axis to confuse, so its bounds broadcast elementwise. The MF2 rule was not
  complete for bounds of ndim ≥ 2: numpy's per-batch norms without keepdims, (b, m) and
  (b, n), were paired with C's rows or columns whenever b = m (and b = m = n for the row
  bound), and certified ρ̂/ρ ≈ 0.006. The rule for C.ndim ≥ 2 is now: a 1-D bound follows
  the axis rule with its exact length; a row bound of ndim ≥ 2 has shape[−1] = 1 and a column
  bound shape[−2] = 1; and (should-fix) a bound carries none or all of C's batch dims, since
  numpy would align a partial set with C's trailing batch dims. Anything else raises
  ValueError. Scalars, 1-D vectors, (…, m, 1) and (…, 1, n) still pass; a transposed (1, m)
  row bound, an (m, 1) column bound and an elementwise (m, n) bound now raise.
  `host_precision` checks per-batch keepdims bounds on unequal batches (one and two batch
  dims) against each batch's own 2-D call bit for bit, and ρ̂ ≥ row_i·col_j / |C_ij| in
  exact rationals; every rejected form raises. *Second re-review: the rule made structural
  for a batched C (binding, MF4); it overrides MF2's "for any batch dims" for C.ndim ≥ 3
  only.* The MF3 rule still accepted per-batch scalars written without keepdims and paired
  them with C's rows or columns. The reviewer's probe (A[0] and B[−1] scaled by 100, the
  bounds rounded up, N = 14, k = 16, every output checked against the true ρ in exact
  rationals): (a) a row bound `norm(A, axis=-1).max(-1)` of shape (b,) on C (b, m, n) with
  b = m was taken as the axis −2 vector: at b = m = 3, n = 5, 45 outputs certified, 15 below
  ρ, min ρ̂/ρ 0.0081; at b = m = n = 4, 20 below, 0.0101. (b) A column bound of shape (b,)
  with b = n: at b = m = n = 4, 16 below, 0.0113; at b = 3, m = 5, n = 3, 10 below, 0.0105.
  (c) A row bound of shape (b, 1), e.g. `norm(A.reshape(b, -1), axis=1, keepdims=True)`,
  with b = m passed the shape[−1] = 1 check and gave the same 0.0081. The documented
  (b, 1, 1) forms gave 0 below. No 1-D or 2-D bound on a batched C can be told apart from a
  per-batch scalar where b equals m or n, so the fix is structural. **The rule:** for
  C.ndim ≥ 3 a bound is a 0-D scalar or has exactly C.ndim dims, each batch dim C's or 1, a
  row bound with shape[−1] = 1 and a column bound with shape[−2] = 1 ((…, m, 1), (…, 1, n),
  (1, …, m, 1), (b, …, 1, 1) and so on); any 1-D or 2-D bound on a batched C raises
  ValueError, and the message names the keepdims spelling (e.g. `r[None, :, None]` for a
  bound shared across batches). For C.ndim = 2 the rule is unchanged (a scalar, a 1-D bound
  on its own axis of exact length, (m, 1) / (1, n)); a 0-D or 1-D C still broadcasts
  elementwise; the partial-batch check is subsumed by the ndim rule. `host_precision` now
  checks, on 3-D and 4-D C: a vector shared by every batch in the full-ndim spelling (also
  with the batch dims spelled out, or a scalar on one side) equals each batch's 2-D call bit
  for bit; per-batch keepdims bounds and per-batch scalars (…, 1, 1) on unequal batches
  equal each batch's 2-D call with no certified ρ̂ below the true ρ in exact rationals; the
  probe's forms ((b,) rows at b = m, (b,) columns at b = n, (b, 1) rows at b = m), shared
  1-D (m,) and 2-D (m, 1) / (1, n) bounds, the MF3 forms and partial-batch bounds raise. A
  rejected form that is accepted fails with its count of certified outputs below the true ρ.
  *Mutation check:* with the ndim rule dropped, and on the pre-MF4 code, the test fails
  first with "norms (3,) x (3, 1, 3) on C (3, 3, 3): accepted, 27 outputs certified, 9 of them
  with rho_hat below the true rho"; collecting every accepted form instead, the mutant
  accepts 17 unsound forms (the pre-MF4 code 11; the other 6 are partial-batch forms, which
  only the ndim rule now rejects), with 2–28 certified outputs below ρ each. The only
  in-repo caller, `bench/accuracy/md1_converge.py`, passes 1-D bounds on a 2-D D and is
  unaffected. *Third review (should-fix items, applied):* (a) the "each batch dim C's or 1"
  half of the rule is pinned by a test: on a 4-D C, bounds shared over one batch axis only,
  (b0, 1, m, 1) × (1, b1, 1, n), equal each batch's 2-D call with no certified ρ̂ below the
  true ρ (the reviewer's probe found them accepted and sound). (b) Norm bounds were
  converted with `np.asarray(…, dtype=float64)`, which rounds to nearest, so an integer
  above 2⁵³, a Fraction or a Decimal could become a value below the caller's upper bound
  before the upward-rounded recipe ran (pre-existing). Decided conservatively: a bound is
  converted exactly or refused. Float arrays of at most 64 bits widen exactly and integers
  of magnitude ≤ 2⁵³ convert exactly; an integer above 2⁵³ raises ValueError, and a long
  double, an object array (Fractions, Decimals, Python ints beyond 64 bits), complex or bool
  raise TypeError. A Python list is converted by numpy before the check (an integer above
  2⁵³ among floats is already rounded), which the docstring and DESIGN §6g.11 say: pass
  numpy float arrays. Every in-repo caller passes float64. (c) DESIGN §6g.11 and the
  docstring say that C and its bounds are in the GEMM view: a tensor output whose modes
  interleave A's and B's free modes (the ring's abij of "kjbc,kiac->abij", as in the module
  example) is permuted to (batch modes, A's free modes, B's free modes) and reshaped first,
  since the helper pairs bounds with outputs by position only.
- **S1-8. `fail_budget`** is per phase (§4.2 step 7: P over the phase's calls). A run budget is
  the caller's split; the example calls `phase()` every iteration with `iterations=1` and
  P / T_max. *Verdict: accepted.*
- **S1-9. A phase names every contraction** (a subset raises ValueError). *Verdict: accepted.*
- **S1-10 (from the MF1 fix).** In a phase where a share-group member is at DIRECT, the
  shared role's reuse still counts that member's calls: it is fixed at construction (the
  members' summed calls), so every phase's choice stays the `configs()` lookup bit for bit
  (configuration and composed time) and the plans created before the loop cover it. The
  other members' pack term is then priced as if those calls read the pack, an optimistic [M]
  figure in a rare case. Counting only the readers per phase would price it exactly; within
  a level it changes every candidate's time by one constant, so in exact arithmetic the
  configuration at a level is unchanged, but the composed times would no longer equal
  `configs()`'s, and floating-point rounding could reorder near-ties. *Verdict (re-review):
  accepted as implemented, the reuse stays fixed (binding).* The promise that matters is
  that every phase's choice is a `configs()` plan, so the plans created and frozen before the
  loop cover every phase (graph capture, `PlanCache.freeze`); a near-tie reordered in
  floating point could pick a configuration whose plan was never created, a hard failure in
  the loop. The cost is a model figure only: the other members' pack term, and
  `Phase.time_s`, are optimistic by at most pack_s × (the DIRECT members' summed calls / the
  summed calls) per iteration, in a phase that already runs a native FP64 (1:64) contraction,
  which dominates its time. Numerics, budgets and footprints are unaffected. The same covers the
  shared role staying packed when one member alone reads it in that phase. The `reuse` and
  `Phase` docstrings say so (`Phase.reuse` reports e.g. {B: 4} where one call reads the
  pack). Revisit only if a consumer runs share groups with DIRECT members in practice; the
  fix would then be a hybrid: each level's configuration from `configs()`, the actual
  readers only for the level comparison and the reported time.
- **S1-11 (from the MF4 fix). Is a separate dim-by-dim check of a batched bound needed?**
  With ndim = C.ndim enforced, MF4 relies on the existing "broadcasts exactly to C.shape"
  check for "each batch dim C's or 1" and the row and column extents, rather than a check of
  its own that would name the argument. *Verdict: accepted as implemented (binding); no
  separate check is required.* When every array has C.ndim dims, broadcasting cannot
  realign axes, so `np.broadcast_shapes(row, col, C) == C.shape` holds exactly when each dim
  of each bound is C's or 1: the rule is enforced exactly, and only the error text is more
  general (it names both shapes). The reviewer re-ran `host_precision` and both mutants (the
  ndim rule dropped; the pre-MF4 code of 74a1eed), which fail with the reported message, and
  an adversarial probe of 5509 realistic bound pairs on 3-D, 4-D and 2-D C (max over any
  subset of axes, keepdims on or off, then optionally `[..., None]` or `[..., None, :]`):
  the MF4 code accepted 384 forms with 0 certified outputs below the true ρ in exact
  rationals, the pre-MF4 code 216 unsound ones. Only explicit mis-spellings at full ndim
  (batch-indexed values placed on the row or column axis) get through, and no shape rule can
  detect those.

### 5.6 Stage 2: the implementation's design questions (independent reviewer, 2026-10-07)

The stage-2 CPU evidence (M-a, M-d1) raised the questions below. Each was decided
conservatively during the study and put to an independent reviewer, whose verdicts are binding.
The review was approve-with-fixes: three must-fix items (the gate-(ii) report, these notes, the
`timeout` order) and the should-fix items, all applied (Status, stage 2).
- **S2-1. Gate (ii): what is a "realized regression"?** The pre-registered measure counts the
  sampled outputs whose error exceeds fast's at the same N by > 2× (1). At equal accuracy that
  fires on independent rounding noise, so the study also recorded two noise-robust measures:
  (2) outputs whose median ratio over the informative N exceeds 2, and (3) the N at which the
  sampled max error exceeds fast's by > 2× while fast's is below 1. *Verdict: this round, the
  literal measure applies, so (ii) is NOT met* (2–19 outputs per guard-accepted Tier-L
  fixture at d*, max ratio 247), and that is reported; the outcome is unchanged because (i)
  fails. *For any revival, (ii) is re-registered before new evidence:* it passes iff (2) and
  (3) both hold on every guard-accepted Tier-L or real fixture, and (1) is reported as a
  diagnostic (§4.5). IR27 exists to prevent post-hoc gate choices, so a pre-registered rule is
  not relabelled after the data; (2) + (3) is the sound, stricter replacement for the future.
  *Correction to the first report:* the events are at N = 4–5 only for d* = 3; at d = 0 they
  also occur at N = 6, 12 and 14 (TRUNC). "Fast mode is saturated" there was misleading: fast
  mode already meets normwise 1e−8 at N = 4 on decay_opp; the saturation is per output
  (high-ρ outputs whose fast-mode D carries no correct digit).
- **S2-2. Elementwise targets on real operands.** Point-group zeros (ρ up to 1e20) make max
  elementwise ≤ 1e−8 unreachable for every scheme at N ≤ 20, so (b)-elementwise and (c) are
  vacuous on real data. The study added a labelled diagnostic, max elementwise over the
  outputs with ρ ≤ 1e8. *Verdict: not retroactively.* This round's gate stands as applied (all
  sampled outputs with x ≠ 0), and the ρ ≤ 1e8 column stays a diagnostic. For any revival,
  (b)-elementwise and (c) are judged on the outputs with ρ_ij ≤ 1e8 (exact x, plain norms),
  DESIGN §6f.1's "outputs that must meet the target", with the all-outputs reading alongside
  (§4.5). On this round's data the outcome is the same either way: guard = 0 on every real
  contraction, and on the ρ ≤ 1e8 outputs K-eq never gains with its costs (R2b saves 1
  modulus at 1e−8 but is 36 % slower).
- **S2-3. Q13: stage 4 closed by the pre-registered rule; acceptable as final for this cycle?**
  The real operands are stand-ins: MP2 amplitudes and bare integrals [I]. *Verdict:
  confirmed.* Stage 4 is closed with the trigger "a real-operand family meeting (i)", stage 6
  is deferred because its gate requires a shipped stage 4, and `oracle.kweights` plus M-a are
  C2's deliverable. The stand-ins are accepted as recorded under IR19; a future consumer dump
  of converged, t1-dressed operands falls under the existing trigger. The result is
  structural, not marginal: R1, R2, R6 and R7 have Δ ≡ 0 by construction, whatever the
  dressing, and R2b, R3, R4 and R5 have aligned profiles with G ∈ [−4, 0]. The reviewer
  checked guard8 = guard14 in every record, 0 fallbacks and 0 FP64 violations, and re-ran the
  engine self-test and test_oracle.
- **S2-4. The guard when w ≡ 0.** The rule gives G = 0 and so guard = 0, although the weighted
  call is then the plain call. *Verdict: keep guard = 0.* guard = 1 is a soundness claim tied
  to one lemma (L5c, G ≥ 3); a special case would add a second meaning and a branch to
  `GuardSound`, the oracle and the GPU tests for no gain, and `num_weighted` already tells the
  caller. Documented in §4.5 (step 5 and the caller recipe), the `oracle.kweights` docstring
  and, later, the stage-4 header: with `num_weighted` = 0, skip the pre-scaling and use the
  plain request.
- **S2-5. The report's ORACLE-CHOICEs.** (a) Under ZERO_ON_FAIL the report describes the
  written w: `clamped` = #{w ≠ w0} = #{w0 ≠ 0}, `num_weighted` = 0, `max_abs_w` = 0, while
  `guard` and `gain` keep their computed (failing) values. (b) When G is undefined,
  `gain_row_min` and `gain_col_min` are INT32_MIN too, even where one side is defined. (c) The
  oracle's detail fields `clamped_max_abs` and `clamped_fp64` describe the pre-zeroing w.
  *Verdict: confirmed, keep all three,* and document them before the C API is fixed: done in
  the `oracle.kweights` docstring and §4.5's report struct and ZERO_ON_FAIL text. (a) is the
  field's own definition applied to the w the helper writes, as §4.5 says ("sets clamped and
  num_weighted to match"); reporting the failing G keeps the capture-safe pattern rigorous.
  (d) The re-review added one more, binding in the same way: `gain_row_min` and
  `gain_col_min` name the rows of the caller's A and the columns of the caller's B as passed
  to the plan, not the operands after lowering's swap, so a swapped plan (column-major C)
  reports the same values as the unswapped one. test_oracle (12) pins this; it is stated in
  the docstring, §4.5 step 5 and the report struct.
- **S2-6. Cuts triggered by a tier's allotment.** Tier S projected about 26 min against its
  25, while M-a as a whole projected about 55 against 90; the pre-declared trigger names the
  total, and cut (1) touches Tier L only. The study applied cut (2). *Verdict: yes, and the
  cut as applied is accepted.* The per-tier allotments are part of the hard budget: a tier
  whose projection exceeds its allotment applies the earliest pre-declared cut that reduces
  that tier, never cutting the real operands, the guard or m = 4096, and the projection that
  triggered it is recorded (§4.3). d* is unaffected: every deadband ties at 30/30 rejections.
- **S2-7. `timeout` around memcap.** `timeout 600 tools/memcap.sh …` did not stop the command,
  which memcap runs in a systemd scope: a timed-out chunk kept running and wrote 14 duplicate
  records (bit-identical; the summary keeps the first, and all CPU time is counted). *Verdict:
  the correct form is `tools/memcap.sh timeout 600 cmd`.* It is now in tools/memcap.sh's usage
  comment, §4.3's budget text, and the usage lines of `c2_gate.py`, `md1_converge.py` and
  immerdink-eval `scripts/c1/psi4_operands.py`, which had taught the wrong order. Keeping the
  first record, deduplicating and counting all CPU time are accepted.
- **S2-8. The reading of IR7's ρ-miss rate.** The study counted only calls whose lagged request
  was MET, with levels up to 20; with the forcing rule's levels (≤ 14), MISSED requests run at
  the top level would have counted as misses of the lag. *Verdict: accepted* (§4.3): a request
  no level can meet is a capacity limit, not a failure of the lagged ρ. The count is 1 miss in
  228 MET calls (the first report said 266). IR7 closes only with the shipped helper: the
  first runs used a local recipe that also raised U by (1 + 2⁻⁴⁰). *Re-run 2026-10-07* with
  `precision.rho_from_output` as of 51dba1e: the same N on every call, ρ̂ within 6.2e−16
  relative of the local recipe's, and the same single miss (PPL at 1e−6, realized 1.013e−6,
  after the ring term's ρ jumped from 2.5e6 to 2.0e9). IR7 is closed.
- **S2-9. Caller guidance from M-d1** ("Krylov: start tight, relax"; "subspace methods:
  recompute at switches"; DIIS reset). *Verdict: accepted as report-only caller guidance
  (IR3), labelled as CPU evidence on one fixture each, not a C1 gate, with two corrections:*
  Davidson's ramp still costs +5 iterations with the recompute, so §4.3's expectation is not
  met; and the relaxed CG rule leaves a residual gap (7.1e−8 against 3.3e−10), so the
  relaxation must be scaled by the expected iteration count. As corrected, it is in §1.3.
- **S2-10. The Davidson fixture was changed after the first runs.** The first H had
  ‖2BBᵀ‖ ≈ 950 Ha and did not converge even with FP64 matvecs and diagonal preconditioning;
  the coupling is now normalized to 1 Ha, and the earlier records are kept. *Verdict:
  accepted.* The case is report-only, the change and its reason are disclosed, and the
  findings note that the normalized case is diagonally dominant and easy.
- **S2-11. The synthetic CC family at natural shapes** (m ≠ n), matched to each tier's class
  by (o, v, naux); cc_ppl and cc_ring have no k = 16384 tier (the (ac|bd) operand would exceed
  the 3-GB RSS cap). *Verdict: accepted.* The class is defined by 2mnk ≥ 1.7e10, not by m = n
  (Tier L-2048: cc_df 2025² × 3600 = 2.95e10, cc_ppl 1024 × 3136² = 2.0e10, cc_ring 2048³ =
  1.7e10), natural shapes are what the consumer runs, cc_df covers k = 16384, and the family
  has w ≈ 0 at every size anyway.
- **S2-12. Q12's escape clause** (decided in the report, not raised as a question). *Verdict:
  Q12 (a) stands for this cycle, but the closure is vacuous on real data and recorded as
  such.* R7 (iaQ,ibQ->iab) is a self-contraction in every batch, so Δ ≡ 0 for shared and
  per-λ weights alike, and the real-fixture trigger could never fire. The synthetic batch
  shows per-λ weights saving 3–4 moduli on all-output max elementwise (0 on the ρ ≤ 1e8
  outputs and normwise). At any stage-4 revival, Q12 is re-tested on a real batched
  contraction that is not a self-contraction, before the C API is fixed (§4.5).
- **S2-13. Q15: K-segmentation beyond K-equilibration?** The one positive case, R2b at normwise
  1e−15, is 2 % faster. *Verdict: Q15 (a) stands,* a documented recipe with no API flag. 2 % is
  inside the model's p90 error (10–13 %) and below the 5 % bar of every other gate here;
  elsewhere K-seg 2 needs 1–4 more moduli than K-eq and loses elementwise accuracy on the
  ρ ≤ 1e8 outputs.
- **Review should-fix items applied besides the above.** B6 synthetic at m = n = 512 with exact
  x (outliers, opp_nonsep and φ = 4; k = 3600 and 16384; 5 seeds; no split pays); the Appendix
  B trigger compared with what a caller gets and restricted to non-separable families of the
  class (still 0 cases); gate verdict lines computed by `c2_summary.py`, with the memory of
  scaled copies in the gate tables; water R1 and R5 labelled not gate-admissible; the engine
  self-test extended to the several-partner and batched paths against `oracle.kweights`, and
  registered as the CPU ctest `c2_engine_selftest`; test_oracle (12) given a swapped view with
  X = A and an unswapped one with X = B; the provenance of each record extended by a dirty
  flag and every study file's sha256.

**Re-review** (independent reviewer, 2026-10-07; approve-with-fixes). One must-fix: the
provenance statements named the wrong commit for most records (now Status, stage 2,
*Provenance*). Its verdicts on the review round's design questions, binding:
- **S2-14. Gate (ii) is evaluated on K-eq d*,** the scheme stage 4 would ship; d = 0 and stale
  weights are diagnostics in the regression table. *Verdict: confirmed* (Q11 fixes d* = 3 as
  the shipped deadband). (ii) is not met whichever scheme is counted (d = 0: 1–14 outputs per
  fixture, d*: 2–19, stale: 2–10). *Binding for any revival:* if the revived recipe allows
  reusing a w on changed data instead of re-running the helper per changed operand, the
  stale-weight runs count as gate-(ii) runs too (§4.5).
- **S2-15. Appendix B's non-separable set.** Separable = {decay_opp, decay_same, uniform,
  same_sign}; every other family is non-separable (outliers, φ, mixed, opp_nonsep, the CC
  families, the real operands). *Verdict: confirmed:* for a revival trigger the wider set is
  the conservative (more sensitive) reading. The trigger is 0 cases under either reading,
  compared with what a caller gets (guard-accepted K-eq d*, otherwise fast mode) and with
  K-seg 2.
- **S2-16. The memory of scaled copies (gate (b))** is reported as 8(mk + nk) bytes, both
  operands pre-scaled and kept beside the plain ones. *Verdict: confirmed as an upper
  estimate, labelled as such;* it cannot flatter K-eq, and every real R1–R6 has guard = 0
  this round. For a revival the gate table also gives the figure for the recipe actually
  proposed, with the bound alongside (§4.5).
- **S2-17. The several-partner and batched self-test** lives in `c2_gate.py`
  (`self_test_modes`, next to `run_multi` / `run_batched`), runs in every `c2_gate` process
  before any record and in the ctest `c2_engine_selftest`; `keep_w` defaults off, so record
  formats are unchanged. *Verdict: confirmed, subject to two fixes, both applied:* the ctest
  is among `tools/test.sh`'s CPU-only tests, so it no longer waits for or holds the shared GPU
  lock; and the self-test asserts its coverage (guard 0 and 1 at P = 1, 2 and 3, X in role A
  and B at P ≥ 2, the FP64 clamp binding at P = 1 and at P ≥ 2, and w = 0 in a
  several-partner case, a batched descriptor and a slice, added for S2-4), so a fixture or RNG
  change cannot remove it silently. G undefined stays covered by test_oracle (12) only.
- **S2-18. The IR7 re-run's provenance:** its records name d89988a with
  `python/immerdink/precision.py` dirty (another agent's uncommitted edit), with that file's
  sha256 equal to 51dba1e's. *Verdict: confirmed, checked by the reviewer:* the sha256 in all
  4 `md1_rho.jsonl` records equals the committed file of 51dba1e and f97696a, the other study
  files match f97696a, and the same holds for the 30 review-round B6 records. Records tied by
  content are what `provenance()` is for.
- **Re-review should-fix items applied.** A spot check of the 52054e9- and d22a686-era records
  re-run on committed code at 928b49c (Status, stage 2, *Provenance*, which names exactly what
  it covered: 3 of 66 Tier L, 2 of 11 real, 1 of 2 multi, 1 of 2 batch, 1 of 7 real-operand
  B6, M-d1 jacobi, davidson and cg; the first wording called it the rest);
  `c2_summary.gate_verdict`'s (iii) sums over every run that scales an operand (Tier S
  included, stale weights included; B6 scales nothing), and a guard-accepted contraction whose
  (b)/(c) the script does not compute (R6) makes (i) INCOMPLETE instead of NOT MET; the
  `oracle.kweights` docstring, §4.5 step 5 and the report struct say that `gain_row_min` /
  `gain_col_min` are the caller's A rows and B columns, independent of lowering's swap (S2-5
  (d)); the B6 findings table shows the reaching and the unreaching seeds separately and says
  why no split pays at m = n = 512; the M-d1 caller guidance's move into DESIGN is tracked as
  an open item (Status, stage 2).

**Second re-review** (independent reviewer, 2026-10-07; approve-with-fixes). One must-fix:
the provenance text again claimed more re-running than was done. It called the 928b49c
re-run "the rest" and said M-d1 diis and ppl "run the same fixed-point code as jacobi",
while that re-run covered 3 of 66 Tier L, 2 of 11 real, 1 of 2 multi, 1 of 2 batch and 1 of
7 real-operand B6 records and no diis, ppl or `rho_levels14` record, and jacobi never runs
the DIIS branch or the PPL class. Fixed: Status, stage 2, *Provenance* calls it a spot
check, names what it covered and what not, and states what its records alone show (gate
(ii) fails). Its should-fix items, applied: every 52054e9-era record that is not superseded
re-run at 5925219 (121 records, all bit for bit: Tier L, real, multi, batch, real B6, M-d1
diis and ppl; within the CPU budgets, now about 51 of 90 and 27 of 30); `repro_check.py`
fails on a field only the stored record has, unless allowed by `--allow-missing`; the
re-run driver committed as immerdink-eval `scripts/c1/repro.sh`; two wording nits in the
findings and `b6_split_cost.py`. Its verdicts, binding:
- **S2-19. `self_test_modes`' coverage** (P = 2 standing for P ≥ 2 in the X-role and
  FP64-clamp checks, guard 0 and 1 required at each P = 1, 2, 3, the clamp not required to
  bind at exactly P = 3, G undefined left to test_oracle (12)). *Verdict: confirmed, with
  one binding condition.* The clamp and the X-role attribution depend on P only through the
  union extrema and the loop over pairs, the same path for P = 2 and 3; the guard's minimum
  over pairs depends on P and is asserted at each. The reviewer's mutation checks trip the
  coverage assert, and none of the 1451 weight vectors in M-a has an undefined G. *Binding:*
  before any record from a fixture that can produce G undefined is used (an all-zero slice in
  a batched descriptor, a zero operand), `self_test_modes` adds such a case and asserts it.
- **S2-20. `c2_summary` reports INCOMPLETE** only when (i) is undetermined and (ii) and (iii)
  both pass; a failing (ii) or (iii) still gives FAILS. *Verdict: confirmed* (checked by
  perturbing the stored records). The (i) INCOMPLETE line stays in the output even when the
  verdict is FAILS, so a revival assessment sees the undetermined contraction; it must stay
  that way.

**Third re-review** (independent reviewer, 2026-10-07; approve-with-fixes). One must-fix:
"Gates (i) and (ii) rest on reproduced records only" (Status and the findings, from the
second re-review's fixes) overstated the provenance. Both gates are evaluated at d* = 3, and
d* comes from Tier S, which is not re-run; Tier L and the real records compute only d = 0
and 3, so another d* could not even be evaluated on them. The verdict does not depend on it
(checked by the reviewer and again on `repro_5925219/`): at d = 0 every R1–R5 record has
guard14 = 0, R6 and R7 have Δ ≡ 0, and gate (ii) fails on the same 16 guard-accepted Tier-L
records, 1–14 outputs above 2× per fixture, max ratio 99.1. Fixed: Status, stage 2,
*What gates (i) and (ii) rest on*, three statements (the records are reproduced; d* is
not; the verdict holds at both computed deadbands), and the same in the findings. Its
should-fix items, applied: jacobi's reproduction is credited to the 928b49c spot check, the
rest to 5925219 (the commit messages of 04363db and immerdink-eval beb07e7 said all of it
at 5925219); `repro_check.py` reports duplicate identity keys in either file and the stored
records with no re-run (coverage "N of N"); the first round's CPU tally reads about 36
(the findings' "37" did not add up); and the stage-1 items (§5.5 S1-7). Its
verdicts, binding:
- **S2-21. Tier S is not re-run in full** (about 24 CPU-min, which would put M-a at about 75
  of 90). *Verdict: not now (binding); Tier S stays labelled as not reproduced, under
  conditions.* Tier S has a hard sub-budget of 25 CPU-min, about 24 already spent, so a full
  re-run would break it although the M-a total fits. Tier S feeds only d* = 3 (a tie-break:
  every deadband rejected all 30 random-structure runs) and its share of gate (iii), which
  is met; the stage-4 verdict does not depend on it, since (i) and (ii) fail at both
  computed deadbands on reproduced records. *Binding:* NOTES and the findings keep Tier S,
  d* and the trend over m labelled "not reproduced on committed code (8b58101-era,
  smoke-checked 4 of 245)". Before any stage-4 revival or any paper or external use of Tier S
  figures, the d*-deciding subset is re-run on committed code under a newly declared budget:
  m = 512, the random-structure fixtures plus decay_opp and opp_nonsep, 5 seeds. (The
  verdict wrote "stage-4 revival (T1–T3)"; T1–T3 are stage 5's triggers, §4.6, and stage 4's
  revival trigger is "a real-operand family meeting (i)", §4.5. The condition is read as
  binding on a stage-4 revival, whatever triggers it.)
- **S2-22. The 4 `md1_rho` records (IR7, d89988a, content-tied) are not re-run**, to keep
  M-d1 under its hard 30 CPU-min. *Verdict: accepted (binding).* They used `precision.py`
  sha256 574c404 (51dba1e); between 51dba1e and 5925219, `rho_from_output` changed only in
  argument validation and docstrings, the numeric recipe is unchanged, and so is the library
  (sha256 466acdb4…). `md1_converge.rho_hat` passes 1-D bounds on a 2-D D, a path MF3 and
  MF4 leave alone (and float64 bounds, so the third review's exact conversion leaves it
  alone too). The 1-of-228 ρ-miss result still describes the shipped helper; a re-run would
  put M-d1 at about 28.6 of 30 on an approximate tally.

### 5.7 Stage 3: the implementation's design questions (independent reviewer, 2026-10-07)

The stage-3 implementation raised the questions below. Each was decided conservatively, in the
spirit of §4.4, implemented and documented, and put to an independent reviewer, whose verdicts
are binding. The review was approve-with-fixes: two must-fix items (a K-empty band's dispatch
in `contract_segments`, MF1; the unsound ELEMENTWISE ρ_S source in DESIGN §3.8, MF2) and ten
should-fix items, all applied (Status, stage 3).
- **S3-1. A request transform for the recipe (L7d).** §4.4 gives L7's bound but not how a
  caller turns a target into N. Decided: DESIGN §3.8 states one rigorous route through the
  existing API. The caller forms t_eff from per-band norm statistics (t/r NORMWISE relative to
  the plain norms, t/ρ_S ELEMENTWISE, t/M̄ ABSOLUTE), computes
  `t′ = RN(RN(min(t_eff, 1)(1 − 2⁻⁵⁰)) − 4S·2⁻⁵³)` in FP64, and requests NORMWISE t′ with
  ℓ1 ratios 0. PROOFS L7d proves that the chosen N meets the target; `cert_kseg_composition`
  checks the FP evaluation in exact rationals on 2137 random (t, r, S); Lean has its exact half.
  No helper ships (Q15 (a): no API); `params.kseg_request` is the reference. Alternative not
  taken: no auto-N guidance (a caller could then only use the unsegmented ν = √k request, which
  certifies the segmented result but never sees its gain). *Verdict: approved with
  conditions, all applied in DESIGN §3.8:* (a) the ELEMENTWISE ρ_S source is rigorous
  (N_S/(|C_S| − W̄·N_S) from a previous segmented output, rounded outward, certified only for
  a finite, normal C_S, or the plain ρ̂ of an unsegmented β = 0 probe, which gives up the
  gain; either predicts the next call, it does not bound it); (b) the guarantee is W ≤ t̄,
  the exact quotient, not ≤ t_eff; (c) the transform holds for α = 1 or 2^e and β = 0, with r
  and ρ_S rounded upward; (d) t′ ≤ 0 means no N is certified. The 4S·2⁻⁵³ margin, about twice
  what the proof needs, stays. *Final review of C1/C2 (must-fix, applied):* in case (d)
  `params.kseg_request` returns None, not 0.0, since rank reads a target of 0 as "no target,
  report only" and would return its cheapest configuration unfiltered; this happens at the
  normwise ~1e−15 targets §3.8 discusses (t_eff = 5e−16 or 8e−16 at S = 2, 1e−15 at S = 4).
  Step 3 requests t′ > 0 with basis BOUND (TYPICAL voids the certification).
- **S3-2. `contract_segments`' signature.** §4.4 names (A, B, C, N, rounding, bounds). Decided:
  `bounds` is the boundary list [0, k_1, …, k], default two near-equal bands
  (`segment_bounds(k, 2)`, numpy's array_split as `c2_gate.segmented`); and `alpha`, `beta`,
  `g`, `chunk` and `trace` are passed through, band 1 carrying the caller's β and the later
  bands β = 1, a literal composition of the calls a caller makes. An empty band or bounds
  that do not partition [0, k) are INVALID_VALUE; a band's own RN rejection propagates.
  *Verdict: approved, with a binding amendment, applied:* each band follows the library's
  dispatch, so a band of k_s = 1 is K-empty and runs DIRECT (one fma, error ≤ u|x_s|, so L7
  holds with E_s = 0); with α = 0, A and B are not required (contract()'s convention). No
  mode parameter: the recipe is an EMULATE recipe, as the docstring says.
- **S3-3. Per-band N and packs.** §2.7 says "S calls of one plan". The recipe allows each band
  its own N and, for unequal k_s, one plan per distinct k_s (L7 takes E_s per band, so the
  bound holds unchanged); packs are per band sub-view, keyed by k_s. *Verdict: approved.*
  §2.7's "one plan" is the equal-k_s case; requesting each band's N on its own sub-view with
  the same t′ is sound, since each W_s ≤ t̄ on its own; packs are keyed by k_s, and by N_s
  where N differs (§6b).
- **S3-4. The §7 rows beyond §4.4's list.** Added: "K-equilibration in the library" (closed by
  its gate, with the trigger, since stage 4's outcome belongs in §7 under IR29) and
  "K-segmentation as a library mode" (Q15). The rows cite M-a and are final, not "provisional
  until M-a". The larger-moduli row cites arXiv:2603.10634 after checking it: Uchino, Ozaki
  and Imamura, Ozaki-II with Karatsuba-split FP8 residues, 3 FP8 GEMMs per modulus and about
  2.5× the multiplications of INT8 Ozaki-II for 53 bits. That is an FP8 fact cited as the
  analogue; the INT8 statement stays [I]. *Verdict: approved.* IR29 puts every gated outcome
  in §7 or the ROADMAP with a trigger, and IR12 allows final wording citing M-a; the reviewer
  checked the cited M-a figures against findings/c1-evidence.md and the citation itself.
- **S3-5. The certificate goes beyond §4.4's list.** The ν counterexample shows an actual
  violation of the band bound with the whole row's ν (3.0× at N = 3), not only ν_s > ν; the
  certificate also shows the lost single rounding (an S = 2 output at N = 20 errs by 2.9 u|x|,
  beyond the bound the plain call meets) and that γ is necessary (the bound without it fails
  on rows aligned band by band). Underflowing products are checked in the ABSOLUTE form with
  the (1+γ)·S·2⁻¹⁰⁷⁵ term. *Verdict: approved*; the reviewer reproduced it bit for bit and
  confirmed by mutation that it fails without γ and without the request's margin. The
  tautological S = 1 check now compares W_1 with `error_estimate`'s own NORMWISE bound.
- **S3-6. Lean scope.** IR5 lists L3c, its compositions, the final rounding and γ_k before L7.
  Also proved: the native chain's γ_k (L3c (3)) as a corollary of the summation lemma, E's
  monotonicity (L7c (4)) and `kseg_rows`, which chains L7 from band rows through L3c (1).
  L3c's final rounding uses `|D − y| ≤ u|y|` (L3c's text); L3b keeps `|D − y| ≤ u|D|`.
  *Verdict: approved*: both are valid RN facts in the normal range, u|y| is what DESIGN
  §6f.2 and PROOFS L3c state, and the extra theorems cover claims DESIGN and PROOFS make.
  IR5's Lean condition for "C2 done" is met.
- **S3-7. HANDOFF.** docs/HANDOFF.md tracked state up to v1.18 and no C1/C2 work; one bullet
  now points to this file's Status for stages 0–3. *Verdict: approved* (one pointer bullet
  with the open items, no duplicated state).

### 5.8 M-b: the design questions (independent reviewer, 2026-10-07)

M-b (§4.3) raised the five questions below. Each was decided conservatively and put to an
independent reviewer, whose verdicts are binding. The review was approve-with-fixes: one
must-fix item and seven should-fix items, all applied (Status, M-b). The must-fix: the raise
cost (shared against separate packs) is a difference of two overheads, not what a pack serving
every N could save, so the sentence "a slice revival that avoids the raise would save at most
that" (§3.3) and DESIGN §7's use of the raise figures as a revival's bar were wrong. §3.3 and
§7 now give that bound from the records, with its pricing stated: 0 in all 12 forcing-rule
cases, 0–9.1 % one level apart and up to 16.0 % in the synthetic floor patterns [M,
model-only]. The reviewer's ideal priced the packs at the shared plan's N (separate packs then
0.19 % above it in the twelfth case); `m_b.py` prices each pack at its own group's top floor,
which gives the larger, more conservative bound, and also counts the best separate
configuration, which in the twelfth case is already below that ideal.
- **MB-1. §2.7's K-segmentation cost, [E] against [M].** The first-round [E] (+17 / +31 / +10 %
  at S = 2, +51 / +93 / +30 % at S = 4) lies above the model's own composition of the S calls
  (+13 / +24 / +8 %, +40 / +72 / +24 %). Decided: keep both. *Verdict: keep both, as
  committed in DESIGN §3.8 and §2.7.* The [M] composition is the reference figure for any
  decision; the [E] stays as the first-round DRAM-roofline cross-check; both are replaced only
  when a GPU measurement of the recipe exists. The [M] composition is the method of every
  other K-seg [M] figure (`c2_gate.perf_block`, DESIGN §3.8's 18.8 against 17.4 TFLOPS-eq) and
  prices what the [E] leaves out (K0 per call, the β = 1 read, the W planes' L2 residency); the
  [E]'s S = 4 row is its own linear formula on rounded S = 2 inputs, an upper-side estimate, not
  an error. No gate depends on these figures.
- **MB-2. The 0.5 % rule.** Seven figures differed in the last printed digit only and were
  left as cited. *Verdict: correct them.* IR9's threshold says when a changed prediction forces
  a replacement; it is no reason to keep a printed value that is not the correct rounding of an
  unchanged one. All seven lay just below a tie (24.348, 82.848, 47.449, 30.849 twice, 18.8945,
  1.19504). Applied: §2.3 and §2.6, with a note.
- **MB-3. The Boo share group.** {hh, R5}, with L2's Boo role (another pack key) left in-line;
  only groups whose members have equal pack keys. *Verdict: accepted; check the key digests at
  every level used.* The library rejects a pack read under another key (PACK_MISMATCH).
  Applied: `m_b.py` now checks every level 6–14, which covers the default grid and the two
  grids of the level-grid check: one key in every group at every level, on every molecule.
- **MB-4. The raise-cost baseline.** Members at equal floors still share one pack per
  (operand, floor), plus a "best separate" variant where a lone member may stay in-line.
  *Verdict: accepted; report both.* Equal-floor sharing is the correct reading of IR15 (3)'s
  comparator (making equal floors pay separate packs would only inflate the no-raise cost), and
  "best separate" adds the in-line option `PrecisionPlan` already ranks. Neither variant is the
  bar for a slice revival (the must-fix).
- **MB-5. Contraction set and scenarios.** ccsd.py's two-operand DF contractions, R2 at
  cc-pVDZ only, equal weights in the forcing rule, the invariant Bov re-packed once per
  4-iteration phase. *Verdict: accepted as M-b's scope; state the caveats next to the
  figures:* the B roles of the one-factor contractions modelled in-line, the bare norms, the
  level grid, and R2's rationale. Applied in §3.3, DESIGN §6g.10 and §7, and the findings.

### 5.9 M-c: the design questions (independent reviewer, 2026-10-07)

Evidence: Status "M-c"; immerdink-eval `findings/c1-m-c.md`. M-c (§4.3) raised the five
questions MC-1 to MC-5 below; MC-6 came up while applying the first review. Each was decided
conservatively and put to an independent reviewer, whose verdicts are binding. The review was approve-with-fixes: three must-fix items and six
should-fix items, all applied (Status, "M-c", "Review"). The must-fix items:
- the cause was misstated. The first write-up blamed the Jacobi view's REPEAT credit alone ("a
  shape, not N"), but the missed median is exactly the mean of the two sq1024 N = 4 calls
  (15.8 %, repeat 16.1 %), a small-N optimism on the KF path. Each effect is necessary:
  without either one the median would be met (Status, "Cause");
- "the anchors included" was wrong for the Jacobi view's 26–29 %: only the anchor N = 8 is
  ≤ 10, and at N = 12–14 (the anchor N = 14 included) the error is −14 to −15 %;
- the verdicts below are recorded here and in Status, §4.3, DESIGN §6g.9 and §6g.10, the
  ROADMAP and HANDOFF.

A second review of those fixes (approve-with-fixes, 2026-10-07) found that the account of the
Jacobi view held for call totals only. The reviewer checked the phase marks of `mc.jsonl`,
`calib_n.jsonl` and the calibration records against the shipped calibration's per-phase
REPEAT and COLD predictions (CPU, one thread). They show two KL effects whose errors cancel
at the total (Status, "Cause"; immerdink-eval `results/c1/m_c/phases.md` reproduces them at
full precision, both runs). Its must-fix items: correct that account everywhere (MF-A), and
record the verdicts of the second review (MF-B: MC-3a and the ruling on the hold-out's
overlaps, MC-6). Its verdicts are in each item below and under "Second review" at the end
of this section; they are binding.

- **MC-1. Is the Q24 refit `n1` adopted?**
  - What `n1` is: the shipped fit's records plus the 43 N-specific calls, with the same
    `perf_fit.py` arguments. The baseline refit reproduces the shipped parameters to rounding.
  - What it does: it meets M-c's first run (< 100 µs median 14.7 %; ≥ 1 ms p90 14.0 %) and
    keeps the product hold-out (3.9 / 10.7, 3.1 / 7.4, 2.3 / 7.4 %). That 14.7 % is not
    hold-out evidence: the Q24 cases were chosen from M-c's residuals, and `q_kl2`
    (1296 × 144 × 1296, `abcd,cdij->abij`) is in the Jacobi view's own class.
  - What it changes (diff of `refit/base.calib` and `refit/n1.calib`; every parameter that
    moves by 5 % or more, plus the K4 byte rates and the power terms; corrected at the second
    review, which found two missing and the byte rates misstated):
    - KL: `kl.wave` 2.16 → 4.45 µs (×2.06), `kl.launch` −11 %, `kl.byte` −24 %, `kl.l2byte`
      −44 %, `kl.op` 0 → 0.15 ps/op.
    - The call term: `call.call` 1.02 → 0.38 µs (−63 %), `call.launch` −28 %.
    - K4: its effective L2 `l2_bytes` 30.2 → 22.6 MB (0.4 → 0.3 × nominal);
      `k4_64x128.mac1` −42 %, `mac2` −9.9 %, `round` +15 %; `k4_64x64.mac_ktail` +56 %,
      `mac2` and `mac3` +26 %; `k4_128x256.mac_ktail` +9.3 %. The byte rates:
      `k4_64x128.byte` +5.1 %, `k4_128x256.byte` and `k4_128x128.byte` −4.9 %,
      `k4_64x64.byte` −9.5 %.
    - K5's `dram` −17 %; the power model's `clock_eff_mhz` 2479 → 2518, `act_w` +5.7 %,
      `tc_pj` and `alu_pj` +3.9 %.
    - Every other parameter moves by less than 5 %.
  - Implementer's decision: not adopted. Its pass is not robust (it misses on the repeat run of
    the small calls, 15.0 %); it fits a cache-state error with path parameters, the per-shape
    compensation §6c.5 rules out; it moves the PPL view, an M-d2 gated system, to +11 … +20 %;
    and shipping it would change every [M] figure M-b certified.
  - ***Verdict: NO, upheld (binding).*** The shipped calibration (`d31f62ca…`) stays.
    `n1.calib` stays in immerdink-eval only and is never committed to the product repo.
    *Rationale:* n1 passes the < 100 µs median by 0.3 points and fails on the repeat; its M-c
    reading is contaminated anyway (the case choice above). It fits a cache-state error with
    path parameters, which §6c.5 forbids ("never per-shape factors"). It degrades the ≥ 1 ms
    class: M-c's p90 7.3 → 14.0 % (max 20.3 %), and the PPL view moves to +11 … +20 %. It
    would also change every [M] figure M-b certified.
  - ***Second review: NO, UPHELD (binding), the rationale corrected.*** n1's `kl.wave` ×2.06
    (2160 → 4453 ns) partly fits a real KL path error: KL B on narrow packs takes ×1.5–2.0
    its prediction in every cache state. So "it fits a cache-state error with path
    parameters" is only half right. NO stands on the other grounds: it misses on the repeat
    (15.0 %); its case choice comes from M-c's residuals; it moves the PPL view to
    +11.2 … +20.3 % and M-c's ≥ 1 ms p90 to 14.0 %; and it would change M-b's [M] figures.
    `n1.calib` stays out of the product repo.
- **MC-2. Does a miss the anchors share fail M-c? Should the marks be lifted for the
  ≥ 100 µs classes?**
  - Implementer's decision: yes, it fails, as IR4 is written ("judged per size class"). The
    question to the reviewer was whether M-c answers its N question, which could lift the
    marks for the ≥ 100 µs classes.
  - ***Verdict: YES, M-c FAILED (binding).*** No partial pass and no per-class lifting of the
    [M, model-only, outside hold-out] marks. DESIGN may cite M-c's measured per-shape
    agreement next to the marks (as §6g.9 does), labeled indicative.
    *Rationale:* IR4 judges each class over all calls; the done condition requires "M-c has
    passed", and a partial pass is not defined. The premise of the alternative, that M-c
    cleanly answers the N question, does not hold: the missed median is set by the two sq1024
    N = 4 calls, a small-N KF optimism 16–21 points below the anchors and reproduced in the
    repeat; the ≥ 1 ms per-N median is 5.4 % at N = 9 (skinny is +6.5 % there through a K4
    step); the Jacobi error persists at the anchor N = 14 (−14 %). Lifting marks on a failed
    gate would be the opposite of conservative.
    *Reading (second review):* "16–21 points" is against the mean of sq1024's anchors'
    errors; against the N = 8 anchor alone it is 8.2 / 15.4 points (repeat 12.8 / 6.8),
    since sq1024's error rises with N (Status, "Cause").
  - ***Second review: UPHELD (binding).*** Nothing in the phase marks changes the call-level
    verdict: the class median missed on both runs. M-d2 waits until the MC-3 path closes.
    The done condition stays as amended, read as "with the MC-3 feature(s)" (§4.11).
- **MC-3. The fix is a model feature beyond Q24. Build it?**
  - *What is wrong* (corrected at the second review: two KL effects, not one; Status,
    "Cause"):
    - *KL A keeps only part of its REPEAT saving.* The REPEAT residency rule
      `res = clamp(L_w − (F − I), 0, I)` (DESIGN §6c.1 "Cache state") is a capacity rule.
      On the Jacobi view KL A keeps 27–50 % of its modeled 26.8 µs credit at N ≤ 10 (Q24's
      KL cases 5–52 %, the in-sample `c_skin1280` and `c_k3hit` 14–33 %). F = I + workspace
      + 8bmn already holds the workspace, the KL residue planes included, so the earlier
      candidate "count the KL pack planes in the REPEAT footprint" (a382ef6) would count
      them twice. `warm.pack_l2_planes`, the precedent, adds K4's W planes,
      g·b·m·n̂·(2·k_chunks − 1) per group, to PACK's set S: that is K4's traffic between two
      re-reads, not a capacity. The mechanism is [I]: a KL + K4 call moves more bytes
      through the L2 between one call's read of A and the next call's than its footprint
      says.
    - *KL B, the narrow operand's pack, is slow in every cache state.* On the Jacobi view
      (1600 × 100, 1.28 MB, credit ≤ 1.7 µs) it takes ×1.65–1.89 its REPEAT and ×1.49–1.72
      its COLD prediction at N ≤ 10 (×1.94–2.02 / ×1.77–1.85 at N = 12–14); on Q24's KL
      cases ×1.50–1.81 / ×1.31–1.65. It is 47–59 % of the Jacobi call's miss at N ≤ 10 and
      92–105 % at N = 12–14. This is a KL path error, not a cache effect; its mechanism is
      unknown.
    - At the call total the two cancel against the COLD prediction (−1.5 to +10 %), which is
      why the first account read the Jacobi view as "no saving at all".
  - *Why no setting of `warm.l2_bytes` or `warm.l2byte` can work* (stronger than the
    un-refit probe of `probe_cache_keys.txt`; premise corrected at the second review, the
    conclusion holds): sq1024's KF phase keeps 99–157 % of its modeled credit at a modeled
    F − I of 49.3 / 52.5 MB (N = 13 / 14; modeled resident inputs 14.8 / 11.7 MB, predicted
    +4.7 / +10.6 % slow), so it needs about 15–18 MB credited (14.8–18.4 MB over both runs
    and both data sets). The Jacobi view's KL A keeps 27–50 % of its credit at F − I =
    13.0–30.5 MB (N = 4–10; 24.7 MB at N = 8; modeled resident 21.8 MB, 20.5 MB of it KL A's
    read), so it needs about 6–10 MB credited (5.5–10.3 MB). A rule in F − I alone that
    credits sq1024 15–18 MB at 49–53 MB credits the Jacobi view at least as much at
    13–31 MB, so a capacity-only rule cannot separate them. (The first version said the
    Jacobi view "loses its whole saving".)
  - *Candidates* (named, not chosen; the choice belongs to (1) below): (a) traffic: add K4's
    split-K W-plane traffic, Σ over groups of g_i·b·m·n̂·(2·k_chunks − 1), to F − I in
    `res_repeat`, as `warm.pack_l2_planes` adds it to S; (b) path: no REPEAT credit for the
    input bytes a KL pack reads (KL path), the KF credit kept.
  - Implementer's decision: not built in M-c, recorded as NOTES-B2 §12 "Remaining".
  - ***Verdict: AUTHORIZED as the MC-3 recalibration under DESIGN §6c.5's feature rule, not
    under Q24 (binding).*** It is the only path to an M-c pass. C1's done condition now reads
    "M-c has passed after a recalibration per Q24 or with the MC-3 feature" (§4.11); nothing
    else changes. Requirements, in order:
    1. DESIGN §6c text first. It names the discriminating quantity (for example K4's split-K
       W-plane traffic, or the KL / KF path), chosen only from the calibration, cache, Q24
       and v1.18 hold-out records, never from `mc.jsonl`. The mechanism stays [I] until
       measured.
    2. One optional key in format 2, absent = 0, so every existing file predicts
       bit-identically. `host_perf` tests cover the absent key and the rule on a KL narrow-n
       case and on a KF case.
    3. A refit with `perf_fit.py` on the shipped records plus `calib_n.jsonl`; a file must not
       carry the key without being refitted with it.
    4. A pass means every §6c.5 class target is met on all of: the v1.18 product hold-out;
       the existing `mc.jsonl`, with and without the repeat's calls; and a fresh full
       160-call M-c run (same case list and method, busy-gated chunks of ≤ 5 min).
    5. If the calibration changes: re-run M-b's [M] figures, then re-derive the M-d2 bars and
       pre-register (commit) them before M-d2.

    If MC-3 is not pursued, M-c stays failed and C1 is not done.
    *Rationale:* Q24 excludes features, and the records show that no calibration key can work
    (the KF calls keep their saving at a larger modeled footprint than the Jacobi view's
    KL A loses most of it at). DESIGN §6c.5 already prescribes "the smallest physically motivated feature,
    additive optional key" for a missed target. The defect is real and planner-relevant:
    12–35 % on narrow-n KL calls (M-c's Jacobi view, Q24's `q_kl1`, `q_kl2`, `q_kl4`), and
    6–20 % through KL A alone on the in-sample `c_skin1280` (−20.4 / −10.6 %) and `c_k3hit`
    (−10.8 / −6.4 %). (Corrected at the second review: the first version said "about
    26–35 % … including the in-sample `c_skin1280`".) Keeping the feature's selection off the
    M-c records and measuring M-c again keeps the pass honest.
  - ***Second review: AMENDED as MC-3a (binding); everything else in MC-3 stands.***
    1. *The §6c text from a per-phase account.* The DESIGN §6c text is built from a
       per-phase decomposition (the phase marks against the REPEAT and COLD predictions per
       phase) of the allowed records only: the calibration, cache, `calib_n.jsonl` and
       v1.18 hold-out records, never `mc.jsonl`. It names both KL effects: KL A's partial
       REPEAT saving (5–52 % kept on Q24, 14–33 % on the calibration records), and the
       narrow-operand pack KL B taking ×1.5–1.8 its prediction in every cache state. It
       also covers KF's credit against N (`q_kf1` and the cache sweep's `x_s1024`; see the
       observation below).
    2. *The cache key is fitted against the phase marks.* It is chosen and fitted against
       the KL A and KF phase marks, not against call totals alone. A key whose call-level
       fit relies on KL A cancelling KL B does not meet §6c.5's "physically motivated" rule
       and does not count. For example, candidate (b) as zero credit puts KL A at its COLD
       prediction, 2–33 % slow on the allowed records.
    3. *At most two optional keys.* MC-3 may carry at most one optional key per physically
       distinct effect, two in all: the REPEAT credit, and the KL narrow-pack time. Each has
       absent = 0, `host_perf` tests and DESIGN text. If KL B is not modeled, it is recorded
       as an open model defect, with its size and a trigger, in NOTES-B2 §12 and DESIGN §7.
    4. *Three more pass conditions.*
       - KL A's per-phase median |error| on the allowed narrow-KL records must improve.
       - No phase class's in-sample median or p90 (`perf_fit`'s table) may worsen by more
         than 1 point. KL A, KL B, KF and K4 are reported before and after, also on M-c's
         Jacobi calls. **For future attempts (MC-3a review, 2026-10-09, binding, §5.11):
         "before" means the shipped fit's `perf_fit` table, as written, and the records a
         candidate has to add to fit what it models are themselves charged to the candidate
         (what they do to other classes counts against it). A same-records reading may be
         reported and is not the condition.**
       - A fresh narrow-KL hold-out must meet every §6c.5 class target. It is committed,
         with a `sweep_disjoint.py` check against `sweep_calib.txt`, `sweep_calib_n.txt`,
         `sweep_mc.txt` and `sweep_perf_validate.txt`, before the refit: at least 3 new
         shapes of the class (KL path, n ≤ 160, inputs inside the REPEAT window) and at least
         1 KF shape of similar footprint, at N ∈ {4, 8, 13}, on both data sets; about 1 min
         of GPU in busy-gated chunks.
    5. *The v1.18 hold-out is judged on both sets* (MC-6 below).
    6. *Run-time provenance of every fresh GPU chunk* (final review of C1/C2, should-fix,
       applied before any MC-3a run). `imd_bench`'s records carry `commit` but no library
       sha256, host API or dirty flag (`m_c/mc.jsonl` was made by a profiling build linked
       34 s before 530ff81 was committed; its hashes were supplied afterwards, and they still
       match the files on disk). immerdink-eval `scripts/c1/m_c_chunk.sh` now appends to each
       chunk's `logs/<chunk>/provenance.txt`, after the chunk: the commit, the dirty flag,
       the sha256 of `imd_bench` and of the `libimmerdink` libraries it loads, and the host
       API. MC-3a's hold-out and refit runs, and the fresh M-c run, go through it (or
       record the same fields at run time).

    *Outcome (2026-10-09): §5.11.*

    *Rationale:* the v1.18 hold-out has no case of this class (its narrow-n cases `v_rsm1`,
    `v_rsm2`, `v_r_mm`, `v_mmg` and `y_k3a` are outside the REPEAT window or not this
    shape). M-c's Jacobi view is the shape that motivated the feature, and the Q24 cases were
    chosen from M-c's residuals. Without these conditions an M-c pass could be a phase
    cancellation measured on its own motivating shape.
  - *Observation for the MC-3a §6c text (second review):* sq1024's "small-N KF optimism"
    sits in the KF phase. KF keeps about 74 % of its modeled credit at N = 4 (measured
    27.2 µs, REPEAT 21.6, COLD 43.6) and about 157 % at N = 14 (35.7 / 44.5 / 59.8 µs);
    `q_kf1` keeps 80 % at N = 4 and 163 % at N = 12. It may be the same cache-rule shape
    and not an N effect. MC-3a examines it on `q_kf1` and the cache sweep's `x_s1024`
    records, not on `mc.jsonl`.
- **MC-4. The squares' spec.** Implementer's decision: `ik,kj->ij`, the spec of §2.3's [M]
  figures (M-b's `SH`), not the bench's usual `ik,jk->ij` (§2.3 notes a 3 % spec effect).
  ***Verdict: UPHELD (binding).*** *Rationale:* these are the specs of the cited §2.3 / §2.4
  [M] figures. The disjointness check works on GEMM views and treats an A / B swap as a match,
  so the spec cannot hide a calibration shape. The reviewer re-ran `sweep_disjoint.py`: all
  16 M-c cases are absent from `sweep_calib.txt`, and all 7 Q24 shapes are absent from the
  hold-out and M-c. Both now hold in ctest (`sweep_disjoint_selftest`).
- **MC-5. decay_opp data; absence checked against `sweep_calib.txt` only.** Implementer's
  decision: `imd_bench --data decay_opp` (c2_fixtures.py's family on the device), since no
  existing mode decays along K; bench-only, committed before the run.
  ***Verdict: UPHELD (binding).*** *Rationale:* `k_fill_decay` / `decay_map` match
  `c2_fixtures.synthetic('decay_opp')`: magnitude ±U(0.5, 1)·10^(−1 − 9h − 3r/(R − 1)), h
  row-major over the canonical K order in A and 1 − h in B, r over each operand's non-K modes.
  It is bench-only, committed (530ff81) before the run, and the only `imd_bench.cu` change
  since the calibration's ec44099; kernel sources are unchanged. It has the intended effect
  (up to 5.8 % faster at N = 4 on the large shapes). The decision requires absence from the
  calibration set only; the listed overlaps with the hold-out, cache and sustained sweeps
  (1024³, 4096³) are acceptable.
- **M-d2 ordering after a failed M-c** (implicit in the report).
  ***Verdict: M-d2 WAITS (binding).*** It runs only after the MC-3 path closes. If the
  calibration changes, the bars are re-derived and committed before M-d2 (IR23). If MC-3 is
  closed without a calibration change, M-d2 runs on the pre-registered bars unchanged: 0.750
  (PPL view) / 0.664 (DF) / 0.673 (4096³), ± the §6c.5 p90 of the class, and ≤ 0.8. M-c's
  indicative measured ratios must never be used to set or adjust the bars.
  *Rationale:* IR23 orders "M-c, and its recalibration if any, strictly before M-d2", and
  MC-3 is now that recalibration. Running M-d2 first would put the evidence out of order, or
  give two M-d2 results to choose between.
- **MC-6. The v1.18 hold-out's overlaps with the fit** (raised after the first review, for
  MC-3 step 4). The v1.18 product hold-out (`bench/sweep_perf_validate.txt`) is disjoint
  from `sweep_calib.txt` by spec and extents, not by GEMM view. `sweep_disjoint.py` finds 7
  of its 76 cases sharing a view with a fit case:
  - `v_s512` (in-line 512³ at N = 8, 14): `c_tcc8` is a tensor with that view, in-line at
    N = 8, 14, 20 (`c_dir1` is DIRECT at N = 8);
  - `w_s512` (512³ at N = 15–17): `c_tcc8` and `c_dir1` again, at other N;
  - `v_cc12` (576³ at N = 8, 14): `k_s576` at N = 15–17;
  - `v_pks` (packed 384³ at N = 8): `c_s384` in-line at N = 8, 14, 20;
  - `v_n15e` (8 × 512² × 2048 at N = 15–17): `c_tikl2` at N = 8, 14;
  - `v_n18a` (2304³ at N = 18, 20): `c_tcc2` at N = 8, 14, 20;
  - `v_mmg` (in-line 1500 × 96 × 40000 at N = 8, 14, 16): `c_pkmm` packed at N = 8, 14.

  4 of the 7 share both view and N with a fit sample, not 2 as the first version implied
  ("each pair differs in method, in N, or as a tensor against a square"):
  `perf_fit.samples_from_records` and `perf_validate` take the `plain_` contract of a packed
  record as an in-line call (COLD state). So `c_pkmm`'s `plain_` (1500 × 96 × 40000,
  N = 8, 14) is an in-line fit sample with the view and N of `v_mmg`; `v_pks`'s own `plain_`
  (384³, N = 8) is evaluated with the view and N of `c_s384`; and `v_s512` against `c_tcc8`
  (N = 8, 14) and `v_n18a` against `c_tcc2` (N = 20) are in-line on both sides.
  - Implementer's decision: the hold-out unchanged; MC-3's report also gives its statistics
    without these 7 cases, reported, not judged.
  - ***Verdict: RULED (binding): the hold-out stays unchanged, but MC-3 step 4 is judged on
    both sets.*** Both must meet every §6c.5 class target: (i) all 76 cases, and (ii) the
    69 cases whose GEMM view matches no fit case. *Rationale:* the v1.18 hold-out stays the
    product's established hold-out, so its history keeps its meaning. Reporting (ii) without
    judging it is not conservative enough; 4 of the 7 share view and N with a fit sample,
    and dropping them costs nothing and removes the doubt. The reviewer confirmed with
    `sweep_disjoint.py`: 7 present against `sweep_calib.txt`; the 4 matches against
    `sweep_mc.txt` are the 1024³ and 4096³ overlaps accepted in MC-5.
  - *Counts and the shipped baseline* (immerdink-eval `results/c1/m_c/phases.md`): 3 of the
    76 cases (`v_x1280`, `v_x2048`, `v_x3072`) are cache pairs, judged by the pair report and
    not in the class statistics. So set (i) is 73 cases and 199 calls in the class
    statistics, set (ii) 66 cases and 171 calls. The shipped calibration meets both: (i)
    3.8 / 10.7, 2.7 / 8.2, 2.5 / 6.4 %; (ii) 3.8 / 10.1, 2.8 / 8.3, 2.2 / 4.5 % (median /
    p90 for < 100 µs, 100 µs – 1 ms, ≥ 1 ms).
- **Second review, the remaining verdicts** (approve-with-fixes, 2026-10-07; binding):
  - The first review's must-fix items: RESOLVED. The cause as two effects at call level
    (the phase level is MF-A, applied in Status "Cause" and MC-3); the Jacobi view at
    N ≤ 10 against N = 12–14; the verdicts recorded everywhere. The reviewer re-ran the
    self-test (29 of 29; the no-swap mutant fails 27 of 29) and found the ctest log and the
    library sha256s (`466acdb4…`, `c4bdcc4f…`) as claimed.
  - The implementer's question on the reviewer's figures ("about 5 %" is the repeat's 5.0 %,
    the first run 3.4 %; "16–21 points" is against the mean of sq1024's anchors' errors):
    ACCEPTED. Citing both runs is right; the baseline and the N = 8-anchor figure are now
    given (MC-2 "Reading"). The counterfactual does not depend on the baseline.
  - Should-fix items applied: the exact n1 change list (MC-1); the overlaps that share N
    (MC-6); the anchors' baseline wherever "16–21 points" appears; the Jacobi error
    reproduces run to run within 1.5 points at full precision (`m_jacd` at N = 10: −28.3 %
    against −26.7 %, 1.6 in the rounded figures), 19 of 20 calls within 1 point (the first
    version said "within 1 point"); the
    `sweep_disjoint_selftest` comment in `tests/CMakeLists.txt` now names the Q24 checks;
    the KF observation (MC-3a above).

### 5.10 M-d2: the protocol's design questions (implementer and independent reviewer, 2026-10-07)

Evidence: Status "M-d2"; §4.3 "Pre-registered protocol"; `bench/accuracy/md2_time.py`
(3254b15, re-registered at daf3583 and again at 35eeb46); immerdink-eval
`findings/c1-m-d2.md`. The implementer decided each question conservatively, and an
independent reviewer then ruled on each; the verdicts are binding. The review was
**approve-with-fixes**, and every fix is applied at daf3583 (the driver) and in the docs.
A re-review of daf3583 was **approve-with-fixes** too; its fixes are applied at 35eeb46
and in the docs ("Re-review", below). No library, calibration or ABI change.

Must-fix items:
1. **The judge did not enforce MD2-3.** It keyed its verdict by system only, and `run`
   appends to one records file, so two records of the same data set passed with no
   decay_opp record. The reviewer reproduced it on the CPU: a smoke `normal` record
   relabelled sq4096 at S14/F14 = 0.70, written twice, gave `sq4096: PASS`.
   - *Fix:* the decision is keyed by (system, data), and a system needs a decisive record
     for both normal and decay_opp.
   - The same relabelled case now gives INCOMPLETE (Status "M-d2").
2. **The contamination, duplicate and correctness rules were missing.** The judge counted a
   contaminated record, or one that failed the bit-exact check, toward PASS or FAIL. That
   left "re-run until clean or inside" as an open forking path in a pre-registered gate.
   - *Fix:* rules (a)–(d), now in §4.3 and the driver's docstring.
   - (a) Contaminated or not bit-exact means inadmissible, and a bit-exact failure stops
     M-d2.
   - (b) One re-run at most.
   - (c) The first admissible record decides, and every record is reported.
   - (d) Two inadmissible attempts leave the pair INCOMPLETE until the cause is recorded.
   - `run` refuses what the rules forbid.
3. **Re-registration before any M-d2 measurement.** The fixed driver is committed (daf3583,
   sha256 `b4cb9326…`). §4.3, Status and the findings name it as normative in place of
   3254b15 / `95ed2f6c…`. The verdicts are recorded here, with a summary in DESIGN §6g.9.
   (Superseded by 35eeb46 after the re-review, below.)

Should-fix items:
- *Provenance checks, all applied.* `judge` refuses, and `run` will not make, an M-d2 record
  that fails any of the following:
  - a clean tree, a `--md2-cleared` reason, and the judging driver's sha256;
  - the bars' configuration at every level (N moduli, g = N, GENERATE, in-line, RN,
    EMULATE), read from each record rather than assumed;
  - the fixed protocol parameters. The implementer added this one: `--reps` and the other
    parameters are flags, so a run with other values would not be the protocol;
  - a functional calibration check (`calib_check`). The library embeds its calibration at
    build time, so the file hash alone could pass a stale build. The loaded library's
    shipped calibration must equal the registered file, as canonical text and as
    bit-identical predictions (in-line DEFAULT/COLD and BURST/REPEAT, and the pack, at
    every level). It passes at daf3583. A changed parameter fails it, and so does a changed
    notes line.
- *MD2-4 wording and labels.* The like-for-like figures are added, P14/S14 = 0.509 and
  PF14/F14 = 0.436 beside P14/F14 = 0.382. Every schedule figure using N = 6 or 9 is
  marked [M, model-only, outside hold-out] (IR4).
- *The MD2-4 caller note* is in §2.4 and DESIGN §6g.12.
- *The report cites both smoke ratios or neither.* They are tooling checks, not model
  evidence (Status "M-d2").
- *Optional, done:* `judge` counts the gated S14/F14 runs not queued ahead, apart from the
  pooled fraction.

The questions and verdicts:
- **MD2-0. A request to run M-d2 while MC-3 is open.** A workflow task of 2026-10-07 asked
  for M-d2 on the shipped calibration. *Decision:* not run. The §5.9 "M-d2 ordering"
  verdict is binding, and running first would create the out-of-order evidence and the two
  M-d2 results it rules out. The tooling was pre-registered instead, and `run` encodes the
  ordering (`--md2-cleared`).
  ***Verdict: UPHELD; M-d2 does not run now (binding). The path is MC-3a first.***
  - MC-3 may be closed without a calibration change only as MC-3a's recorded outcome (the
    feature fails its pass conditions and is not adopted, with a §5.9-style verdict) or by
    Eric's explicit decision. A workflow request, or the wish to unblock M-d2, is not
    grounds to close it.
  - If MC-3a changes the calibration: re-run M-b's [M] figures and `predict` (which exits
    1); update the bars, the band classes and `REGISTERED_CALIB`; commit before any M-d2
    run.
  - If MC-3 closes unchanged, M-d2 runs on 0.750 / 0.664 / 0.673 as registered.

  *Rationale:* several binding decisions put M-c and its recalibration strictly before
  M-d2: §5.9, IR23, the order of work and the definition of done. A workflow task carries
  no authority to override them. The n1 refit showed that the PPL view's prediction is
  sensitive to the KL parameters (+11 … +20 %), so MC-3a may move a gated bar. M-d2 records
  made now would have to be discarded, giving two M-d2 results, or could leak per-phase PPL
  errors into MC-3a's feature choice, which MC-3a restricts to the allowed records. The
  reviewer re-ran `predict` at 99de0ef: byte-identical.
- **MD2-1. What is timed.** *Decision:* end-to-end 16-call runs, back to back on one stream
  through the device API. The gated statistic is the ratio of the median run totals. It is
  not composed from per-N single-call medians: M-c's indicative §6g.9 figures did that, and
  the verdict excludes them from M-d2. The phase switches (another plan; the first call at
  a new N) are inside the runs. The paired per-rep ratios are reported.
  ***Verdict: UPHELD.*** This is the quantity the bars predict, and the paired ratios and
  the per-phase table never adjust the bars or the band.
- **MD2-2. The band.** *Decision:* relative, bar × (1 ∓ p90), not bar ∓ p90 in absolute
  ratio units. That is narrower (± 0.11 rather than ± 0.15 on the PPL view), and p90 is a
  relative error. The class comes from the predicted (DEFAULT/COLD) calls; if a system's
  levels span classes, it takes the largest p90 among them. The cap ≤ 0.8 applies to the
  gated three.
  ***Verdict: UPHELD.*** §6c.5's p90 is a relative error of the total. Fixing the class from
  the predictions, not from measured times, removes a data-dependent choice of band. Every
  level of the gated three is predicted ≥ 1 ms (DF at N = 6 is the closest, at 1.202 ms),
  so p90 = 15 %. *No action:* §6c.5's targets are validated in BURST/REPEAT, the bars in
  DEFAULT/COLD, and the gated ratios move by under 1 % between the two (0.7501/0.7501,
  0.6635/0.6696, 0.6728/0.6796).
- **MD2-3. Data.** *Decision:* both normal and decay_opp are gated, and a system passes only
  when both are inside. This is stricter than either alone. decay_opp is
  `c2_fixtures.synthetic` on the GEMM view, with K in the spec's mode order.
  ***Verdict: UPHELD, with the implementation fixed*** (must-fix 1): the judge checks
  {normal, decay_opp} per system.
- **MD2-4. In-line bars against packed use.** The registered bars are in-line ratios, the
  configuration of the [M] figures. The re-pack runs (P14 / PF14) are reported, not gated.
  *Observation [M, model-only, outside hold-out]:* on the PPL view a caller who packs the
  invariant B per phase is predicted at P14/PF14 = 0.876, above the 0.8 cap. The four
  packs of the 680 MB B (1.72–2.41 ms each) compete with packed calls of 0.67–1.59 ms.
  - Like for like, packing B at the same schedule gives P14/S14 = 0.509 (about 1.97×), and
    at fixed N = 14 PF14/F14 = 0.436. P14/F14 = 0.382 combines the schedule and packing
    effects.
  - DF (0.667) and 4096³ (0.678) change little.

  *Decision:* the binding bars stand. The implementer asked whether §2.4 or the caller
  guidance should say that the in-line gain (1.33×) overstates a packed-B caller's (1.14×).
  ***Verdict: bars unchanged (binding); the packed runs stay reported only. YES to a caller
  note*** in §2.4 and DESIGN §6g.12, labelled [M, model-only, outside hold-out], with the
  like-for-like figures. *Rationale:* §5.9 binds the in-line bars, and a packed gate added
  after registration would change the gate after the fact. A caller who packs the invariant
  integrals, the realistic consumer, would read 1.33× as their gain. N = 6 and 9 and the
  large-pack predictions lie outside the hold-out until M-c passes. M-d2's measured
  P14/PF14 replaces the figures once it runs.
- **MD2-5. "The gain against a fixed N = 8–9".** *Decision:* read as §2.4's schedule ending
  at N = 9, S9 = 6×N6 + 10×N9 against F9, plus the per-call t14/t8 and t14/t9 from F8 and
  F9. All reported.
  ***Verdict: UPHELD.*** It is §2.4's own schedule, word for word.
- **MD2-6. Method details.** *Decision:*
  - a 3 s warm-up of the system's own N = 14 call (warm_run.sh's 3 s, on the system itself
    rather than 4096³), then every level once, so no first call is timed;
  - a 1 GiB memset primer before each run, so that the host enqueues the run ahead of the
    GPU. Its cost is that each run's first call starts with a flushed L2, which is
    immaterial for ≥ 1 ms calls;
  - 15 reps in rotated order;
  - warm_run.sh's 5 % spread rule with M-c's 2 attempts;
  - the raw entry points `contract_ptr` / `pack_ptr` in the timed loop, with the checked
    calls (`contract`, `Pack.pack` into a `PackedBuffer`) in the bit-exact check.

  ***Verdict: UPHELD, amended*** by must-fix 2 (rules (a)–(d)) and by the provenance
  should-fix items. *Rationale:* the warm-up loads every kernel instantiation before any
  timed call. The primer gets each run enqueued without a host gap (the smoke: enqueue
  ≤ 0.23 ms against a 1.1 ms primer, 35 of 35 runs). The flushed L2 at each run's first
  call affects S14 and F14 alike and is immaterial at ≥ 1 ms. Rotation balances position
  effects. Stock clocks is the regime of §6c.5 and of M-c. What was missing was what
  happens to a contaminated, duplicated or non-bit-exact record.
- **MD2-7. Configuration.** *Decision:* `PrecisionPlan.configs()` with GENERATE only and
  in-line, the bars' configuration (M-b's view). It picks g = N at every level on all four
  systems (`predict`).
  ***Verdict: UPHELD; the judge verifies it per record.*** The configuration is chosen at
  run time through the loaded library, so each record is checked rather than assumed (done:
  each record carries every level's num_moduli, g, vec mode, packed, rounding and mode).
- **MD2-8. Which driver is "the registered one"** (implementer, while applying the review).
  A file cannot contain its own sha256.
  *Decision:* `judge` requires each M-d2 record's `driver_sha256` to equal the judging
  driver's own, and the record's tree to be clean (`dirty` false, this driver included).
  The judge is then byte-identical to the committed driver that made the record. §4.3 and
  the findings name the registered commit and sha256, and `judge` prints its own for
  comparison. A separate registration file was considered and judged no stronger. The
  decision's sentence "Any later change to the driver re-registers it" is struck by the
  verdict.
  ***Verdict (re-review): UPHELD in mechanism, AMENDED (binding).*** A file cannot carry
  its own hash, and a separate registration file is no stronger than the docs naming the
  commit and sha256, which `judge` prints for comparison, so the mechanism is sound.
  - Before the first M-d2 record, re-registration is a commit plus updated §4.3 and
    findings, as now.
  - From the first M-d2 record on, the driver is frozen: a change needs an
    independent-review verdict recorded here. Earlier records stay binding, judged by the
    driver that made them.

  *Rationale:* otherwise refusing whole files by driver sha256 becomes a way to discard
  unwanted records (the re-review's must-fix, below).

**Re-review of daf3583 (approve-with-fixes, 2026-10-07; binding).** No M-d2 record exists;
every fix is applied at **35eeb46** (the driver, sha256 `90fa7f92…`, re-registered; the
ctest) and in the docs, before any M-d2 record.

Must-fix: **close the forking paths outside a single records file.** Rules (a)–(d) were
scoped to whatever `--out` file `run` is given, and `judge` refuses a whole file whose
records came from another driver. With "any later change to the driver re-registers it",
"start a new file" and "edit the driver and re-run" stayed open: the "re-run until clean or
inside" path that the first review's must-fix 2 was meant to close. *Applied* in §4.3
"Records, freeze and continuation" and in the driver's docstring:
1. One normative records file, immerdink-eval `results/c1/m_d2/md2.jsonl`. Every M-d2
   record `run` ever writes, in any file, is an attempt under (b) and (c) and is reported.
2. The driver is frozen from the first M-d2 record on (MD2-8 as amended).
3. After rule (d) or STOPPED nothing continues automatically. A further attempt, a new
   records file, or a restart after a library fix needs a recorded reviewer verdict or
   Eric's decision that names the cause or the fix and the new file. All earlier records
   stay reported.

Should-fix items, all applied at 35eeb46:
- *Library identity.* `--lib` was a free flag, and `lib_sha256`, `host_lib_sha256` and
  `host_api` were recorded but never compared: two records of one system from different
  builds were judged as one. Now:
  - `run` refuses the M-d2 systems unless `--lib` is `build/libimmerdink.so`;
  - `judge` refuses a file whose M-d2 records do not share one (`lib_sha256`,
    `host_lib_sha256`, `host_api`), and `run` will not add such a record;
  - `run` records `predict`'s `registered_check` for the system (the bar to 3 decimals and
    the p90 class on the loaded host library), and `judge` requires it. `calib_check`
    compares two calibrations on the same model code, so a model-code change with an
    unchanged `.calib` used to pass.
  - §4.3 states that the library is built at the registered commit before the first M-d2
    run, and that the findings name its sha256.
- *`inadmissible()` matched neither its docstring nor §4.3.* A level was bit-exact unless
  its value was literally false, so None, 0 or "yes" decided a pair. Now any value that is
  not true is inadmissible, and only false stops M-d2. The selftest checks None, 0, 1,
  "yes" and "true".
- *`md2_time_selftest` is a CPU-only ctest* (`tests/CMakeLists.txt`, `tools/test.sh`'s
  CPU-only list), with all three bands, the bars and p90, the runs and the protocol
  constants pinned as literals. Planting df bar = 0.70 or df p90 = 0.20 used to pass; it now
  fails 3 checks each. The selftest has 95 checks, against 60 before.
- *`--md2-cleared` cites the closure:* a commit, the record's commit or an ancestor of it,
  and a "NOTES-C1 §n.m" section (§4.3). `run` and `judge` check both.
- *Wording:*
  - A record with no `calib_check` now says "no calib_check".
  - Under STOPPED, the failing pair now reads "M-d2 STOPPED by a bit-exact failure: no
    re-run", not "one re-run allowed".
  - A bit-exact failure on any record made with the M-d2 library, smoke included, blocks
    M-d2 until fixed (§4.3 rule (a)). `judge` and `run` apply it to every record in the
    file; the selftest's smoke case now expects STOPPED.
- *This record:* MD2-8's verdict, the Status bullet, and DESIGN §6g.9.

The re-review's verdicts on the questions:
- *First review's must-fix 1 (keyed by system only):* **RESOLVED.** The reviewer re-ran
  `judge` on `repro_same_data_twice` (INCOMPLETE) and `both_data_inside` (PASS).
- *First review's must-fix 2 (rules (a)–(d)):* **RESOLVED within a records file.** The
  cross-file and continuation gap is the must-fix above.
- *First review's must-fix 3 (re-registration before any M-d2 measurement):*
  **RESOLVED.** No M-d2 record exists.
- *Nothing regressed* (predict, the bars, the smoke, the provenance claims):
  **CONFIRMED.** The reviewer reproduced the functional `calib_check` and the caller-note
  figures.
- *MD2-8:* **UPHELD in mechanism, AMENDED** (above).
- *Provenance failures refuse the whole file rather than marking the record
  inadmissible:* **UPHELD, with a condition.** The registered `run` cannot write such a
  record, so it is not an attempt under (b). Counting it as inadmissible would let a
  non-protocol record use up or occupy a pair's re-run, and letting it decide would be
  worse. *Condition:* a refused file is a procedural breach, not a reset. It is recorded
  here (which record, how it got there), and continuing in a new file needs a recorded
  reviewer verdict or Eric's decision (§4.3).
- *The fixed protocol parameters in the provenance check:* **UPHELD.** `--lib`, the one
  free flag left, is now covered by the library-identity item.
- *`selftest` not a ctest:* **ACCEPTED for that round, should-fix at the next code
  commit.** Done at 35eeb46.
- *MD2-0 (M-d2 not run; MC-3a first):* **UPHELD (binding).** The fixes add no
  measurement and change no library, calibration or ABI.

### 5.11 MC-3a: the one attempt, its checks and its outcome (implementer, 2026-10-09)

Executed as §5.9 MC-3a binds, on the bound set by the decision to run it once: at most two
new keys, one refit, one fresh M-c run, at most 45 minutes of GPU. **Outcome: every §6c.5
class target is met on every set it is judged on, and one pre-registered condition is not
met under its literal reading; no pass is declared.** M-c stays FAILED, the
[M, model-only, outside hold-out] marks stay, the shipped calibration (`d31f62ca…`) is
unchanged, and **MC-3 closes here without a calibration change** (§5.10 MD2-0: "only as
MC-3a's recorded outcome"). The commit that records this closure is the one that adds this
section; M-d2 may run on its pre-registered bands (below). Source: immerdink-eval
`findings/c1-mc3a.md`; records and logs `results/c1/mc3a/` and `results/c1/m_c/logs/mc3a_*`
(untracked).

**Order, as bound.**
1. The allowed-record account and the DESIGN §6c text (§6c.9) and the key, in d974bfc. The
   fresh hold-out's case list was committed before, in 7be9700 (a commit earlier than the
   text; both precede the refit, and the list contains no model decision).
2. The refit (one run, 7 s, `scripts/c1/mc3a/refit.sh`): the shipped records plus
   `calib_n.jsonl`, `perf_fit.py --cache-terms delta,omega`, the new `warm.kl_drop` scan on.
   Candidate `refit/mc3a.calib`, sha256 `64734979…`; it stays in immerdink-eval and is not
   committed to the product repo (as n1).
3. The pass checks (CPU), then the fresh GPU runs (7.2 minutes of chunk time: 6 chunks at
   82 / 56 / 60 / 71 / 87 / 74 s, each after its own FREE busy check, each with its
   `provenance.txt`: commit d974bfc, dirty false, `imd_bench` `edfd9006…`, `libimmerdink.so`
   `51612554…`, `libimmerdink_host.so` `0f75428f…`, host API 13). The correctness tests of
   `tools/test.sh` took 3.2 minutes of GPU on the same build. The records do not depend on the
   calibration the library embeds.

**What was decided (design questions, for the reviewer).**
- *One key, not two.* `warm.kl_drop` (the REPEAT credit key). The second key, the KL
  narrow-pack time, is not adopted: the only candidate the records name, the compute chain
  `nc · values · g` of the busiest SM (the v1.8 chain that fitted to 0), was tried offline
  on the KL class with the allowed records (diff `results/c1/mc3a/chain_exploration.diff`,
  scripts `kl_explore.py`, `kl_cv.py`). It implies 57 ns per step from Q24's own slopes (A and
  B alike) but fits 6–13 ns per step against the calibration's wide launches, and the
  leave-one-shape-out median error of a held-out Q24 shape's KL B goes 13.7 → 12.7 % at best.
  A variant restricted to underfilled launches would fit but breaks the monotonicity of the
  prediction. KL B is therefore an **open model defect** (DESIGN §6c.9 and §7, NOTES-B2 §12,
  with its size and trigger). This is the cost of "one attempt": the exploration fits were the
  KL class alone on the allowed records, no full calibration was refitted for it.
- *The key's storage.* A new field in `imd_perf_calib_t` would grow the struct (API 14), so
  `warm.kl_drop` lives in a parameter slot past the classes (the capacity holds four classes
  more than exist); the struct, host API 13 and format 2 are unchanged, and a library without
  it rejects a file that has it. The alternative, a free slot of the `call` class, was
  rejected as a name that does not describe a cache term. The slot was `param[72]` (the first
  one past the classes) at d974bfc; on the reviewer's must-fix it is the named constant
  `IMD_PERF_PARAM_KL_DROP_SLOT` = `IMD_PERF_PARAM_CAPACITY - 1` = 95 (header, `perf.cpp`,
  `host.py`, tests), with a `static_assert` that the classes' slots end before it. The file
  names the key and the slot is only the in-memory place, so the on-disk format and every file
  read or written are independent of the slot: the shipped calibration has no such key
  (byte-identical to before; predictions with the key absent unchanged), and a key written by
  the earlier build parses to the same value. A hand-built `imd_perf_calib_t` must zero
  `param` (header note).
- *The "before" of the in-sample condition* (MC-3a item 4, second bullet). Read literally it
  is the shipped fit's `perf_fit` table (the shipped records only); the refit adds Q24's
  records, and the K4 classes' fit depends on them and not on the key. Both readings are
  reported below; the literal one decides, since a pass may not be claimed that a
  pre-registered check does not show.

**The pass checks (all with the candidate calibration; the shipped one beside it).**

| check | result |
|---|---|
| KL A per-phase median \|error\|, allowed narrow-KL records (30 phases) | **35.4 → 4.8 %** (p90 50.5 → 13.8): improves. KL B 34.4 → 12.1 % (signed −10.9 %: still under-predicted), K4 11.4 → 10.5 % |
| the fit's `warm.kl_drop` | **0.70**, the scan's minimum (relative SSE 4.44 at 0.70; 8.05 at 0, 5.21 at 1.0): neither the rule nor zero credit |
| in-sample phase classes, `perf_fit` table, **literal** (shipped fit's table → candidate; median / p90, %) | k0 4.4 / 15.4 → 4.4 / 15.5; kf 6.4 / 23.8 → 6.3 / 23.7; kl 6.3 / 16.9 → 6.7 / 17.6; pack unchanged; k4_128x256 3.8 / 15.3 → 3.7 / 14.5; k4_128x128 8.4 / 14.1 → 8.4 / **15.1 (+1.02)**; k4_64x128 9.0 / 22.3 → **10.6 (+1.63)** / 20.8; k4_64x64 9.0 / 28.1 → 9.2 / 24.2; k5 4.5 / 14.4 → 4.5 / 14.1; direct unchanged. **Not met** (two entries, both K4) |
| the same, **same records** (n1: the refit with Q24 and no key → candidate) | K4 and k0, kf, pack, k5, direct identical (the key does not touch them); kl 8.1 / 19.7 → **6.7 / 17.6**. Met |
| v1.18 hold-out, all 73 cases in the class statistics (199 calls) | < 100 µs 3.7 / 10.6, 100 µs – 1 ms 2.9 / 7.3, ≥ 1 ms 2.2 / 7.5 % (shipped 3.8 / 10.7, 2.7 / 8.2, 2.5 / 6.4): all met |
| v1.18 hold-out, the 66 whose GEMM view matches no fit case (69 minus the 3 cache pairs; 171 calls) | 3.7 / 10.1, 2.8 / 7.7, 1.9 / 4.2 % (shipped 3.8 / 10.1, 2.8 / 8.3, 2.2 / 4.5): all met |
| `mc.jsonl`, 160 calls | < 100 µs 2.9 / 6.9, 100 µs – 1 ms 2.3 / 6.6, ≥ 1 ms 3.0 / 13.5 % (shipped 15.8 / 28.4, 2.5 / 14.7, 3.0 / 7.3): all met, on the M-c N only, the anchors, normal and decay_opp too |
| `mc_with_repeat_small.jsonl` | 2.1 / 9.5, 2.1 / 4.8, 3.0 / 13.5 %: all met |
| fresh narrow-KL hold-out (`h_*`, 30 calls: 18 KL, 12 KF; 20 calls < 100 µs, 10 in 100 µs – 1 ms) | **< 100 µs 4.3 / 15.7, 100 µs – 1 ms 2.7 / 6.6 %: met** (shipped 8.2 / 25.5, 3.6 / 15.7: also met). KL shapes +0.7 … +6.6 % except `h_kl3` (`ik,jk->ij`, 1152 × 80 × 1664) +11 … +16 %; KF −7.3 … +4.1 % unchanged |
| **fresh full M-c run** (160 calls, 4 chunks, `mc_fresh.jsonl`, commit d974bfc) | **< 100 µs 1.8 / 7.6, 100 µs – 1 ms 1.8 / 7.4, ≥ 1 ms 3.3 / 13.2 %: all met**, and on the M-c N only (1.8 / 7.6, 1.8 / 5.1, 3.3 / 14.0), the anchors, normal and decay_opp. Shipped calibration on the same records: 13.8 / 28.8, 2.0 / 14.7, 3.5 / 7.9 (also met this time: the first run's 15.8 % is not reproduced by this run, so the miss was marginal and run-dependent) |

Per N of the fresh run (candidate; median / p90 of the class that dominates): the < 100 µs
calls at N = 4 are 7.4 / 14.0 %, at N = 5–10 1.3–4.4 / 1.9–7.5 %; the ≥ 1 ms calls 0.7–5.6 /
10.0–18.5 % (N = 4: 4.8 / 18.5, N = 5: 5.4 / 16.0: above the p90 target at the smallest N as a
per-N figure, which is reported and not judged, §5.9 MC-2). The Jacobi view is within
−4.8 … +3.4 % at every N (shipped: −26 … −29 % at N ≤ 10). The KF small-N optimism is **not**
fixed: sq1024 at N = 4 is −13.6 / −14.1 % (normal / decay_opp) and +11.8 / +10.3 % at N = 14,
and it no longer sets the median because the Jacobi calls no longer sit above it. Per phase on
the fresh calls ≥ 1 ms: KL A 7.7 / 22.5 %, KL B 3.9 / 19.1 %, K4 3.1 / 11.2 %, K5 15.4 / 29.7 %.
On M-c's Jacobi calls (phase marks, reporting only; signed medians): KL A +8.7 %, KL B −18.2 %,
K4 +9.4 % against the shipped −33.0 / −45.1 / +23.6 %. The Jacobi totals are right partly by
cancellation, the KL A and K4 over-prediction against KL B's remaining under-prediction: the
key is fitted on the KL phase marks and removes the KL A error without that cancellation (KL A
4.8 % on the allowed narrow-KL records), but KL B is the open defect above.

**What the candidate costs.** The PPL view (100 × 9216 × 9216, an M-d2 system) is predicted
+10 … +17.5 % slow at every N (shipped +2 … +9.7 %), the same shift as n1 (+11 … +20 %); the
≥ 1 ms class p90 on `mc.jsonl` and the fresh run is 13.2–14.0 % against 7.3–8.3 % shipped,
close to its 15 % target. The KL parameters move far from the shipped ones (`kl.wave`
2160 → 4431 ns, `kl.byte` −28 %, `kl.launch` −22 %, `kl.l2byte` −38 %, `kl.op` 0 → 0.22 ps)
and K4's exactly as n1 (`k4_64x128.mac1` −42 %, effective L2 0.4 → 0.3 × nominal), because Q24
enters the fit. The call terms stay near the shipped ones (`call.call` +6 %). So Q24's K4
records, not the key, are what fail the literal in-sample condition and what move the PPL
view; the key is what removes n1's KL degradation (kl 8.1 / 19.7 → 6.7 / 17.6).

**Verdict.** Independent review of MC-3a (2026-10-09, binding): ruling A. Item 4 of §5.9 MC-3a
("no phase class's in-sample median or p90 may worsen by more than 1 point") is read against
the shipped fit's `perf_fit` table, as written. Two K4 entries worsen by more than 1 point
(64x128 median +1.63, 128x128 p90 +1.02), because MC-3 step 3 puts Q24's records into the
refit. The same-records (n1) reading, against which the candidate meets every check, is
informative and is not adopted. The pre-registered conditions do not show a pass, so MC-3a
does not pass. M-c stays FAILED, the shipped calibration (`d31f62ca…`) stays, and MC-3 closes
without a calibration change. The candidate carries the costs for which §5.9 MC-1 upheld the
NO on n1 (PPL view +10 … +17.5 % against +2 … +9.7 % shipped, ≥ 1 ms p90 13.2–14.0 % against
7.3–8.3 %, K4 path parameters fitted to a cache state, and a change to every [M] figure of
M-b). The shipped calibration also meets every §6c.5 target on the fresh M-c run (13.8 / 28.8,
2.0 / 14.7, 3.5 / 7.9 %), so there is no demonstrated need that outweighs them; that
observation is recorded and is not a pass (M-c's first-run miss was marginal and
run-dependent). `mc3a.calib` stays in immerdink-eval. `warm.kl_drop` stays in the library and
the fitter (optional, absent = 0, tested, bit-identical when absent; slot
`IMD_PERF_PARAM_KL_DROP_SLOT`). KL B stays an open model defect (DESIGN §6c.9, §7, NOTES-B2
§12) with its size and trigger; no second key is tried. M-d2 runs on the shipped calibration
and its pre-registered bars (0.750 / 0.664 / 0.673, ± the class p90, ≤ 0.8) with run-time
provenance (commit, dirty flag, library sha256). Whether C1's done condition ("M-c has
passed") is amended is for Eric to decide after M-d2. No M-c re-run on the shipped
calibration is to be used to claim a pass.

Consequences, as bound: M-c is FAILED and the [M, model-only, outside hold-out] marks stay;
C1 is open on its done condition and on M-d2. **M-d2's provenance includes the library
sha256 and the commit**: the library now contains the `warm.kl_drop` code (d974bfc and the
slot constant), which is bit-identical to the design-phase [M] library's predictions when the
key is absent (tested: `suite_perf_v119` and `test_host_c`; the shipped file has no key), so
the library hash differs from `466acdb4…` and each M-d2 record names the hash it ran on.

**Provenance.** The hold-out, the refit and both M-c judgements use the product commits above;
the refit's input records are those of the shipped calibration (`calib-20261006-2110`,
`sustained-merged`, `cache-20261006-2110`) and `calib_n.jsonl`. Nothing was run twice except
`perf_fit.py` once more with a three-decimal copy of its table (`/tmp`, same inputs, same
output, sha256 equal) to read the K4 differences above exactly. The mc.jsonl-based reports were
read once, after the refit, as the pass checks; no choice depends on them.

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

**Completeness critic, accepted by the independent reviewer (IR1–IR29) — claims of the previous
version of these notes:**

42. **Hold-out coverage.** "The hold-out covers N = 8 / 14 plus 10 / 16 only, so no N ≤ 7" was
    wrong both ways. The calibration set has N = 4 (2 rows), and the hold-out has N = 15–20. No
    case anywhere is at N = 5, 6, 7, 9 or 13 (§1.2 item 8, IR4).
43. **§2.4's "predictions at N = 6 and 8, outside the hold-out".** N = 8 is the best-covered N.
    The uncovered ones are 6 and 9 for the schedules, and 6 and 12 for the default levels (IR4).
44. **§2.7's "K-eq at about 0 per-call cost"** contradicted §2.6's recurring 6–13 % per call
    per changed operand. That holds only for invariant operands or pre-scaling fused into the
    producer (IR12).
45. **§1.3's "Krylov … tighten precision as the residual falls."** Relaxed (inexact) Krylov
    methods may loosen the matvec precision as the residual shrinks (IR3).
46. **"Rebuild before stage 2."** Stages 0 and 1 already need the API-13 symbols, so the
    rebuild is stage 0's first step (IR14).
47. **M-a at m = n ≤ 512 as the Q13 evidence.** The gate needs the ≥ 2048³ class, and
    random-structure gains change sign with m (IR1).
48. **Q12's escape clause** ("unless M-a finds batch-dependent profiles") could not fire: M-a had
    no batched fixture (IR16).
49. **Q16's "the recipe covers the need"** rested on documentation alone; no shipped code
    implemented the certified FP evaluation, and α ≠ 1 or β ≠ 0 were not addressed (IR7).
50. **F3's "no byte beyond `pack_bytes`" as ABI** rested on one shape's guard zones (IR17).
51. **§4.5's "every partner of a pre-scaled packed operand must carry the same w"** gave no rule
    for choosing that w across several partners (IR6).
52. **The stage-5 cost bar named `ncu`**, but GPU performance counters are admin-only on this
    machine (IR26). *Corrected 2026-10-07: that premise is wrong (the counters are open to all
    users). The restated bar stands because it measures time directly and `ncu`'s replay does
    not reproduce production timing; `ncu` is a diagnostic only, with `--clock-control none`,
    under the GPU lock (§4.6, "Correction").*
53. **The stage-0 GPU tests** assumed a phase's first launch could happen inside a capture
    without any check. K5<N> gets no plan-time attribute call, so lazy module loading may land
    there (IR10).

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
of the 2048³ class or larger. *Checked 2026-10-07 (M-a): not met.* No non-separable Tier-L
or real case of the class qualifies. Against K-seg 2 and what a caller gets from K-eq (K-eq d*
where its guard holds, otherwise fast mode), the largest saving is 1 modulus.
