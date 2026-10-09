# Handoff (2026-10-04; C1 / C2 state updated 2026-10-07)

State of immerdink for a fresh session, written before switching the dev box to a
console-only (no graphics) setup. Read this, `docs/ROADMAP.md` and the auto-loaded memory
first.

## Where things stand

- **Branches:** all work is on `dev` (pushed). `main` = PR #1 (v1.3 state); later work reaches
  main by another PR from dev. Never push to main directly.
- **DESIGN v1.12** (v1.10 §6c performance-model follow-up, v1.11 §6e planner, v1.12 §6f accuracy-aware planning) (docs/DESIGN.md): v1.4 KF fused small path, v1.5 K4 configs and rule, v1.6
  KL fused large path, v1.7 §6b packed operands, v1.8 §6c performance model, v1.9 §6d Python.
- **Host API v8** (`include/immerdink/immerdink_host.h`, CUDA-free `libimmerdink_host`, Python
  `python/immerdink/host.py`): the single source of truth for lowering, legality, workspace,
  plan bytes, K4 config, launch counts, packed-operand sizing and compatibility, and (v6) the
  performance model.
- **Performance model (B2, DESIGN v1.8 §6c, host API v6):** `imd_perf_predict`,
  `imd_perf_predict_pack`, `imd_perf_rank`, `imd_perf_calib_*` in libimmerdink_host (Python
  `host.predict`, `host.rank`, `host.Calibration`); shipped RTX 4090 calibration
  `src/host/calib/rtx4090_sm89.calib`, recalibrate with `tools/calibrate.sh [--validate]`
  (after kernel changes too); results and open items in docs/NOTES-B2.md; test `host_perf`.
- **Performance model v1.10 (DESIGN §6c, host API 7, calibration format 2, branch
  `b2-fidelity`, NOTES-B2 §6):** wave / occupancy features (launch rules shared with
  contract.cu in `src/host/analysis.hpp`), request options `regime` (BURST / SUSTAINED, default
  SUSTAINED) and `cache` (COLD default / WARM / REPEAT), per-call and per-phase bands
  (`time_lo_s` / `time_hi_s`), rank ties (`tie`, `raw_rank`, `order`). The shipped 4090
  calibration was refitted on post-BIOS data (2026-10-05 13:20–14:00); `tools/calibrate.sh
  --validate` now also runs bench/sweep_sustained.txt and bench/sweep_cache.txt (~40 min of GPU in
  total) and writes the bands. Design questions for the reviewer: NOTES-B2 §6 (SUSTAINED and
  COLD defaults, the tie rule).
- **Python device API (B1, DESIGN v1.9 §6d):** `python/immerdink` (`Plan`, `Pack`,
  `immerdink.contract`, `PlanCache`, `DeviceArray`): ctypes over the C ABI, any
  `__cuda_array_interface__` / DLPack array or raw pointer, no array library needed; test
  `python_device` (pytest), overhead `bench/python_overhead.py`.
- **Hot-path planner (DESIGN v1.11 §6e, branch `planner-fast`, 2026-10-05):**
  `python/immerdink/planner.py` (`Query`, `Planner`, opt-in `Memo`, lazy views): host.py's
  queries bit for bit at 0.8–1.8 µs instead of 8–20 µs (C floor 0.25–0.86 µs; `rank` one C
  call); pure ctypes, host API unchanged (v6), no nanobind. Test `host_planner`; per-call
  numbers `bench/planner_overhead.py` (C floor: `imd_planner_floor`). Design questions open
  for review (§6e.7).
- **Accuracy-aware planning (DESIGN v1.12 §6f, host API 8, branch `b2-accuracy`,
  2026-10-05, NOTES-B2 §7):** the planner input is the caller's `imd_err_request_t`: a metric
  (normwise, elementwise, absolute), a target, and operand statistics. It gets:
  - `imd_err_estimate`, the estimates from scalars;
  - `imd_err_min_moduli`, the smallest N that meets the target, from one lowering;
  - `err_bound` / `err_typical` in `imd_perf_predict` and in rank's candidates;
  - rank marks the candidates that miss the target as infeasible (`infeasible` bits).

  The rigorous bound has 0 violations in 300 held-out GPU cases. The typical estimate
  (Hoeffding, RN, opt-in, default p = 2⁻⁴⁰) is not valid for rows of equal entries. Reviewed 2026-10-05 (rulings applied: NOTES-B2 §7.4). The bound math lives only in
  `src/host/accuracy.cpp`; `spec/params.py error_estimate` + `cert_error_estimate` are the
  reference. Python: `host.Accuracy`, `host.min_moduli`, `planner.Accuracy`,
  `Query.min_moduli`. Tests: `host_accuracy`, `accuracy_err_api_quick`; the full GPU check is
  `bench/accuracy/err_api.py`. No TLA+ change. Design questions and rulings: NOTES-B2
  §7.4.
- **K4 load alignment (DESIGN v1.13 §4b, ROADMAP A6, NOTES-K4ALIGN):** stage 1 (K_c' =
  131008) and **stage 2** (2026-10-06, branch `k4-align-s2`, host API 9): k̂ = ⌈k⌉_A with
  A = 64 iff k > 96, else 16 (a function of k alone); pack format 2 (`IMD_PACK_FORMAT_A16`
  / `_A64`; a pack's format is that of its k); `imd_lowering_t` v6 `k_align` (192 bytes);
  test hook `imd_test_set_k_align` (libimmerdink_host; Python `host.test_set_k_align`);
  `imd_bench --k-align`. `spec/params.py` / `spec/oracle.py` run `K_ALIGN_POLICY = "AUTO"`.
  Done 2026-10-06: the A/B sweep (eval findings), the θ refit and the `k4_select` re-check
  (rule unchanged; branch `k4-align-s2-refit`, NOTES-B2 §8). A caller
  that persists pack bytes must persist `format` and `pack_bytes` and re-pack on a change.
- **Follow-ups combined (2026-10-06, branch `followups-combined`, not pushed):**
  followup-integration + `bf-eps` (DESIGN v1.14 §3.4.1, ROADMAP A7, docs/NOTES-BFEPS.md:
  branch-free ε pack emitter, same bytes, N ≥ 15 packs −20 to −41 %) + `hotclock` (DESIGN
  v1.15 §6c.7, host API 11: `power.byte_f_pj`; its draft "v1.14" renumbered to v1.15). v1.15
  also retires the model's eps path (§6c.1: `op_eps` features 0, request `eps_fraction`
  validated then ignored, `eps_s` / `eps_fraction` out 0, `op_eps` keys optional, format 2
  kept but API-11 files do not load in API ≤ 10 (§6c.2); no ABI change). The shipped
  calibration is the combined refit (NOTES-B2 §10; hold-out targets met, almost entirely by
  retiring eps: the refit is worse than old θ on < 100 µs calls; scaled power form e_tc 0.708
  pJ/op + e_byte^f 301 pJ/B, γ = 1). Open: recollect **all 14** sustained windows on the
  current library (all ran pre-v1.14 commit 0787164; the refit used the 9 N ≤ 14 ones); KF
  N-dependence (N = 8 1.2× slow), K3 pack 1.07–1.20× slow, N = 18–20 hold-out cases
  (NOTES-B2 §10). W = 1 for N ≤ 8 was evaluated and not adopted (DESIGN v1.16, branch
  `emitter-w1`, NOTES-BFEPS §4: pack −2.7 %, below the 3 % bar, for +11 % library size);
  v1.16 also makes the W = 2 shift defined (PTX shl) and the occupancy query W-aware.
- **Model fix (2026-10-06, branch `model-fix` on `emitter-w1`, not pushed; DESIGN v1.17 §6c,
  host API 12, NOTES-B2 §11):** two additive optional features in the freed eps slots,
  `kf.elem` (KF per-element chain) and `pack.dram` (PACK bytes that are not L2-hit re-reads,
  L_hit = `warm.pack_l2_bytes` = 0.8 × L2), calibration format 2 kept (absent = 0 predicts
  bit-identically; `op_eps` keys retired), f_ref by a parabola vertex with a nested
  leave-one-out figure, the `--l2 flush` phase fix. Shipped calibration refit on the current
  library with **all 14 sustained windows recollected**: hold-out (189 calls) 3.8 / 12.9 /
  23.2 %, 2.6 / 9.6 / 13.9 %, 2.2 / 6.6 / 16.8 % (< 100 µs, 100 µs – 1 ms, ≥ 1 ms; median /
  p90 / max), better than old θ (eps ignored) on medians and p90s, maxima +1.5 / +1.0 points
  on single K4-bound calls (within run-to-run noise below 100 µs, borderline at ≥ 1 ms); K3
  phase 13.3 / 22.4 → 4.9 / 14.7 %. Open: KF roofline and cache-term cap, W = 3 KF on tiny
  E = 4 shapes, K4 on tiny shapes, capped clock 0.8–3.4 % high and the power gaps (r_mm −43 %,
  the small high-clock windows u_s512 / u_s1024 −6 … −23 %; needs memory-bound and small
  windows at several clocks), L_hit between 57 and 77 MB (S is a proxy without K4's planes);
  reviewer question: format 2 kept (reviewer: keep it). Hold-out caveats (NOTES-B2 §11
  [review]): the features were chosen on 178 of the 189 calls; the PACK L2-hit side is
  checked in-sample only.
  Verified: full `tools/test.sh` 13/13 CPU + 35/35 GPU, `params.py --check` 30/30,
  `test_oracle` all pass. GPU: 56 min of sweep wall time (over the 45 min budget).
- **Perf minor (2026-10-06, branch `perf-minor` on origin/dev 684f459, not pushed; DESIGN
  v1.18 §6c.8, host API 13, NOTES-B2 §12):** the model-fix "Remaining" items after two offline
  analyses (eval `findings/perfminor-time.md`, `perfminor-power.md`). Adopted, all optional
  keys (format 2 kept, absent = 0 predicts as API 12): cache term with a per-call REPEAT saving
  `warm.call_ns` and a clean-L2 credit `warm.clean_byte` in a new state
  `IMD_PERF_CACHE_CLEAN` (COLD stays the default: evicted by other, dirty work); the PACK set
  with K4's W planes (`warm.pack_l2_planes`, L_hit 1.1 × L2); power with busy-GPU power
  `power.act_w` / `act_gap_us`, DRAM energy `power.dram_pj`, boost clock
  `power.clock_boost_mhz` (BURST at the boost clock, time flat above f_ref, `clock_mhz` out up
  to it), γ 2.5 as a prior; `imd_bench` sustained `--mode direct` variant fixed. Not adopted:
  KF read / write split (format bump: `kf.row` non-zero in format 1), KF roofline (refuted),
  E = 4 per-modulus term (fits 0), fixed PACK hit price. Shipped calibration refit (sustained
  windows reused: 24 of 26 from model-fix / perfminor on the same kernels): hold-out (199 calls)
  < 100 µs 3.9 / 10.8 / 15.8 → 3.7 / 12.2 / 19.0 %, 100 µs – 1 ms 2.7 / 10.3 / 15.7 → 2.7 / 6.9 /
  13.4 %, ≥ 1 ms 2.4 / 6.6 / 16.0 → 2.6 / 6.4 / 16.4 % (median / p90 / max vs the v1.17
  calibration on the same records; against a v1.17 refit on the same records ≥ 1 ms median
  1.6 → 2.6 and < 100 µs p90 9.8 → 12.2: a trade-off), 1024³ N = 14 +14.9 → +6.3 % (in-sample
  for the cache term: x_s1024 pairs; the held-out cache evidence is the three v_x* pairs);
  SUSTAINED nested LOO on 26 windows t 2.34 → 1.62 %, clock 6.6 → 1.3 %, power max 81 → 10 %.
  **Review fixes (branch `perf-minor-final`, CPU only):** the power / SUSTAINED clock solve now
  divides by the request's cache-state time (was COLD time while fitted on REPEAT windows),
  perf_fit recomputes the boost clock per LOO fold; power keys refit offline (f_ref 2483):
  nested LOO 1.40 %, clock 0.7 %; BURST and COLD predictions and the hold-out unchanged; doc
  fixes and open items in NOTES-B2 §12 "Review" / "Remaining".
  **c dropped (branch `perf-minor-noc`, CPU only; the independent reviewer's binding
  verdict, NOTES-B2 §12 "Decision: c dropped"):** shipped calibration refit with `perf_fit
  --cache-terms delta,omega` (now the default; `warm.call_ns` absent = 0, the key stays
  supported): L_w 0.85 × L2, Δ 1.31 ps/B, ω 0.104 ps/B, f_ref 2479, nested LOO 1.42 %; hold-out
  < 100 µs 3.7 / 12.2 / 19.0 → 3.8 / 10.7 / 17.1 %, 100 µs – 1 ms 2.7 / 6.9 / 13.4 → 2.7 / 8.2 /
  13.3 %, ≥ 1 ms 2.6 / 6.4 / 16.4 → 2.5 / 6.4 / 16.4 %; 1024³ N = 14 +6.3 → +10.8 %; small COLD
  calls ≈ 1.2 µs faster (64³ N = 14 16.68 → 15.45 µs); two of the three held-out cache pairs
  worse (clean credit 11.5 → 6.7 µs). The per-call saving is not identifiable: open, tied to
  the launch-timing question (power reviewer item 5).
  Open: reviewer questions (COLD's meaning, `clock_mhz` scale), the per-call saving c
  (dropped, above; with the launch-timing check), streaming K2 of g < N KL
  calls (+2 … +5 %), pre-packed g < N in the S rule, tiny-shape K4 / KF. GPU: 18.7 min (90 min
  sweep wall time, mostly gate waits for other GPU users).
- **Adaptive precision C1 / C2 (2026-10-07, branch `claude/ozaki-kernels-immerdink-1089fd`,
  PR #6 into dev; DESIGN v1.19 §6g and §3.8, host API 13, no ABI-layout, calibration-format or
  TLA+ change; `imd_perf_calib_t` keeps its layout and size, but `param[95]` is now the optional
  key `warm.kl_drop`, so a hand-built struct must zero `param`):** the design and the current state are docs/NOTES-C1.md (read its Status, "Status
  at the C1/C2 close-out", first). **C2 is done** by its definition of done (NOTES-C1 §4.11);
  **C1 is done as amended** (decision by Eric, 2026-10-09): M-c failed (MC-3a did not pass,
  ruling A), **M-d2 ran on 2026-10-09 and PASSES** (NOTES-C1 §5.12), and Eric amended C1's
  done condition "M-c has passed" because M-d2 passed on the shipped calibration; M-c stays
  FAILED and the `[M, model-only, outside hold-out]` marks stay.
  - Shipped: stage 0, precision phases (a plan and a pack per N, published footprint
    monotonicity F1–F3, the re-pack rule, `PackedBuffer`); stage 1,
    `immerdink.precision.PrecisionPlan` and `rho_from_output`; stage 3, the K-segmentation
    caller recipe (§3.8, PROOFS L7, `cert_kseg_composition`, `oracle.contract_segments`) and
    L3c + L7 in Lean; `oracle.kweights` and the M-a study (C2's deliverable); the M-d1 caller
    guidance (DESIGN §6g.12). The round-2 stage-0 GPU tests pass (4ba78b1). M-b is done.
  - Closed or deferred, each with its trigger (DESIGN §7, ROADMAP C1/C2): stage 4 (caller-side
    K-equilibration) closed by its gate (no real psi4 contraction of the ≥ 2048³ class passes the guard); stage 6
    (library-internal weights) deferred; stage 5 (K5 statistics + device certificate) deferred,
    triggers T1–T3.
  - **C1 record (GPU), closed as amended:** M-c ran on 2026-10-07 and **failed** (binding, NOTES-C1 §5.9): the
    < 100 µs median was missed (15.8 % against 15 %) through a small-N optimism on the KF path
    (1024³ at N = 4) and narrow-n KL calls (KL A keeps only part of its REPEAT credit, the
    narrow pack KL B is slow in every cache state; per-phase table in immerdink-eval
    `results/c1/m_c/phases.md`); the Q24 refit was not adopted and the [M, model-only, outside
    hold-out] marks stay. **MC-3a was executed once on 2026-10-09 (NOTES-C1 §5.11, DESIGN
    §6c.9):** one key (`warm.kl_drop`, fitted 0.70), a fresh narrow-KL hold-out, one refit
    with Q24, a fresh 160-call M-c run. Every §6c.5 class target is met on every set, but one
    pre-registered in-sample condition fails under its literal reading (two K4 entries, from
    Q24's records), so no pass is declared: M-c stays FAILED, the marks stay, the shipped
    calibration is unchanged, KL B is an open model defect (DESIGN §7; [GitHub issue #7](https://github.com/Yurgenschmidt/immerdink/issues/7) (label backlog)), and MC-3 is closed
    without a calibration change as MC-3a's recorded outcome. **Independent review, binding
    ruling A (2026-10-09):** the in-sample condition is read against the shipped fit's table, so
    MC-3a does not pass; the candidate calibration (immerdink-eval
    `results/c1/mc3a/refit/mc3a.calib`) stays there, `warm.kl_drop` stays in the library and
    the fitter (slot `IMD_PERF_PARAM_KL_DROP_SLOT` = 95), no second key. **Decision by Eric,
    2026-10-09, after M-d2 (NOTES-C1 §5.11 "Verdict", §5.12):** C1's done condition ("M-c has
    passed") is amended, because M-d2 passed on the shipped calibration; C1 is done as amended.
    `perf_fit.py --kl-drop` now defaults to 0 (key absent, the v1.18 rule; `scan` is opt-in), so
    a routine `tools/calibrate.sh` run cannot ship a nonzero `warm.kl_drop`. **M-d2 ran on 2026-10-09 and PASSES** (NOTES-C1 §5.12, DESIGN §6g.9;
    driver `bench/accuracy/md2_time.py` re-registered at fce29e9 with the NaN-filled bit-exact
    check; records and findings in immerdink-eval `results/c1/m_d2/md2.jsonl`,
    `findings/c1-m-d2.md`): the schedule against fixed N = 14, normal / decay_opp, PPL view
    0.769 / 0.769 (bar 0.750), DF 0.695 / 0.698 (0.664), 4096^3 0.696 / 0.692 (0.673), all
    inside their bands; `bench/first_call.py --sweep` re-run on 243d9c0 (DESIGN §6g.6). Nothing
    remains for C1.
  - Open outside C1: a `PrecisionPlan` member role "A=B" (one shared pack read through both
    roles; named "AB" before the close-out review), trigger the ehrenfest-side integration of
    the B2 performance model declaring DF-factor sharing (ROADMAP C1).
  - Close-out checks (2026-10-07, CPU only): `tools/test.sh`'s 18 CPU-only ctest tests,
    `spec/params.py --check` 33 of 33 and `spec/test_oracle.py` pass (immerdink-eval
    `findings/c1-closeout.md`); the GPU suite last ran at 4ba78b1 (56 of 56; no `src/` or
    `include/` change since). The close-out review's fixes (docs and one docstring) re-ran the
    18 CPU-only tests: pass.
  - Evaluation records and summaries are in immerdink-eval `results/c1/` and
    `findings/c1-*.md`.
- **Verification on dev:** 8 CPU + 28 GPU ctest tests, 284 TLC models, 20 Apalache checks,
  oracle suite, 26 params certificates, all passing (b2-accuracy: 9 CPU + 29 GPU ctest tests, 27 certificates) (`tools/test.sh`, `spec/check.sh`,
  `spec/apalache/check.sh`, `spec/test_oracle.py`, `spec/params.py --check`).
- **D3 gate (Eric, 2026-10-04):** D3 begins with a design/theory discussion with Eric to narrow
  the questions. Do not start D3 agents or specs before that.
- **Roadmap done:** A1, A2, A3, A4, A5 (2026-10-04, docs/NOTES-A5.md), D1, D4 part 1, B1, B2;
  C1 and C2 (2026-10-09 and 2026-10-07; PR #6 into dev, the C1/C2 bullet above; C1 done as amended).
  Next by Eric's priority: D3 (packed
  symmetric contractions; design preference: one generic kernel, else split; TLA+ first).
  SE items are parked. For tiny-call timings use `imd_bench --methods duel` (two immerdink
  configurations interleaved; docs/BENCHMARKING.md, NOTES-A5 §1.2). The bench is
  immerdink-only; comparisons with other libraries live in the evaluation workspace.

- **Comparison (2026-10-04):** the clock-locked comparison with other libraries lives in the
  local workspace `~/repos/immerdink-eval` (never pushed; data untracked), together with the
  raw results and campaign sweeps the NOTES files refer to. Product benches stay
  immerdink-only (docs/BENCHMARKING.md).

## Open items needing Eric

1. **DONE 2026-10-04 (accepted)** — D1 §6b.11 clarification: a missing pack handle is PACK_MISMATCH; a missing packed
   buffer is INVALID_VALUE only when it would be read. Implemented that way; awaiting OK.
2. **DONE 2026-10-04 (keep optional, default off)** — `k4_tune` (plan-time autotune, default off, no measured gain): keep, remove, or leave.
3. **DONE 2026-10-04 (bench mode, the evaluation workspace, legacy/docs/BENCHMARKS.md "Clean re-run")** — clean benchmark re-run: the A3/D4/D1 sweeps were partly contaminated by desktop GPU
   time-slicing (docs/NOTES-K5-bimodal.md). Re-run `bench/sweep_compare.txt` (and
   `bench/sweep_d1.txt`) once the 4090 has no desktop on it, ideally with persistence mode on
   (`sudo nvidia-smi -pm 1`). Records now carry `attempts` and `contaminated`.
4. **Decided 2026-10-04: not now** (revisit for a paper or sub-percent tiny-shape numbers) — GPU clock locking for paper-grade small-shape numbers (`sudo nvidia-smi -lgc`): optional.
5. **C1/C2 (2026-10-07; docs/NOTES-C1.md "To report to Eric")**, none blocking:
   - Q6 and Q7 (ROADMAP C1's decision column, C2's new title) were approved on Eric's behalf
     by the independent reviewer (IR18); Eric may overrule.
   - MC-3 can close without a calibration change only as MC-3a's recorded outcome or by
     Eric's explicit decision (NOTES-C1 §5.10 MD2-0); done: MC-3a closed it, M-d2 ran (§5.11, §5.12).
   - Stage 5 (device statistics, the a-posteriori certificate) is deferred with triggers
     T1–T3; T3 is Eric asking for the device certificate (NOTES-C1 §4.6).
   - IR26's premise was wrong: GPU counters are open to all users here. Stage 5's cost bar
     keeps its content; `ncu` is a diagnostic only, with `--clock-control none` (NOTES-C1
     §4.6, "Correction").

## Machine notes

- RTX 4090 shared with the ehrenfest session(s): all GPU work under
  `flock /home/ef/repos/ehrenfest-ipdps/_wip/gpu.lock` (the tools take it themselves; never
  nest it around them).
- Heavy jobs run inside `tools/memcap.sh` (shared cgroup budget, RAM/2). The tools already do.
- PCIe was stuck at gen1 on 2026-10-03 and recovered after the reboot (gen4 x16, 21 GB/s).
- Nsight Systems at `~/nsight-systems-2026.5.1/target-linux-x64/nsys`; `ncu` in the immerdink
  env; GPU counters enabled for all users. Run `ncu` only with `--clock-control none` (its
  default, `boost`, locks the clocks), under the GPU lock, once a busy check shows no
  ehrenfest timing tests running, in holds of at most 5 minutes (NOTES-K5-bimodal ran it with
  `--clock-control none` under the lock; NOTES-C1 §4.6, "Correction").
- Benchmarks: `bench/run_sweep.sh` / `bench/run_ab.sh` (quiet-gated, warm-up, retries,
  clock-state records). Correctness tests are not quiet-gated.

## Resuming this conversation

The original session can be resumed from the CLI (sessions are keyed by directory):
`cd /home/ef/repos/immerdink/.claude/worktrees/ozaki-kernels-immerdink-1089fd && claude --resume b6642cd7-986f-4a6d-a271-daef6ce2dedf`.
If that fails, start a new session in the repo and read this file.
