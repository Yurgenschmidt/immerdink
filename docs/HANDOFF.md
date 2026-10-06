# Handoff (2026-10-04)

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
  kept; no ABI change). The shipped calibration is the combined refit (NOTES-B2 §10;
  hold-out targets met, scaled power form e_tc 0.708 pJ/op + e_byte^f 301 pJ/B, γ = 1).
  Open: recollect the 5 N = 16 sustained windows (the refit used the 9 N ≤ 14 windows of
  the hotclock run); W = 1 for N ≤ 8 (NOTES-BFEPS §4).
- **Verification on dev:** 8 CPU + 28 GPU ctest tests, 284 TLC models, 20 Apalache checks,
  oracle suite, 26 params certificates, all passing (b2-accuracy: 9 CPU + 29 GPU ctest tests, 27 certificates) (`tools/test.sh`, `spec/check.sh`,
  `spec/apalache/check.sh`, `spec/test_oracle.py`, `spec/params.py --check`).
- **D3 gate (Eric, 2026-10-04):** D3 begins with a design/theory discussion with Eric to narrow
  the questions. Do not start D3 agents or specs before that.
- **Roadmap done:** A1, A2, A3, A4, A5 (2026-10-04, docs/NOTES-A5.md), D1, D4 part 1. Next by Eric's priority: D3 (packed
  symmetric contractions; design preference: one generic kernel, else split; TLA+ first),
  then B2 (performance model that ehrenfest can call), B1, then C1 after Ozaki-II is polished.
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

## Machine notes

- RTX 4090 shared with the ehrenfest session(s): all GPU work under
  `flock /home/ef/repos/ehrenfest-ipdps/_wip/gpu.lock` (the tools take it themselves; never
  nest it around them).
- Heavy jobs run inside `tools/memcap.sh` (shared cgroup budget, RAM/2). The tools already do.
- PCIe was stuck at gen1 on 2026-10-03 and recovered after the reboot (gen4 x16, 21 GB/s).
- Nsight Systems at `~/nsight-systems-2026.5.1/target-linux-x64/nsys`; `ncu` in the immerdink
  env; GPU counters enabled for all users.
- Benchmarks: `bench/run_sweep.sh` / `bench/run_ab.sh` (quiet-gated, warm-up, retries,
  clock-state records). Correctness tests are not quiet-gated.

## Resuming this conversation

The original session can be resumed from the CLI (sessions are keyed by directory):
`cd /home/ef/repos/immerdink/.claude/worktrees/ozaki-kernels-immerdink-1089fd && claude --resume b6642cd7-986f-4a6d-a271-daef6ce2dedf`.
If that fails, start a new session in the repo and read this file.
