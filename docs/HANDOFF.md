# Handoff (2026-10-04)

State of immerdink for a fresh session, written before switching the dev box to a
console-only (no graphics) setup. Read this, `docs/ROADMAP.md` and the auto-loaded memory
first.

## Where things stand

- **Branches:** all work is on `dev` (pushed). `main` = PR #1 (v1.3 state); later work reaches
  main by another PR from dev. Never push to main directly.
- **DESIGN v1.7** (docs/DESIGN.md): v1.4 KF fused small path, v1.5 K4 configs and rule, v1.6
  KL fused large path, v1.7 §6b packed operands (implemented).
- **Host API v5** (`include/immerdink/immerdink_host.h`, CUDA-free `libimmerdink_host`, Python
  `python/immerdink/host.py`): the single source of truth for lowering, legality, workspace,
  plan bytes, K4 config, launch counts and packed-operand sizing and compatibility.
- **Verification on dev:** 6 CPU + 26 GPU ctest tests, 284 TLC models, 20 Apalache checks,
  oracle suite, 26 params certificates, all passing (`tools/test.sh`, `spec/check.sh`,
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
