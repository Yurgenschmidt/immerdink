# Benchmarking immerdink

How to measure immerdink's performance reliably. This file covers methodology only. The
benchmark here measures immerdink alone and needs no other library. Comparison campaigns
against other libraries, their harness and their raw results are kept outside this
repository, in a separate local evaluation workspace.

## The tool

`bench/imd_bench` times one contraction case per process. Each run uses CUDA events on a
non-blocking stream: warm-up calls, then timed repetitions, capped at `--max-seconds`. It
reports the median, min, max and spread (`(max − min) / median`) as JSON lines.

Every record carries:
- the shape and GEMM view, N, the method and variant;
- TFLOPS-equivalent (`2·m·n·k·b / t_median`), the workspace bytes and the accuracy;
- the machine conditions, the git commit, the env, and the GPU, driver and CUDA versions;
- peak host RSS and peak cgroup memory.

| method | what it times |
|---|---|
| `imd` (default) | `imd_plan_create` once, then `imd_contract`. Per N in `--N`, and per `--vec`, `--g` and `--k4` value. `--phase-reps` extra calls with the library's per-phase event marks fill `phases` (built with `IMMERDINK_PROFILE`, which the bench enables). |
| `packed` | D1 operand reuse: R × `imd_contract` against `imd_pack` + R × `imd_contract_packed` (`--pack A,B`, `--reuse 1,4,16`), interleaved per repetition; the packed C is checked bit for bit against the plain one. |
| `duel` | Two immerdink configurations interleaved call by call in one process (`--duel-a`, `--duel-b`; see "Tiny shapes" below). |
| `bw` | A device-to-device copy of A: the achievable DRAM bandwidth, a hardware reference. |

- **Data:** `(u − ½)·exp(φ·g)`, with `u ~ U[0,1)`, `g ~ N(0,1)` and φ = 0.5. It is generated
  on the device from a counter-based hash, so it is reproducible.
- **Accuracy:** 256 sampled outputs, or all of them when C is small, against a double-double
  (Dot2) reference computed on the device. Two metrics are reported:
  - the max normwise error `|c − r| / (‖a_i‖₂‖b_j‖₂)`, the DESIGN §3.8 metric;
  - the max relative error `|c − r| / |r|`.
- **Memory:** the host never holds an operand. Each method's buffers are freed before the
  next method runs.

## How to run

```sh
bench/run_sweep.sh [sweep file] [out.jsonl]   # default: bench/sweep_imd.txt
CASES='sq8192|t_cc' bench/run_sweep.sh        # a subset
bench/run_ab.sh <before imd_bench> <after imd_bench> <sweep> <before.jsonl> <after.jsonl>
python3 bench/summarize.py <run>.jsonl [--before <older>.jsonl]
python3 bench/ab_ratio.py <before.jsonl> <after.jsonl>
```

- A sweep file has one case per line: `name | einsum spec | extents | methods | N list | extra
  imd_bench args`.
- `bench/run_ab.sh` runs each case with the before build, then with the after build, as
  separate quiet-gated processes. Both run the case's methods; `AFTER_METHODS` and
  `AFTER_ARGS` change the after side only.

- `bench/` is a standalone CMake project (`bench/build.sh`).
  - `IMMERDINK_LIBRARY=/path/libimmerdink.so` links a prebuilt library.
  - `IMMERDINK_CUTLASS_DIR` points at a CUTLASS checkout.
- **Tiny shapes (≲ 100 µs):** a tiny call moves by ±10–20 % between performance states that
  last from milliseconds to seconds (host core type, GPU DVFS; NOTES-A5 §1.2). A sequential
  measurement of a few calls samples one of those states. Two remedies:
  - **To compare two configurations of one build, use `--methods duel`.** It times side A and
    side B alternately, call by call, in one process, so both see the same state, and it
    reports the median over pairs of `t_a / t_b` (`ratio_median`, > 1: B is faster) with
    `ratio_p10` and `ratio_p90`. It also checks both sides' accuracy, whether their C are
    bit-identical, and each side's plan bytes and kernel launches.
    - A side is `default` or a comma list of overrides: `N=<n>`, `vec=generate|resident`,
      `g=<g>` (0 = N), `k4=rule|tune|0..3`, `k5=0|1|2|4` (0 = the launch rule). Unset keys
      keep the plan defaults, and N defaults to each `--N` value. Forced `k4` and `k5` use the
      profiling hooks (the default build).
    - The defaults are `--duel-a default --duel-b k4=tune`. This compares the two K4
      selection mechanisms that the library ships: the fitted rule, and the optional
      plan-time autotune (`k4_tune`, off by default). That choice is made per shape, its
      effect on tiny calls is a few percent, and only an interleaved measurement resolves
      it. When both sides pick the same configuration (`k4_config` in both records), the run
      is an A/A check that shows the case's own resolution.
    - Other uses:
      - `--duel-b k4=2` (or 0, 1, 3) checks the rule against one forced configuration.
        In one ungated run at s512 and N = 8, k4=2 gave 0.964 [0.941, 0.987]: the rule was
        3.6 % faster.
      - `--duel-a N=8 --duel-b N=14` gives the cost of more moduli.
      - `--duel-b vec=resident` gives the time a resident plan saves per call, against its
        exact plan bytes. On the KF small path, K0 is not a separate launch (the same launch
        count on both sides), so expect about 1.00 there.
      - `--duel-b k5=1` checks the K5 elements-per-thread rule.
      - `--duel-b default` (A = B) gives the case's own resolution. In the same run it gave
        1.005 [0.985, 1.038].
    - Use `--reps 300` or more. The duel adds at least 20 warm-up pairs and a second C
      buffer. Through the sweep scripts, set `BENCH_RETRIES=1`: across hundreds of tiny
      calls, max − min always exceeds 5 % of the median, and the ratio does not need the
      retries.
  - **To compare two builds (before / after a change), use `bench/run_ab.sh`.** For tiny cases
    use many timed calls (`--reps 300` in the sweep's extra arguments) and repeat the A/B,
    because each side is its own process.
- **Profiling:**
  - `bench/nsys_case.sh` and `bench/nsys_sweep.sh` (Nsight Systems; add
    `--gpuctxsw=true` to see context switches) and `bench/ncu_sass.sh` (Nsight Compute;
    GPU counters must be enabled for all users);
  - `bench/clocklog.sh` logs clocks during a run;
  - `bench/k5bimodal/` holds the context-switch analysis tools.

## Getting clean numbers

1. **No desktop on the GPU.** Display renderers (a desktop session, browser or app GPU
   processes) time-slice the GPU in ~2.2 ms slices, costing a running kernel 0.4–0.5 ms per
   switch (docs/NOTES-K5-bimodal.md). Run in a console-only session, or with the display on
   another GPU.
2. **Persistence mode on** (`nvidia-smi -pm 1`), so GPU state survives between benchmark
   processes.
3. **The quiet gate** (`tools/run_bench.sh` → `tools/run_quiet.sh` → `tools/quiet_check.sh`):
   - it takes a machine-wide lock and the shared GPU lock;
   - it waits for a low load average, no CPU hog, a low clock-weighted GPU load, and no large
     GPU compute processes;
   - then up to 300 s for raw GPU utilization ≤ 1 %;
   - it records the conditions, and never bypasses the gate.
4. **Warm-up and retries** (`bench/warm_run.sh`, used by the sweep and A/B scripts):
   - a fixed 3 s dense warm-up in the same lock hold (`BENCH_WARMUP_S`);
   - up to 3 attempts while the spread is > 5 % (`BENCH_RETRIES`, `BENCH_MAX_SPREAD`);
   - records get `attempts` and `contaminated`, and a `clock_state` line follows each case.
     In a clean bench-mode session, `contaminated` on tiny shapes usually just means jitter
     above 5 %.
5. **Clock locking**, for publication-grade numbers: `sudo nvidia-smi -lgc <f>,<f>` at a
   frequency the card can sustain under its power cap; reset with `-rgc`. It shows up as
   `applications_clocks_setting: Active` in `clock_state`.

## Memory and GPU sharing

- Builds and runs go through `tools/memcap.sh`, a shared cgroup budget (default RAM/2, swap
  off). Builds use at most 4 jobs.
- When the GPU is shared with other projects, GPU work takes the advisory lock in
  `IMMERDINK_GPU_LOCK`; the tools take it themselves.

## Where past results went

Until 2026-10-04 the repository held every campaign's raw results (`bench/results/`), the
campaign sweep files (`bench/sweep_compare.txt`, `sweep_m4`, `sweep_small`, `sweep_a*`,
`sweep_d1*`, `sweep_g`, `sweep_v12`), the comparison write-up (`docs/BENCHMARKS.md`) and the
GEMMul8 fetch script (`tools/get_gemmul8.sh`). They moved to the local evaluation workspace
(`immerdink-eval/legacy/`), which also holds new comparison campaigns. The NOTES files keep
their historical references to those paths; the numbers they quote are summaries of those runs.

The comparison methods of `imd_bench` went the same way. Up to `db8bfbd` the bench also timed
native cuBLAS DGEMM (`dgemm`), cuBLAS FP64 emulation (`emu`), a cuBLASLt int8 roofline
(`int8`), cuTENSOR (`cutensor`) and GEMMul8 (`gemmul8`), plus `perm+X` wrappers that permute
the operands to the GEMM view. Its `duel` then meant imd against GEMMul8. The evaluation
workspace keeps that version of the bench. NOTES files that quote those methods, or
`AFTER_METHODS=gemmul8,imd`, refer to it.
