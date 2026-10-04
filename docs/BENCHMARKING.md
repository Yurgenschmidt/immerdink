# Benchmarking immerdink

How to measure immerdink's performance reliably. This file covers methodology only.
Comparison campaigns against other libraries, and their raw results, are kept outside this
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

Method `imd` creates a plan once, then times `imd_contract`. `--phase-reps` extra calls with
the library's per-phase event marks fill `phases` (built with `IMMERDINK_PROFILE`, which the
bench enables). Other methods (native cuBLAS, cuBLAS emulation, cuTENSOR, and optional
external baselines when their dependencies are present) exist for reference.

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
bench/run_sweep.sh [sweep file] [out.jsonl]   # default: bench/sweep_imd.txt (immerdink only)
CASES='sq8192|t_cc' bench/run_sweep.sh        # a subset
bench/run_ab.sh <before imd_bench> <after imd_bench> <sweep> <before.jsonl> <after.jsonl>
python3 bench/summarize.py <run>.jsonl [--before <older>.jsonl]
```

- `bench/` is a standalone CMake project (`bench/build.sh`).
  - `IMMERDINK_LIBRARY=/path/libimmerdink.so` links a prebuilt library.
  - `IMMERDINK_CUTLASS_DIR` points at a CUTLASS checkout.
- **Tiny shapes (≲ 100 µs):** use `--methods duel`, which alternates two methods call by call in
  one process (NOTES-A5 §1.2). Sequential runs sample per-process performance states.
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
