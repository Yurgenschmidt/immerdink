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
| `imd` (default) | `imd_plan_create` once, then `imd_contract`. Per N in `--N`, and per `--vec`, `--g` and `--k4` value (`--mode direct`: the DIRECT kernel). `--phase-reps` extra calls with the library's per-phase event marks fill `phases` (built with `IMMERDINK_PROFILE`, which the bench enables). |
| `packed` | D1 operand reuse: R × `imd_contract` against `imd_pack` + R × `imd_contract_packed` (`--pack A,B`, `--reuse 1,4,16`), interleaved per repetition; the packed C is checked bit for bit against the plain one. |
| `duel` | Two immerdink configurations interleaved call by call in one process (`--duel-a`, `--duel-b`; see "Tiny shapes" below). |
| `bw` | A device-to-device copy of A: the achievable DRAM bandwidth, a hardware reference. |

- **Data:** `(u − ½)·exp(φ·g)`, with `u ~ U[0,1)`, `g ~ N(0,1)` and φ = 0.5. It is generated
  on the device from a counter-based hash, so it is reproducible. `--data dominant` gives each
  row one entry 2²⁰ larger; `--data decay_opp` (C1 M-c) is `bench/accuracy/c2_fixtures.py`'s
  `decay_opp`: magnitudes falling over 9 decades along K in A and rising in B, and over 3
  decades along the rows.
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
      `g=<g>` (0 = N), `k4=rule|tune|0..3`, `k5=0|1|2|4` (0 = the launch rule), `ka=auto|16|64` (v1.13 §4b: the side's K alignment through the host library's test hook at plan creation; times A = 16 against A = 64 in one process). Unset keys
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

## Calibrating the performance model (DESIGN §6c)

The performance model in `libimmerdink_host` (`imd_perf_predict`, `imd_perf_rank`; Python
`host.predict`, `host.rank`) needs one calibration per GPU. The RTX 4090 calibration ships
with the library (`src/host/calib/rtx4090_sm89.calib`, embedded at build time). For another
GPU, or after a kernel change that moves the per-phase times:

```sh
tools/calibrate.sh [-o my_gpu.calib] [--validate]
```

- It builds the profiling `imd_bench`, runs `bench/sweep_calib.txt` through
  `bench/run_sweep.sh` (quiet gate, GPU lock, warm-up: the usual rules above), and fits the
  parameters with `bench/perf_fit.py` (numpy only): per kernel class a non-negative
  least-squares fit in relative error of the phase times (K4: a DRAM-rate and an effective-L2
  scan), then the per-call term on the timed medians. About 12 minutes of GPU time on an
  RTX 4090; it never changes clocks (the file's `clocks` line records the state it ran in).
  The records carry the GPU's SM count and L2 size; for a GPU other than the RTX 4090 pass its
  datasheet peaks (`perf_fit.py --peak-dram --peak-int8 --peak-fp64`).
- `--validate` also runs the hold-out sweep `bench/sweep_perf_validate.txt` (shapes disjoint
  from the fit set) and prints `bench/perf_validate.py`'s error table against the DESIGN §6c.5
  targets; `--per-call` and `--phases` show every call and every phase.
- Use a calibration file with `imd_perf_calib_load` (C) or `host.Calibration.load(path)`
  (Python), passed to every predict / rank call; replacing the shipped file and rebuilding
  changes the default. The datasheet peaks in the file (`peak_*`) only scale the reported
  bounds; edit them for another GPU.
- Results of past calibrations: docs/NOTES-B2.md.

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
`sweep_d1*`, `sweep_g`, `sweep_v12`) and the comparison write-up (`docs/BENCHMARKS.md`). They
moved to a local evaluation workspace outside this repository (`legacy/` there). The NOTES
files keep their historical references to those paths; the numbers they quote are
immerdink's own results from those runs. Comparisons with other libraries are kept outside
this repository.

The comparison methods of `imd_bench` went the same way: up to `db8bfbd` the bench also
timed other libraries (and `perm+X` wrappers that permute the operands to the GEMM view),
and its `duel` paired imd with one of them. The evaluation workspace keeps that version of
the bench. NOTES files that mention other libraries' calls in the same process refer to it.

## Test definitions

The metrics suite is two sweeps of `bench/imd_bench`: throughput and accuracy
(`bench/sweep_imd.txt`) and board energy (`bench/sweep_energy.txt`). Both are driven by
`bench/run_metrics.sh` (see "Recreating the metrics on another machine").

**What every test is.** One *case* is one FP64 contraction `C = A·B` (an einsum spec on row-major
operands) computed by immerdink's Ozaki-II emulation (INT8 tensor cores) with N moduli. Each case
runs in its own process. Tensor cases are lowered to a GEMM view `m×n×k` with batch `b`; the
TFLOPS-equivalent numerator is always `2·m·n·k·b` (the FP64 operation count), whatever N is.

- **Data:** `(u − ½)·exp(φ·g)`, φ = 0.5, `u ~ U[0,1)`, `g ~ N(0,1)`, generated on the device from a
  counter-based hash (seed 1): every run sees the same inputs.
- **Accuracy:** against a double-double (Dot2) reference on the device, for 256 sampled outputs
  (all of C if it is smaller; 262 144 in the energy sweep). The metric is the max normwise error
  `|c − r| / (‖a_i‖₂ ‖b_j‖₂)`; the max relative error `|c − r| / |r|` is recorded too.
  "DGEMM-level" means at or below cuBLAS DGEMM's normwise error on the same inputs (about 3.5e-16
  at φ = 0.5); N = 14 is the smallest N that reaches it for this data.

### Cases

| case | einsum | extents | GEMM view m×n×k (b) | what it is |
|---|---|---|---|---|
| s256 | `ik,jk->ij` | i=j=k=256 | 256³ | tiny GEMM (µs scale) |
| s512 | `ik,jk->ij` | i=j=k=512 | 512³ | small GEMM |
| s_cc16 | `kjbc,kiac->abij` | i=j=k=16, a=b=c=96 | 1536³ | small coupled-cluster-style contraction |
| s_batch | `lik,ljk->lij` | i=j=k=64, l=256 | 64³, b=256 | tiny batched GEMM |
| sq1024 | `ik,jk->ij` | i=j=k=1024 | 1024³ | square GEMM |
| sq4096 | `ik,jk->ij` | i=j=k=4096 | 4096³ | square GEMM |
| sq8192 | `ik,jk->ij` | i=j=k=8192 | 8192³ | large square GEMM |
| r_lowk | `ik,jk->ij` | i=j=8192, k=1024 | 8192×8192×1024 | low k (CRT-bound) |
| r_bigk | `ik,jk->ij` | i=j=2048, k=32768 | 2048×2048×32768 | big k |
| r_skinny | `ik,jk->ij` | i=16384, j=512, k=4096 | 16384×512×4096 | skinny |
| r_mm | `ik,jk->ij` | i=1000, j=64, k=50000 | 1000×64×50000 | matrix-vector-like, memory-bound |
| t_batch | `lik,ljk->lij` | i=j=k=1024, l=16 | 1024³, b=16 | batched GEMM |
| t_ikl | `ikl,kjl->ijl` | i=j=k=1024, l=16 | 1024³, b=16 | batched GEMM, batch index last |
| t_df | `acQ,bdQ->acbd` | a=b=c=d=64, Q=2048 | 4096×4096×2048 | density-fitting contraction |
| t_cc | `kjbc,kiac->abij` | i=j=k=32, a=b=c=128 | 4096³ | CC particle-hole contraction |
| t_permout | `abk,kcd->cadb` | a=b=c=d=64, k=4096 | 4096³ | permuted output |

The energy sweep runs sq2048 (2048³, `ik,jk->ij`), sq4096, sq8192, t_df, t_cc, t_batch and s_batch.

The native batched baseline (`cublasGemmStridedBatchedEx` FP64) and every other comparison library
(cuBLAS DGEMM, GEMMul8, cuBLAS emulation, cuTENSOR) exist only in the evaluation workspace's
`eval_bench`; this repository's sweeps measure immerdink only.

### Test 1: throughput and accuracy (`bench/sweep_imd.txt`)

For each case and N ∈ {8, 14}: 2 warm-up calls, then timed calls with CUDA events on a non-blocking
stream, at least 3 and at most 10 repetitions, capped at 15 s per method (6 s for sq8192). The
median is reported with min, max and the spread `(max − min)/median`. Then 5 profiled calls give
the per-phase times (K0 vec, K1 scale, K2/K3 pack, KL, K4 INT8 GEMM, K5 CRT) against the
memory-model lower bounds. Before each case: a 3 s dense warm-up in the same lock hold, and up to 3
attempts while the spread exceeds 5 % (records get `attempts` and `contaminated`).

Reported: median time, `TFLOPS-eq = 2·m·n·k·b / median`, spread, both error metrics, workspace
bytes, phase times, and per case a clock-state line after the warm-up (SM MHz, power, temperature,
event reasons). s256, s512 and s_batch take tens of µs per call: their spread is jitter-dominated
(often flagged `contaminated`); use `--methods duel` for a stable A/B number on those.

### Test 2: board energy (`--methods energy`, `bench/sweep_energy.txt`)

Per case, for N ∈ {8, 11, 13, 14, 16}, one N at a time: an accuracy check, a 3 s back-to-back
warm-up, then back-to-back calls for at least 10 s (at least 50 calls while that fits in 30 s),
queued ahead so the GPU never idles. Energy is the NVML total-energy counter
(`nvmlDeviceGetTotalEnergyConsumption`) over whole counter steps with exactly two reads: the counter
advances in steps on a firmware timer (about 100 ms on an RTX 4090) and every read costs a little
energy, so the window starts at an observed step and ends at a step, with no reads inside (details
in the comment above `struct ERead` in `bench/imd_bench.cu`). The step period is **measured at idle
at the start of every case** (30 consecutive steps; the run fails with a clear error if the counter
does not step regularly) unless `--energy-step-ms` sets it; each record carries
`energy_counter.step_period_ms`, `step_period_calibrated` and `next_step_resid_ms` (the step after
the window must fall on the step grid).

Reported (per case and N): `gflops_per_w = 2·m·n·k·b / (energy per call)` (GFLOPS-eq per watt =
TFLOPS-eq per kW), `energy_j_per_call`, `avg_power_w`, the sustained `tflops_eq_mean` (back-to-back,
gaps included), the same quantities above the case's awake-idle power (an idle measurement per case,
record `method: idle`), the SM clock and clock-event reasons over the window (`nvml`), and the
instant-power cross-check `avg_power_w_inst`. The energy counter is a property of the GPU and
driver: on a part whose counter is missing or does not step (some data-centre or virtualized GPUs),
the energy records are errors, not wrong numbers.

### Machine conditions

Every case runs through `tools/run_bench.sh` (see "Getting clean numbers"): machine-wide lock,
shared GPU lock, a wait for low load, no CPU hog, no compute processes and raw GPU utilization ≤ 1 %.
Note whether the clocks were locked (`applications_clocks_setting` in the `clock_state` lines,
`conditions.txt` of `bench/run_metrics.sh`): with the clocks free, large shapes sit at the power
limit and efficiency tracks throughput. The per-record provenance (commit, driver, CUDA, GPU and
VBIOS, clocks, conditions) is in the JSONL.

## Recreating the metrics on another machine

```sh
# 1. Build for the GPU's architecture into a fresh build directory (default sm_89; Blackwell
#    workstation parts are sm_120). IMMERDINK_ENV is a prefix with bin/nvcc, gcc, g++ and python3 (a
#    conda env); CUTLASS comes from third_party/cutlass or IMMERDINK_CUTLASS_DIR.
CMAKE_CUDA_ARCHITECTURES=120 IMMERDINK_BENCH_BUILD=$HOME/imd-build-sm120 bench/build.sh

# 2. Lock the clocks (needs root) at a frequency the card sustains under its power cap;
#    verify with nvidia-smi -q -d CLOCK, undo afterwards with -rgc.
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc <f>,<f>

# 3. Run both sweeps; the output directory must be outside the repository.
#    (a dedicated node with no other GPU jobs: IMMERDINK_GPU_LOCK=none skips only the shared
#    advisory lock, the quiet checks stay)
CMAKE_CUDA_ARCHITECTURES=120 IMMERDINK_BENCH_BUILD=$HOME/imd-build-sm120 \
  bench/run_metrics.sh $HOME/imd-metrics/$(hostname)-$(date +%Y%m%d)

sudo nvidia-smi -rgc
```

`SWEEPS=imd` or `SWEEPS=energy` runs one sweep; `CASES='sq8192|t_cc' bench/run_sweep.sh
bench/sweep_energy.txt out.jsonl` runs a subset. The architecture is a build option only: the
library's kernels are developed on sm_89, so a new architecture has to pass the correctness tests
(`tools/test.sh`) before its numbers mean anything. If the machine never gets quiet, the sweep stops
with exit 3 instead of recording contaminated numbers.
