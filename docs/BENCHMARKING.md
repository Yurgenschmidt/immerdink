# Benchmarking immerdink

How to measure immerdink's performance reliably. This file covers methodology only. The
benchmark measures immerdink alone and needs no other library; the comparison baselines
(cuBLAS DGEMM, cuBLAS FP64 emulation, GEMMul8, cuTENSOR native and emulated) are optional extras
that the same tool can time on identical inputs (Test 3 and Test 4 below). The earlier
campaigns on the RTX 4090, their harness and their raw results are kept outside this
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
| `energy` | Board energy per call of imd at each N (Test 2). |
| `cutensor`, `cutensor-emu` | Optional baselines (built with cuTENSOR): cuTENSOR native FP64 and its INT8 emulation, on the original layouts (Test 3). |
| `dgemm` | Optional baseline: cuBLAS FP64 GEMM (`CUBLAS_COMPUTE_64F_PEDANTIC`; strided-batched when b > 1) (Test 4). |
| `emu` | Optional baseline: cuBLAS FP64 fixed-point emulation, one record per `--emu` variant. Needs a CUDA 13 cuBLAS (Test 4). |
| `gemmul8` | Optional baseline: GEMMul8 (INT8 backend), `--g8mode fast,accu`, `--N` moduli. Built in with `GEMMUL8_DIR` (Test 4). |
| `energy-cutensor`, `energy-cutensor-emu`, `energy-dgemm`, `energy-emu`, `energy-gemmul8` | The energy protocol of Test 2 for each baseline. |

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
  - `bench/profile_nsys.sh` (Test 5, run by `bench/run_metrics.sh` as its last sweep): short nsys runs of
    immerdink and the baselines, summarized by kernel category;
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
   - up to 3 attempts (`BENCH_RETRIES`) while more than 10 % of the case's records (`BENCH_MAX_BAD_FRAC`, 0.1)
     have a spread above 5 % (`BENCH_MAX_SPREAD`); the attempt kept has the most ok records, then the fewest
     noisy ones (`bench/warm_run_pick.py`);
   - records get `attempts` and `contaminated` (the record's own spread above the limit, or every record when the
     kept attempt still has too many noisy ones), and a `clock_state` line follows each case.
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
the bench (`eval_bench`, which also has the interleaved `pareto` method and the 4090
campaigns). NOTES files that mention other libraries' calls in the same process refer to it.
The baselines came back to `imd_bench` as optional methods (Test 3 and Test 4); `duel` stays
immerdink against immerdink, and the interleaved `pareto` timing stays in `eval_bench`.

## Test definitions

The metrics suite is sweeps of `bench/imd_bench`: throughput and accuracy
(`bench/sweep_imd.txt`), board energy (`bench/sweep_energy.txt`), the cuTENSOR baselines
(`bench/sweep_baselines.txt`), the comparison with all baselines (`bench/sweep_compare.txt`,
`bench/sweep_compare_energy.txt`) and the nsys profile pass (`bench/sweep_profile.txt`). All are driven by `bench/run_metrics.sh` (see "Recreating the
metrics on another machine").

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

Besides immerdink, the sweeps can measure cuTENSOR native and emulated FP64 (Test 3), and cuBLAS DGEMM
(`cublasGemmEx`, and `cublasGemmStridedBatchedEx` on the batched cases), cuBLAS FP64 emulation and GEMMul8
(Test 4): all optional, each compiled in only where its library is available.

### Test 1: throughput and accuracy (`bench/sweep_imd.txt`)

For each case and N ∈ {8, 14}: 2 warm-up calls, then timed calls with CUDA events on a non-blocking
stream, at least 3 and at most 10 repetitions, capped at 15 s per method (6 s for sq8192). The
median is reported with min, max and the spread `(max − min)/median`. Then 5 profiled calls give
the per-phase times (K0 vec, K1 scale, K2/K3 pack, KL, K4 INT8 GEMM, K5 CRT) against the
memory-model lower bounds. Before each case: a 3 s dense warm-up in the same lock hold, and up to 3
attempts while more than 10 % of the case's records have a spread above 5 % (records get `attempts` and
`contaminated`; see "Getting clean numbers" for the exact rule).

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

### Test 3: cuTENSOR native and emulated FP64 (`bench/sweep_baselines.txt`, energy sweep)

The only comparison libraries in this repository are the two cuTENSOR modes (optional: `imd_bench`
is built with them only if `cutensor.h` and `libcutensor` are found, `CUTENSOR_ROOT`; otherwise those
methods are compiled out and a run asking for them records an error). Same cases, data, reference
and timing code as Test 1, with the einsum given to cuTENSOR directly (no GEMM lowering, no extra
permutations):

- `cutensor`: `cutensorContract` with `CUTENSOR_COMPUTE_DESC_64F` (native FP64; this is also the
  "DGEMM-level" accuracy reference, about 3.5e-16 normwise at φ = 0.5).
- `cutensor-emu`: the same with `CUTENSOR_COMPUTE_DESC_8XINT8`, cuTENSOR's INT8-based FP64 emulation
  (cuTENSOR ≥ 2.4; its header lists it as Hopper+; the number of retained mantissa bits is chosen at
  runtime to match native accuracy). Since cuTENSOR 2.6 it **falls back to native FP64 silently**
  where it does not apply, so a record alone cannot tell whether the emulation ran:
  the emulated record carries an **identical-results test**: after the timed loop the bench runs the
  native plan once on the same inputs and compares the whole C bit for bit (`identical_to_native`,
  `native_c_hash`; `c_hash` is a hash of every bit of C in every record with an accuracy;
  `repeat_identical`: a second call of the same plan gives the same bits). `bench/summarize.py` reads
  that: an emulated C equal to the native one is "NOT engaged" (certain). A C that differs does not prove
  that the emulation ran: on the RTX 4090 (not Hopper+) `cutensor-emu` returned a C different from the
  native plan's but identical to cuBLAS DGEMM's, with the native time and error (another native
  algorithm); so a different C counts as "likely engaged" only when the time or error also differ, and the
  Nsight Systems kernel list of the `cutensor-emu` run (INT8 tensor-core GEMM kernels) is the decisive
  check. Only without those fields does the summary fall back to the older rule (time within 5 % of
  native, same error). A plan that cuTENSOR refuses is an error record, not a failure of the sweep.
- Energy: methods `energy-cutensor` and `energy-cutensor-emu` in `bench/sweep_energy.txt` use the
  protocol of Test 2 (same process, same idle baseline).

Records carry `compute_desc`, `cutensor_version` and `emulated_requested`. cuTENSOR's emulation needs
a recent cuTENSOR and CUDA (its documentation: it falls back to native FP64 silently where the CUDA
version or the device does not support it, and recommends the newest CUDA); on a CUDA 12.9 toolchain it
did not engage on any card in the first PACE run (errors identical to native). Run it with the CUDA 13
toolchain (Test 4) and read the engaged check.

### Test 4: comparison with the baselines (`bench/sweep_compare.txt`, `bench/sweep_compare_energy.txt`)

The comparison of the RTX 4090 campaigns, reproducible on any card: immerdink against

| method | baseline | notes |
|---|---|---|
| `gemmul8` | GEMMul8 v3.5.2 (commit `6a9eff8`, API generation 3, INT8 backend), fast and accurate mode | optional: `GEMMUL8_DIR`; the pin is `bench/GEMMUL8_PIN`; records carry `g8_name`, `g8_commit`, `g8_api`, `g8_mode` |
| `dgemm` | cuBLAS FP64 GEMM, `CUBLAS_COMPUTE_64F_PEDANTIC` | always built |
| `emu` | cuBLAS FP64 fixed-point emulation, `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, EAGER strategy, dynamic (`dyn`) or fixed (`fix<bits>`) mantissa control | needs a **CUDA 13 cuBLAS** (the build tests for the API); otherwise a recorded error "needs CUDA 13 cuBLAS", never fatal |
| `cutensor`, `cutensor-emu` | cuTENSOR native FP64 and `CUTENSOR_COMPUTE_DESC_8XINT8` (Test 3) | optional: cuTENSOR |

`energy-dgemm`, `energy-emu`, `energy-gemmul8` (with `energy-cutensor`, `energy-cutensor-emu`) measure the
same configurations with the energy protocol of Test 2.

**The rules that make it a fair comparison** (the rules of the evaluation campaigns, kept so that the
results are comparable with them):
- *Same inputs, same reference.* Every method of a case runs in the case's one process on the same data
  and is checked against the same double-double reference on the same sampled outputs. The comparison
  sweeps check 262 144 outputs (1 048 576 on the squares up to 4096; all of C when it is smaller), not
  `imd_bench`'s default 256: the maximum error depends on the number of checked outputs, so **never
  compare an error of the comparison sweeps with one of `sweep_imd.txt`**.
- *GEMM view and `perm+X`.* `dgemm`, `emu` and `gemmul8` are matrix-only: they need A = [L][I|K],
  B = [L][J|K], C = [L][I][J]. A case in another layout (the tensor cases) is timed as `perm+<method>`:
  cuTENSOR permutations of A and B before the GEMM and of C after it are inside the timed window, as
  the price of using a matrix library on a tensor contraction (immerdink and cuTENSOR contract the
  original layouts directly). `--gemm-view 1` also records the bare GEMM on pre-permuted operands.
  Without cuTENSOR in the build, a `perm+` case is an error record.
- *Batched cases.* `dgemm` and `emu` use `cublasGemmStridedBatchedEx`; GEMMul8 has no batched API (one
  call per batch slice, noted in the record).
- *Flops.* `2·m·n·k·b` for every method, whatever N or mantissa width: TFLOPS-eq.
- *Configurations.* `N` is the number of moduli for `imd` and `gemmul8` (`--N`, or `--g8fast-N` /
  `--g8accu-N` per mode); for `emu`, `N` in the record is the mantissa bit count of `fix<bits>` (0 for
  `dyn`) and the emulation workspace is `--emu-ws-mib` (default 2048; use 4096 at 16384³). The N and
  bit counts of `sweep_compare.txt` bracket DGEMM-level accuracy at φ = 0.5 (N = 14 or 15, fix55; N = 13
  and fix47 fail it) plus lower-accuracy points.
- *DGEMM-level.* Per method, the fastest (energy: the most efficient) configuration whose max normwise
  error is at or below cuBLAS DGEMM's own error on the same case. `bench/summarize.py --only-compare
  compare.jsonl` makes the table (also for `compare_energy.jsonl`): per case and method the best such
  configuration with TFLOPS-eq and time (or GFLOPS-eq per watt and joules), the ratio of immerdink to the
  best emulator, to DGEMM and to cuTENSOR native, then every configuration with its flag. A method that
  did not run (not built in, unsupported on this card) gives an error record, a dash in the table, and a
  row in the list of failures; the rest of the comparison is unaffected.
- *Is the cuTENSOR emulation real?* See Test 3: the table has a verdict per case from the
  identical-results test (`identical_to_native`), with `c_hash` as the second source. Identical = not
  engaged; different = "likely engaged" only with a different time or error too. The decisive
  confirmation is the Nsight Systems kernel list of the `cutensor-emu` run (INT8 tensor-core kernels).
- *Is the cuBLAS emulation real?* The `emu` method sets the EAGER strategy ("emulate whenever possible"); where
  cuBLAS does not emulate (device, shape, workspace) it presumably runs native FP64 without an error (not
  verified on any card). `bench/summarize.py` compares the `c_hash` of every `emu` record with the `dgemm` and
  `cutensor` records of the case: an equal hash means the emulation did not run, the record is marked
  `native_fallback`, cannot be the best emulator, and the table shows "not engaged" (the energy records carry
  no hash, so this check applies to the timing tables only). A different C does not prove that it ran: the
  Nsight Systems run of `emu` (`engaged_int8`) is the decisive check (`bench/sweep_profile.txt`).

**What is not the same as the evaluation workspace's campaigns.** The methods of a case run one after the
other in the process, not interleaved round-robin as `eval_bench`'s `pareto` method does: all see the same
inputs and, after the 3 s warm-up of `bench/warm_run.sh`, a similar GPU state, but the clock can move
between configurations (a power-limited or unlocked card moves 10 to 20 % between performance states), so
the ratios of tiny shapes (calls of tens of µs: s256, s512, s_cc16, s_batch, sq1024) are indicative only;
the records' `contaminated` flags and spreads mark them. The energy runs are sequential by construction.
Check on the RTX 4090 (not a result): the maximum normwise errors of every method equal the campaign's to
all printed digits, so the numerics and the data are the same. At 4096³ (unlocked, same clock state as the
interleaved unlocked campaign) the sequential TFLOPS-eq agree within 3 % for every method (immerdink N = 14
31.8 against 31.7, GEMMul8 fast N = 14 22.9 against 22.4, cuBLAS emulation fix55 19.9 against 20.0, DGEMM 1.21
against 1.22). At 1024³ the sequential numbers are higher than the interleaved ones (GEMMul8 fast N = 14: 12.5
against 7.7; immerdink N = 14: 17.8 against 17.2). The 4096³-sized tensor cases are 5 to 23 % above the
campaign's locked-clock numbers, the clock: unlocked 2.8 GHz against locked 2.2 GHz for the compute-bound
DGEMM. Do not mix the two protocols in one table.

### Test 5: nsys profiles (`bench/profile_nsys.sh`, `bench/sweep_profile.txt`)

Where the GPU time goes, per method, and whether the INT8 tensor cores were used at all. `bench/run_metrics.sh`
runs it last (`SWEEPS` element `profile`; alone: `bench/profile_nsys.sh <outdir>` or
`SWEEPS=profile bench/run_metrics.sh <outdir>`). Each line of `bench/sweep_profile.txt` is one short
`imd_bench` run (5 repetitions after 2 warm-up calls, 4096 error samples, one method and one N or variant)
under `nsys profile -t cuda` (CUDA trace only: no CPU sampling, so no `perf_event_paranoid` requirement), started
through `tools/run_bench.sh` (quiet gate, shared GPU lock). Default list (33 runs, a few seconds each plus
nsys start-up and export): immerdink N = 14 and N = 8 on `sq4096`, `sq8192`, `t_cc`, `t_df`, `s_cc16`,
`r_lowk`; cuTENSOR native, cuTENSOR emulated (`cutensor-emu`), cuBLAS DGEMM, cuBLAS emulation (`dyn`,
`fix55`) and GEMMul8 (fast and accurate, N = 14) on `sq4096` (a plain GEMM) and `t_cc` (a permuted
contraction: the matrix-only methods run as `perm+<method>`); `cutensor`, `cutensor-emu`, `dgemm` on
`sq8192` and `cutensor`, `cutensor-emu` on `t_df`; cuBLAS emulation `fix55` on `sq8192` and on the batched
`s_batch`. `PROFILE_TAGS='<regex>'` selects runs by tag, `PROFILE_LIST` another list.

Output in `<outdir>/profile/` (and `<outdir>/profile.md`, a table of all runs):
- `index.json`: `[{tag, case, method, N, variant, nsys_rep, summary, engaged_int8}]` (paths relative to `profile/`;
  `N` is the moduli count for `imd` / `gemmul8`, null otherwise; `variant` is `fast` / `accu` for GEMMul8 and
  `dyn` / `fix55` for `emu`);
- `<tag>/summary.json`: `{tag, case, method, N, variant, gpu, wall_ms, kernels: [{name, category, time_ms,
  share_pct, count}], engaged_int8, int8_share_pct, windows, bench_status, nsys_version, notes}`;
- `<tag>/records.jsonl` (the run's `imd_bench` records, including errors) and `run.log`;
- `<tag>/<tag>.nsys-rep` only with `PROFILE_KEEP=1` (open it with `nsys-ui`). The reports and the sqlite exports
  (tens of MB) are otherwise deleted after the summary is written.

How a summary is made (`bench/nsys_summarize.py`, standard library only): the sqlite export is read; only GPU
activity (kernels, memcpy, memset) inside the **timed windows** of `imd_bench` counts, so setup, the accuracy
reference, the warm-up and the untimed identity checks (for `cutensor-emu`, a native cuTENSOR call) are excluded,
and the first window is dropped. `time_ms` is the mean time per timed call spent in a kernel name, `count` its
launches per call, `share_pct` its share of the GPU activity time of a call (sum of durations, not the span).
`wall_ms` (the median window, first activity to last, under the profiler) is for orientation: throughput numbers
come from the sweeps. Categories (name patterns in `CATEGORY_RULES`, tested in `bench/test_nsys_summarize.py`;
`python3 bench/nsys_summarize.py classify '<kernel name>'` prints one): `int8_gemm` (cuBLAS / CUTLASS INT8
tensor-core GEMM names with `imma`, `i8816`, `i16832`, `s8s8`, `int8`..., immerdink's `k4_kernel`), `fp64_gemm`
(`dgemm`, `d884`, `f64`..., e.g. cuTENSOR native's `cutlass_80_tensorop_d884gemm`), `conversion_pack_crt` (the
Ozaki-II scaling / residue / pack / CRT kernels of immerdink, GEMMul8 and the cuBLAS emulation), `permute`
(cuTENSOR permutations, transposes), `memcpy_memset`, `other` (a share above 20 % is flagged in the notes: the
name patterns need a look). **`engaged_int8`** is true when `int8_gemm` is at least 10 % of the call's GPU time,
false below that (`int8_share_pct` says how much), null without data. It is the decisive check for
`cutensor-emu`: native FP64 fall-back shows only `fp64_gemm` kernels, an engaged emulation shows INT8 GEMMs
(the `notes` of those tags say which); the identical-results check of Test 3 is the second source.
The name patterns were derived from real profiles on an RTX 4090 (sm_89, CUDA 13.4, cuBLAS 13.8, cuTENSOR 2.8.1:
immerdink, GEMMul8, cuBLAS DGEMM and emulation, cuTENSOR native; there `cutensor-emu` ran only
`cutlass_80_tensorop_d884gemm` kernels, i.e. native FP64, so `engaged_int8` was false); the Hopper and
Blackwell kernel names (`sm90_xmma_*`, `cutlass3x_sm90_*`, `sm100_*`, `nvjet_sm90_*`) are matched by their
published naming conventions but were **not seen** by the author: on a new architecture read the `other`
share first and extend `CATEGORY_RULES` (and its test) with the names the first profile shows.

nsys is looked for in `$NSYS`, `PATH`, `$IMMERDINK_ENV/bin`, `$CUDA_HOME/bin`, `/usr/local/cuda*/bin`,
`/opt/nvidia/nsight-systems*` and `~/nsight-systems-*` (`bench/preflight.sh` checks). Without it the pass logs the
reason, writes `profile/NSYS_MISSING.txt`, an `index.json` and summaries with `engaged_int8: null` and the
reason in the notes, and exits 0: a missing profiler never fails the metrics run. Methods that are not built in
(no GEMMul8, no CUDA 13 cuBLAS, no cuTENSOR) give a summary with the `imd_bench` error in the notes. Variables:
`NSYS`, `PROFILE_KEEP`, `PROFILE_TAGS`, `PROFILE_LIST`, `PROFILE_REPS`, `PROFILE_WARMUP`, `PROFILE_TMP`
(where the reports are written before they are summarized: node-local scratch on an HPC node),
`QUIET_RAW_WAIT` (10 s here, not the 300 s of timed runs: a profile is not a throughput measurement).

### Calibration (`bench/calibrate.sh`)

Before the sweeps `bench/run_metrics.sh` calibrates the GPU: a sustained dense load (4096³, N = 14,
back to back, through the same gate as every timed run) while `nvidia-smi` is sampled at 1 Hz, until
the state is **stable**: over the last 20 s the SM clock varies by ≤ 30 MHz and the temperature by
≤ 1 °C (after ≥ 30 s of load; timeout 180 s). It reports the steady SM clock (median/min/max of the last
30 s), power, temperature, the share of samples with the power-cap / thermal / HW slowdown event
reasons, whether the clocks are locked, and a **recommended lock**: the lowest sustained SM clock minus
30 MHz, floored to a 15 MHz step, with the exact `sudo nvidia-smi -lgc F,F` command (it must be a
supported graphics clock: `nvidia-smi -q -d SUPPORTED_CLOCKS`). Result: `calibration.json`, copied into
`conditions.txt`. With the clocks locked it verifies the lock holds (SM at the lock in every sample, no
cap). It never changes clocks itself; `LOCK_CLOCKS=auto` (passwordless sudo needed) locks at the
recommendation, re-calibrates, and restores the clocks on exit. `CALIBRATE=0` skips it. Each case still
begins with the 3 s warm-up of `bench/warm_run.sh`, now on an already hot GPU.

### Machine conditions

Every case runs through `tools/run_bench.sh` (see "Getting clean numbers"): machine-wide lock,
shared GPU lock, a wait for low load, no CPU hog, no compute processes and raw GPU utilization ≤ 1 %.
Note whether the clocks were locked (`applications_clocks_setting` in the `clock_state` lines,
`conditions.txt` of `bench/run_metrics.sh`): with the clocks free, large shapes sit at the power
limit and efficiency tracks throughput. The per-record provenance (commit, driver, CUDA, GPU and
VBIOS, clocks, conditions) is in the JSONL.

## Recreating the metrics on another machine

Needs only a CUDA toolkit (nvcc >= 12.8, 13.x preferred), a C++20 host compiler (gcc >= 11), cmake >= 3.28
and git; no conda. On an HPC system: `module load cuda gcc cmake` (names vary).

```sh
git clone --branch dev https://github.com/Yurgenschmidt/immerdink.git && cd immerdink
git submodule update --init --depth 1 third_party/cutlass   # login node: needs network

# On the GPU node (e.g. srun --gres=gpu:1 --exclusive --pty bash):
bench/preflight.sh                       # checks toolchain, cuTENSOR, GPU, NVML, sudo, quiet; says what to fix
export CUTENSOR_ROOT=/path/to/cutensor   # only if `module load cutensor` does not make it findable (>= 2.4)
# optional: the GEMMul8 baseline (the pinned commit, bench/GEMMUL8_PIN; clone it on a login node)
git clone https://github.com/RIKEN-RCCS/GEMMul8.git ../gemmul8-src && git -C ../gemmul8-src checkout $(awk '!/^#/{print $2; exit}' bench/GEMMUL8_PIN)
bench/build_gemmul8.sh ../gemmul8-src 90 $HOME/gemmul8-sm90   # 90 = H100/H200, 120 = RTX PRO 6000 Blackwell, 89 = RTX 4090
export GEMMUL8_DIR=$HOME/gemmul8-sm90    # printed by the script's last line
bench/build.sh                           # detects the GPU architecture (120 on RTX PRO 6000 Blackwell)
bench/run_metrics.sh $HOME/imd-metrics/$(hostname)-$(date +%Y%m%d)
```

- `bench/run_metrics.sh <outdir>` builds if needed, then calibrates the GPU (see "Calibration"), runs the
  throughput/accuracy sweep, the cuTENSOR native/emulated baselines (if built with cuTENSOR), the
  comparison sweeps of Test 4 (`compare`, `compare_energy`; the baselines that are not built in record
  errors), the energy sweep and, last, the nsys profile pass of Test 5 (`profile/`; it needs nsys, and without
  it records a note). The output directory must be outside the repository.
- **Toolchain for the full comparison:** the cuBLAS FP64 emulation baseline needs a CUDA 13 cuBLAS, and
  cuTENSOR's INT8 emulation is documented to need a recent CUDA too (nvcc, cuBLAS and cuTENSOR from a CUDA
  13 install, e.g. the conda env named by `IMMERDINK_ENV`): on CUDA 12.x `emu` records "needs CUDA 13
  cuBLAS" and `cutensor-emu` can fall back to native FP64 silently (the engaged check tells). The GEMMul8
  build needs the same nvcc, no GPU.
- **Clock lock (recommended, needs root):** the calibration prints the exact command
  (`sudo nvidia-smi -lgc <f>,<f>`, a frequency the card sustains under its power cap); run it, start again,
  and undo with `-rgc` afterwards. With passwordless sudo, `LOCK_CLOCKS=auto bench/run_metrics.sh <outdir>` does
  lock, re-calibrate and unlock by itself. Without a lock the run still works; say "unlocked" in the report.
- **Dedicated node:** the gate is on by default (quiet machine, machine-wide lock). If the batch system
  already gives you the GPU exclusively, `IMMERDINK_GPU_LOCK=none` skips only the shared advisory lock.
  If the quiet check blocks on harmless node daemons, relax it (`QUIET_LOAD`, `QUIET_CPU`,
  `QUIET_GPU_IGNORE`, see `tools/quiet_check.sh`); never edit it out.
- `SWEEPS="imd energy"` (any of `imd baseline compare compare_energy energy profile`; the default is all six, in
  that order; `profile` is the nsys pass of Test 5) selects sweeps; `CALIBRATE=0` skips the calibration; `CASES='sq8192|t_cc' bench/run_sweep.sh
  bench/sweep_energy.txt out.jsonl` runs a subset. `CMAKE_CUDA_ARCHITECTURES=<n>` overrides the
  detected architecture (use a fresh `IMMERDINK_BENCH_BUILD` directory when changing it).
- **Check the accuracy before believing the speed:** the library's kernels were developed on sm_89, so on a
  new architecture the first thing to read is `max_normerr` in the records (about 1e-16 at N = 14,
  1e-9 at N = 8; `nonfinite` must be 0). `tools/test.sh` runs the correctness tests.
- If the machine never gets quiet, the sweep stops with exit 3 instead of recording contaminated numbers.
