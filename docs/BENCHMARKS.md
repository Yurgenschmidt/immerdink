# Benchmarks

M4 benchmark harness, the M4 sweep (after the M4 kernel work, commit `b29b464`) and the
first baseline sweep (the M1 library, commit `e4d8845`) it is compared with. What changed
and why, step by step with per-phase times: [NOTES-M4.md](NOTES-M4.md).

## What is measured

`bench/imd_bench` times one contraction case per process, for each method and variant.
It uses CUDA events on a non-blocking stream: 2 warmup calls, then up to 10 timed
repetitions, capped at `--max-seconds` of timed work after at least 3 repetitions. It
reports the median, min, max and spread (`(max − min) / median`). Each record is one
JSON line (`bench/results/*.jsonl`). Every record also carries:
- the shape and the GEMM view;
- N, method and variant;
- TFLOPS-equivalent, `2·m·n·k·b / t_median`, for every FP64 method;
- the workspace bytes;
- the accuracy;
- the machine conditions that `tools/quiet_check.sh` recorded right before the run;
- the git commit and conda env;
- the GPU name, max clocks, and clocks/temperature/power/throttle reasons after each
  variant;
- the driver, CUDA runtime, cuBLAS and cuTENSOR versions;
- the process's peak host RSS and its peak cgroup (memcap scope) memory.

| method | what |
|---|---|
| `dgemm` | cuBLAS FP64 GEMM with `CUBLAS_COMPUTE_64F_PEDANTIC` and `CUBLAS_PEDANTIC_MATH`, so an env var can't turn it into emulation |
| `emu dyn` / `emu fixB` | cuBLAS fixed-point FP64 emulation (`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, strategy EAGER). It uses its own handle with a 2 GiB workspace (4 GiB at 16384³). Mantissa control is either DYNAMIC (`dyn`: cuBLAS picks the slices from the data) or FIXED with a maximum of B mantissa bits (`fix47/55/63`) |
| `int8` | cuBLASLt int8 TN GEMM with int32 out, on the GEMM view with k padded to 16. This is the time of **one** modulus' GEMM; ×N is the GEMM-phase roofline at that N |
| `gemmul8 N=…` | GEMMul8 6a9eff8 (v3.5.2), INT8 backend, fast mode, FP64. It has no batched API, so batched cases make one call per batch |
| `cutensor` | cuTENSOR 2.8 FP64 contraction (`CUTENSOR_COMPUTE_DESC_64F`) on the original layouts |
| `imd N=…` | immerdink `imd_plan_create` once, then `imd_contract` timed. Settings: RN, g = N, β = 0, workspace from `imd_workspace_size`. Then `--phase-reps` (5) more calls with the library's per-phase event marks on (`phases` in the record; see "Per-phase timing") |
| `perm+X` | for a matrix-only method X on a case that isn't already a GEMM view: cuTENSOR permutes A and B in, X runs, and C is permuted out, all timed together. Plain `X` on those cases times the GEMM on pre-permuted operands |
| `bw` | D2D `cudaMemcpy` of A: achievable DRAM bandwidth |

**Data**: `(u − ½)·exp(φ·g)` with `u ~ U[0,1)`, `g ~ N(0,1)` and φ = 0.5. This is
GEMMul8's test distribution. It is generated on the device from a counter-based hash, so
it is reproducible.

**Accuracy**: 256 sampled outputs, or every output when C is small. The reference is a
Dot2 (double-double) dot product computed **on the device** and kept as hi + lo, so errors
are measured against an essentially exact value. A check of the reference: immerdink at
N = 20 shows a max relative error ≤ 2⁻⁵³ (correct rounding) against it. The table reports:
- **max normwise error**: `|c − r| / (‖a_i‖₂‖b_j‖₂)`, the DESIGN §3.8 metric. This is the
  one to compare at;
- **max relative error**: `|c − r| / |r|`. It is dominated by cancellation in
  near-zero outputs.

**Memory**: the host never holds an operand. Inputs are generated on the device, the
reference runs on the device, and only the samples are copied back. Each method's
buffers and workspaces are freed before the next method runs. Peak cgroup memory per
process in this sweep was **≤ 1.0 GiB** for every case, including 16384³ (662 MiB; peak
host RSS 1.03 GiB, all of it CUDA/cuBLAS/cuTENSOR/GEMMul8 library images). Every case
stays far below the 8 GB per-process target. Device memory is the limit at 16384³, so
that case runs only N ∈ {8, 14}.

## How to run

```sh
tools/get_gemmul8.sh        # once: clone GEMMul8 @ 6a9eff8 into bench/_deps/ (git-ignored), build (-j ≤ 4, memcap)
bench/run_sweep.sh          # build bench + libimmerdink (bench/build.sh), then every case of bench/sweep_m4.txt
python3 bench/summarize.py bench/results/<run>.jsonl [--before <older run>.jsonl]   # the tables below
```

- `CASES='sq8192|t_cc' bench/run_sweep.sh` runs a subset.
- A single case: `tools/run_bench.sh build/bench/imd_bench --name X --spec 'ab,bc->ac'
  --ext a=..,b=..,c=.. --methods ... --N ...` (see `imd_bench` with no arguments).
- `bench/` is a standalone CMake project. It adds the repository as a subdirectory
  with tests off, so the top-level `CMakeLists.txt` is untouched.
- `IMMERDINK_LIBRARY=/path/libimmerdink.so` links a prebuilt library instead.
- `IMMERDINK_CUTLASS_DIR` points at a CUTLASS checkout, if the submodule isn't
  initialized in this worktree.

Every case is one process run through `tools/run_bench.sh`. That script takes the
machine-wide lock, waits (`QUIET_WAIT`) for `tools/quiet_check.sh`, records the
conditions in `QUIET_LOG`, and runs the case inside `tools/memcap.sh`. If the machine
never gets quiet, the sweep stops and writes a `not run` record. It never bypasses the
gate. The sweep ignores the Claude desktop GPU *renderer* process for the GPU-process
check, as it does Xorg and gnome-shell. GPU utilization is still gated.

Builds (`bench/build.sh`, `tools/get_gemmul8.sh`) run under `nice`, inside
`tools/memcap.sh`, with at most 4 jobs. They hold the same lock, so they never overlap a
timed run.

## M4 sweep (2026-10-03, after the M4 kernel work)

Same machine, driver, libraries, cases and settings as the baseline below. Commit
`b29b464`, env `immerdink`; every case passed the quiet gate. Raw data:
`bench/results/m4-after-b29b464.jsonl`. Accuracy columns are unchanged from the baseline
to the printed digit: the results are bit-identical (the M4 changes are kernel-internal
and the test suite checks bitwise equality with the oracle).

### All cases (N = 14 for the Ozaki-II methods)

| case | spec | m×n×k | method | median ms | spread % | TFLOPS-eq | max normwise err | max rel err |
|---|---|---|---|---|---|---|---|---|
| sq1024 | `ik,jk->ij` | 1024×1024×1024 | dgemm | 1.79 | 0.3 | 1.20 | 1.3e-16 | 2.0e-14 |
|  |  |  | emu dyn | 0.435 | 0.5 | 4.93 | 9.1e-18 | 1.6e-16 |
|  |  |  | emu fix55 | 0.2 | 1.5 | 10.75 | 1.3e-17 | 8.5e-15 |
|  |  |  | int8 | 0.0143 | 14.3 | 150 TOPS (×14: 0.201 ms) | — | — |
|  |  |  | gemmul8 N=14 | 0.183 | 1.7 | 11.75 | 1.3e-16 | 1.2e-13 |
|  |  |  | imd N=14 | 0.234 | 35.4 | 9.16 | 9.0e-17 | 3.2e-14 |
| sq2048 | `ik,jk->ij` | 2048×2048×2048 | dgemm | 15.1 | 4.5 | 1.14 | 1.7e-16 | 1.1e-13 |
|  |  |  | emu dyn | 2.04 | 0.8 | 8.44 | 6.4e-18 | 1.8e-16 |
|  |  |  | emu fix55 | 1.14 | 1.3 | 15.01 | 8.5e-18 | 2.0e-14 |
|  |  |  | int8 | 0.0532 | 11.5 | 323 TOPS (×14: 0.745 ms) | — | — |
|  |  |  | gemmul8 N=14 | 0.965 | 1.0 | 17.80 | 1.6e-16 | 4.5e-13 |
|  |  |  | imd N=14 | 0.971 | 6.3 | 17.70 | 7.8e-17 | 3.6e-13 |
| sq4096 | `ik,jk->ij` | 4096×4096×4096 | dgemm | 109 | 0.0 | 1.26 | 1.5e-16 | 1.1e-13 |
|  |  |  | emu dyn | 10.4 | 0.4 | 13.19 | 4.9e-18 | 2.0e-16 |
|  |  |  | emu fix55 | 6.65 | 1.7 | 20.66 | 7.7e-18 | 1.8e-14 |
|  |  |  | int8 | 0.253 | 2.8 | 542 TOPS (×14: 3.55 ms) | — | — |
|  |  |  | gemmul8 N=14 | 5.94 | 1.2 | 23.13 | 1.5e-16 | 6.0e-14 |
|  |  |  | imd N=14 | 5 | 2.1 | 27.46 | 7.1e-17 | 2.2e-13 |
| sq8192 | `ik,jk->ij` | 8192×8192×8192 | bw (D2D copy) | 1.17 | 0.4 | 917 GB/s | | |
|  |  |  | dgemm | 871 | 0.0 | 1.26 | 2.0e-16 | 2.0e-13 |
|  |  |  | emu dyn | 63.9 | 5.0 | 17.22 | 3.5e-18 | 1.8e-16 |
|  |  |  | emu fix55 | 45 | 6.2 | 24.43 | 4.8e-18 | 1.2e-14 |
|  |  |  | int8 | 1.78 | 0.7 | 616 TOPS (×14: 25 ms) | — | — |
|  |  |  | gemmul8 N=14 | 35.3 | 0.3 | 31.15 | 1.5e-16 | 3.4e-13 |
|  |  |  | imd N=14 | 31.4 | 1.5 | 35.05 | 5.6e-17 | 1.5e-13 |
| sq16384 | `ik,jk->ij` | 16384×16384×16384 | dgemm | 6968 | 0.0 | 1.26 | 1.7e-16 | 2.0e-13 |
|  |  |  | emu dyn | 479 | 4.5 | 18.35 | 3.4e-18 | 1.9e-16 |
|  |  |  | emu fix55 | 318 | 4.0 | 27.68 | 3.6e-18 | 4.2e-14 |
|  |  |  | int8 | 13.6 | 0.3 | 646 TOPS (×14: 191 ms) | — | — |
|  |  |  | gemmul8 N=14 | 242 | 2.5 | 36.34 | 1.3e-16 | 4.9e-13 |
|  |  |  | imd N=14 | 234 | 10.6 | 37.58 | 6.7e-17 | 2.7e-13 |
| r_lowk | `ik,jk->ij` | 8192×8192×1024 | dgemm | 109 | 0.3 | 1.26 | 1.6e-16 | 6.1e-14 |
|  |  |  | emu dyn | 12.1 | 3.6 | 11.32 | 1.0e-17 | 1.9e-16 |
|  |  |  | int8 | 0.407 | 3.0 | 338 TOPS (×14: 5.69 ms) | — | — |
|  |  |  | gemmul8 N=14 | 11.9 | 2.9 | 11.55 | 1.5e-16 | 1.7e-13 |
|  |  |  | imd N=14 | 6.94 | 3.3 | 19.82 | 8.3e-17 | 2.3e-13 |
| r_bigk | `ik,jk->ij` | 2048×2048×32768 | dgemm | 218 | 0.0 | 1.26 | 4.7e-17 | 1.1e-12 |
|  |  |  | emu dyn | 31.4 | 2.4 | 8.76 | 1.6e-18 | 1.6e-16 |
|  |  |  | int8 | 0.711 | 1.0 | 387 TOPS (×14: 9.96 ms) | — | — |
|  |  |  | gemmul8 N=14 | 11 | 0.5 | 24.94 | 1.3e-16 | 1.2e-12 |
|  |  |  | imd N=14 | 11.6 | 2.6 | 23.65 | 7.3e-17 | 4.8e-13 |
| r_skinny | `ik,jk->ij` | 16384×512×4096 | dgemm | 54.5 | 0.0 | 1.26 | 1.6e-16 | 4.8e-13 |
|  |  |  | emu dyn | 12.9 | 0.3 | 5.31 | 5.7e-18 | 1.7e-16 |
|  |  |  | int8 | 0.142 | 9.4 | 483 TOPS (×14: 1.99 ms) | — | — |
|  |  |  | gemmul8 N=14 | 4.51 | 3.2 | 15.25 | 2.1e-16 | 1.3e-12 |
|  |  |  | imd N=14 | 4.44 | 4.3 | 15.47 | 7.9e-17 | 3.2e-13 |
| r_mm | `ik,jk->ij` | 1000×64×50000 | dgemm | 5.97 | 0.1 | 1.07 | 5.0e-17 | 2.6e-13 |
|  |  |  | emu dyn | 11.4 | 0.7 | 0.56 | 1.3e-18 | 1.9e-16 |
|  |  |  | int8 | 0.173 | 3.0 | 37 TOPS (×14: 2.42 ms) | — | — |
|  |  |  | gemmul8 N=14 | 3.1 | 3.0 | 2.06 | 1.1e-16 | 7.1e-13 |
|  |  |  | imd N=14 | 3.19 | 2.9 | 2.01 | 4.5e-17 | 3.0e-13 |
| t_nn | `ik,kj->ij` | 4096×4096×4096 | dgemm | 109 | 0.0 | 1.26 | 1.5e-16 | 9.2e-14 |
|  |  |  | emu dyn | 10.4 | 0.2 | 13.18 | 4.4e-18 | 1.6e-16 |
|  |  |  | int8 | 0.251 | 2.4 | 548 TOPS (×14: 3.51 ms) | — | — |
|  |  |  | gemmul8 N=14 | 6.12 | 0.4 | 22.47 | 1.9e-16 | 3.3e-13 |
|  |  |  | cutensor | 109 | 0.0 | 1.26 | 1.5e-16 | 9.2e-14 |
|  |  |  | imd N=14 | 5.11 | 2.1 | 26.88 | 5.9e-17 | 4.9e-13 |
| t_batch | `lik,ljk->lij` | 1024×1024×1024 b=16 | dgemm | 28.4 | 10.6 | 1.21 | 2.0e-16 | 4.4e-14 |
|  |  |  | emu dyn | 6.86 | 0.6 | 5.01 | 9.2e-18 | 1.7e-16 |
|  |  |  | int8 | 0.124 | 7.4 | 277 TOPS (×14: 1.73 ms) | — | — |
|  |  |  | gemmul8 N=14 | 2.78 | 2.3 | 12.36 | 1.4e-16 | 7.9e-14 |
|  |  |  | cutensor | 27.4 | 0.5 | 1.25 | 2.0e-16 | 4.4e-14 |
|  |  |  | imd N=14 | 2.77 | 1.4 | 12.38 | 7.4e-17 | 4.0e-14 |
| t_ikl | `ikl,kjl->ijl` | 1024×1024×1024 b=16 | dgemm | 27.3 | 1.0 | 1.26 | 2.1e-16 | 2.8e-13 |
|  |  |  | perm+dgemm | 28.1 | 0.0 | 1.22 | 2.1e-16 | 2.8e-13 |
|  |  |  | emu dyn | 6.86 | 0.7 | 5.01 | 1.0e-17 | 1.7e-16 |
|  |  |  | perm+emu dyn | 7.76 | 0.5 | 4.43 | 1.0e-17 | 1.7e-16 |
|  |  |  | int8 | 0.123 | 12.4 | 278 TOPS (×14: 1.73 ms) | — | — |
|  |  |  | gemmul8 N=14 | 2.79 | 2.2 | 12.33 | 1.9e-16 | 4.8e-13 |
|  |  |  | perm+gemmul8 N=14 | 3.69 | 2.1 | 9.32 | 1.9e-16 | 4.8e-13 |
|  |  |  | cutensor | 28.3 | 0.2 | 1.22 | 2.1e-16 | 2.8e-13 |
|  |  |  | imd N=14 | 2.91 | 1.3 | 11.80 | 9.3e-17 | 5.3e-13 |
| t_df | `acQ,bdQ->acbd` | 4096×4096×2048 | dgemm | 54.5 | 11.0 | 1.26 | 1.3e-16 | 2.3e-13 |
|  |  |  | emu dyn | 5.62 | 0.3 | 12.24 | 7.1e-18 | 2.1e-16 |
|  |  |  | int8 | 0.145 | 7.7 | 473 TOPS (×14: 2.04 ms) | — | — |
|  |  |  | gemmul8 N=14 | 4.02 | 0.3 | 17.09 | 1.4e-16 | 2.5e-13 |
|  |  |  | cutensor | 54.7 | 0.1 | 1.26 | 1.3e-16 | 2.3e-13 |
|  |  |  | imd N=14 | 2.96 | 3.2 | 23.25 | 7.6e-17 | 7.9e-13 |
| t_cc | `kjbc,kiac->abij` | 4096×4096×4096 | dgemm | 109 | 4.8 | 1.26 | 1.2e-16 | 4.0e-13 |
|  |  |  | perm+dgemm | 110 | 0.3 | 1.25 | 1.2e-16 | 4.0e-13 |
|  |  |  | emu dyn | 10.4 | 0.1 | 13.17 | 4.4e-18 | 1.8e-16 |
|  |  |  | perm+emu dyn | 11.3 | 0.2 | 12.13 | 4.4e-18 | 1.8e-16 |
|  |  |  | int8 | 0.255 | 4.0 | 539 TOPS (×14: 3.57 ms) | — | — |
|  |  |  | gemmul8 N=14 | 5.94 | 0.5 | 23.12 | 1.4e-16 | 4.2e-13 |
|  |  |  | perm+gemmul8 N=14 | 6.86 | 0.5 | 20.05 | 1.4e-16 | 4.2e-13 |
|  |  |  | cutensor | 110 | 0.0 | 1.25 | 1.0e-16 | 2.9e-14 |
|  |  |  | imd N=14 | 5.01 | 2.0 | 27.45 | 7.0e-17 | 2.1e-13 |
| t_permout | `abk,kcd->cadb` | 4096×4096×4096 | dgemm | 109 | 0.0 | 1.26 | 1.1e-16 | 1.0e-13 |
|  |  |  | perm+dgemm | 109 | 0.0 | 1.26 | 1.1e-16 | 1.0e-13 |
|  |  |  | emu dyn | 10.4 | 0.6 | 13.19 | 5.7e-18 | 2.0e-16 |
|  |  |  | perm+emu dyn | 10.8 | 0.2 | 12.76 | 5.7e-18 | 2.0e-16 |
|  |  |  | int8 | 0.249 | 2.5 | 552 TOPS (×14: 3.48 ms) | — | — |
|  |  |  | gemmul8 N=14 | 6.12 | 0.5 | 22.45 | 1.7e-16 | 1.8e-13 |
|  |  |  | perm+gemmul8 N=14 | 6.42 | 0.3 | 21.42 | 1.7e-16 | 1.8e-13 |
|  |  |  | cutensor | 110 | 0.0 | 1.25 | 1.1e-16 | 1.0e-13 |
|  |  |  | imd N=14 | 5.12 | 0.6 | 26.86 | 6.1e-17 | 6.2e-14 |

### N sweep

| case | method | N | median ms | TFLOPS-eq | max normwise err |
|---|---|---|---|---|---|
| sq4096 | dgemm | — | 109 | 1.26 | 1.5e-16 |
| sq4096 | emu | dyn | 10.4 | 13.19 | 4.9e-18 |
| sq4096 | emu | fix47 | 6.09 | 22.56 | 1.7e-15 |
| sq4096 | emu | fix55 | 6.65 | 20.66 | 7.7e-18 |
| sq4096 | emu | fix63 | 8.03 | 17.13 | 4.9e-18 |
| sq4096 | gemmul8 | 8 | 3.46 | 39.69 | 1.7e-09 |
| sq4096 | gemmul8 | 12 | 5.06 | 27.18 | 2.7e-14 |
| sq4096 | gemmul8 | 14 | 5.94 | 23.13 | 1.5e-16 |
| sq4096 | gemmul8 | 16 | 6.68 | 20.57 | 3.2e-18 |
| sq4096 | gemmul8 | 20 | 8.27 | 16.61 | 3.2e-18 |
| sq4096 | imd | 8 | 3.38 | 40.69 | 8.1e-10 |
| sq4096 | imd | 12 | 4.41 | 31.15 | 1.1e-14 |
| sq4096 | imd | 14 | 5 | 27.46 | 7.1e-17 |
| sq4096 | imd | 16 | 5.87 | 23.43 | 3.2e-18 |
| sq4096 | imd | 20 | 7.28 | 18.87 | 3.2e-18 |
| sq8192 | dgemm | — | 871 | 1.26 | 2.0e-16 |
| sq8192 | emu | dyn | 63.9 | 17.22 | 3.5e-18 |
| sq8192 | emu | fix47 | 40.7 | 27.04 | 1.2e-15 |
| sq8192 | emu | fix55 | 45 | 24.43 | 4.8e-18 |
| sq8192 | emu | fix63 | 52.5 | 20.95 | 3.5e-18 |
| sq8192 | gemmul8 | 8 | 20.7 | 53.08 | 1.1e-09 |
| sq8192 | gemmul8 | 12 | 30.2 | 36.35 | 4.0e-14 |
| sq8192 | gemmul8 | 14 | 35.3 | 31.15 | 1.5e-16 |
| sq8192 | gemmul8 | 16 | 40 | 27.46 | 1.6e-18 |
| sq8192 | gemmul8 | 20 | 49.3 | 22.29 | 1.6e-18 |
| sq8192 | imd | 8 | 20.2 | 54.36 | 5.5e-10 |
| sq8192 | imd | 12 | 28.1 | 39.11 | 1.7e-14 |
| sq8192 | imd | 14 | 31.4 | 35.05 | 5.6e-17 |
| sq8192 | imd | 16 | 39.1 | 28.15 | 1.6e-18 |
| sq8192 | imd | 20 | 46.5 | 23.66 | 1.6e-18 |

### immerdink phase lower bounds (MEMORY_MODEL §4, N = 14, ms)

| case | N | K1 scale | K2/K3 pack | K4 GEMM | K5 CRT | Σ lower bound | N × cuBLASLt int8 | measured imd | Σ LB / measured |
|---|---|---|---|---|---|---|---|---|---|
| sq1024 | 14 | 0.0166 | 0.0458 | 0.0455 | 0.0229 | 0.131 | 0.201 | 0.234 | 56 % |
| sq2048 | 14 | 0.0666 | 0.183 | 0.364 | 0.0915 | 0.705 | 0.745 | 0.971 | 73 % |
| sq4096 | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.55 | 5 | 85 % |
| sq8192 | 14 | 1.07 | 2.93 | 23.3 | 1.46 | 28.8 | 25 | 31.4 | 92 % |
| sq16384 | 14 | 4.26 | 11.7 | 186 | 5.86 | 208 | 191 | 234 | 89 % |
| r_lowk | 14 | 0.133 | 0.366 | 2.91 | 1.46 | 4.88 | 5.69 | 6.94 | 70 % |
| r_bigk | 14 | 1.07 | 2.93 | 5.83 | 0.0915 | 9.91 | 9.96 | 11.6 | 85 % |
| r_skinny | 14 | 0.549 | 1.51 | 1.46 | 0.183 | 3.7 | 1.99 | 4.44 | 83 % |
| r_mm | 14 | 0.422 | 1.16 | 0.74 | 0.0014 | 2.32 | 2.42 | 3.19 | 73 % |
| t_nn | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.51 | 5.11 | 84 % |
| t_batch | 14 | 0.266 | 0.732 | 0.728 | 0.366 | 2.09 | 1.73 | 2.77 | 75 % |
| t_ikl | 14 | 0.266 | 0.732 | 0.728 | 0.366 | 2.09 | 1.73 | 2.91 | 72 % |
| t_df | 14 | 0.133 | 0.366 | 1.46 | 0.366 | 2.32 | 2.04 | 2.96 | 79 % |
| t_cc | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.57 | 5.01 | 85 % |
| t_permout | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.48 | 5.12 | 84 % |

### immerdink measured phases (event marks, N = 14): ms (§4 lower bound as % of it)

| case | K0 vec | K1 scale | K2/K3 pack | K4 GEMM | K5 CRT | Σ phases | Σ LB (excl. K0) / Σ |
|---|---|---|---|---|---|---|---|
| sq1024 | 0.0515 (3 %) | 0.0448 (37 %) | 0.0512 (89 %) | 0.0829 (55 %) | 0.041 (56 %) | 0.268 | 49 % |
| sq2048 | 0.0947 (3 %) | 0.131 (51 %) | 0.207 (89 %) | 0.46 (79 %) | 0.128 (72 %) | 1.03 | 69 % |
| sq4096 | 0.101 (6 %) | 0.362 (73 %) | 0.836 (88 %) | 3.25 (90 %) | 0.474 (77 %) | 5.03 | 85 % |
| sq8192 | 0.161 (8 %) | 1.31 (81 %) | 3.39 (86 %) | 25.9 (90 %) | 1.96 (75 %) | 32.8 | 88 % |
| sq16384 | 0.409 (6 %) | 6.38 (67 %) | 13.8 (85 %) | 213 (87 %) | 8.09 (72 %) | 242 | 86 % |
| r_lowk | 0.127 (7 %) | 0.199 (67 %) | 0.424 (86 %) | 4.41 (66 %) | 1.86 (79 %) | 7.01 | 70 % |
| r_bigk | 0.261 (7 %) | 1.93 (55 %) | 3.37 (87 %) | 5.99 (97 %) | 0.127 (72 %) | 11.7 | 85 % |
| r_skinny | 0.136 (8 %) | 0.657 (84 %) | 1.69 (89 %) | 1.71 (85 %) | 0.242 (76 %) | 4.45 | 83 % |
| r_mm | 0.347 (7 %) | 0.661 (64 %) | 1.36 (86 %) | 0.914 (81 %) | 0.0113 (12 %) | 3.29 | 71 % |
| t_nn | 0.101 (6 %) | 0.488 (55 %) | 0.831 (88 %) | 3.24 (90 %) | 0.47 (78 %) | 5.13 | 83 % |
| t_batch | 0.0281 (6 %) | 0.348 (77 %) | 0.831 (88 %) | 1.12 (65 %) | 0.466 (79 %) | 2.8 | 75 % |
| t_ikl | 0.0271 (6 %) | 0.446 (60 %) | 0.856 (86 %) | 1.12 (65 %) | 0.475 (77 %) | 2.93 | 72 % |
| t_df | 0.0905 (6 %) | 0.204 (65 %) | 0.42 (87 %) | 1.79 (81 %) | 0.469 (78 %) | 2.97 | 78 % |
| t_cc | 0.101 (6 %) | 0.366 (73 %) | 0.834 (88 %) | 3.25 (90 %) | 0.469 (78 %) | 5.02 | 85 % |
| t_permout | 0.0734 (8 %) | 0.494 (54 %) | 0.836 (88 %) | 3.24 (90 %) | 0.47 (78 %) | 5.13 | 83 % |

### Before / after (N = 14; before: bench/results/m4-baseline-e4d8845.jsonl)

| case | m×n×k | imd before ms | imd after ms | speedup | TFLOPS-eq after | GEMMul8 ms (before / after run) | imd / GEMMul8 |
|---|---|---|---|---|---|---|---|
| sq1024 | 1024×1024×1024 | 0.266 | 0.234 | 1.14× | 9.16 | 0.175 / 0.183 | 1.28 |
| sq2048 | 2048×2048×2048 | 1.23 | 0.971 | 1.26× | 17.70 | 0.965 / 0.965 | 1.01 |
| sq4096 | 4096×4096×4096 | 6.38 | 5 | 1.27× | 27.46 | 5.98 / 5.94 | 0.84 |
| sq8192 | 8192×8192×8192 | 39.9 | 31.4 | 1.27× | 35.05 | 36.9 / 35.3 | 0.89 |
| sq16384 | 16384×16384×16384 | 380 | 234 | 1.63× | 37.58 | 259 / 242 | 0.97 |
| r_lowk | 8192×8192×1024 | 8.87 | 6.94 | 1.28× | 19.82 | 11.9 / 11.9 | 0.58 |
| r_bigk | 2048×2048×32768 | 16.2 | 11.6 | 1.39× | 23.65 | 11 / 11 | 1.05 |
| r_skinny | 16384×512×4096 | 6.92 | 4.44 | 1.56× | 15.47 | 4.5 / 4.51 | 0.99 |
| r_mm | 1000×64×50000 | 4.89 | 3.19 | 1.54× | 2.01 | 3.1 / 3.1 | 1.03 |
| t_nn | 4096×4096×4096 | 6.57 | 5.11 | 1.29× | 26.88 | 6.12 / 6.12 | 0.84 |
| t_batch | 1024×1024×1024 b=16 | 4.07 | 2.77 | 1.47× | 12.38 | 2.75 / 2.78 | 1.00 |
| t_ikl | 1024×1024×1024 b=16 | 10.9 | 2.91 | 3.76× | 11.80 | 2.76 / 2.79 | 1.04 |
| t_df | 4096×4096×2048 | 3.68 | 2.96 | 1.24× | 23.25 | 4.02 / 4.02 | 0.74 |
| t_cc | 4096×4096×4096 | 6.21 | 5.01 | 1.24× | 27.45 | 5.96 / 5.94 | 0.84 |
| t_permout | 4096×4096×4096 | 6.59 | 5.12 | 1.29× | 26.86 | 6.11 / 6.12 | 0.84 |

### M4 exit targets

| target | result |
|---|---|
| sq8192, N = 14 ≥ GEMMul8 on this box (29.8 TFLOPS-eq in the baseline run) | **35.05** TFLOPS-eq (31.4 ms); GEMMul8 31.15 in the same run |
| sq16384: close most of the gap to GEMMul8 (baseline 380 vs 259 ms) | **234 ms** (37.6 TFLOPS-eq); GEMMul8 242 ms in the same run |
| permuted contractions within 15 % of the matrix case at equal m, n, k | `t_cc` 27.45 and `t_permout` 26.86 vs `sq4096` 27.46 TFLOPS-eq (0 %, −2 %); `t_nn` 26.88 (−2 %); `t_ikl` 11.80 vs `t_batch` 12.38 (−4.7 %, was −63 %) |
| each phase as % of its MEMORY_MODEL §4 bound | the measured-phase table above; also in every `imd` record |

### Observations (M4)

- **Every case is faster**, 1.14–3.76×; 10 of 15 now beat GEMMul8 N = 14 (which is at
  about 2× immerdink's normwise error), the rest are within 1–5 % (GEMMul8 timed on pre-permuted operands) except `sq1024` (1.28×).
- **Large squares** are at 88–92 % of the summed §4 bound. K4 (the int8 GEMMs) runs at
  ≈ 600 TOPS, the cuBLASLt int8 rate; the pack is at 85–89 %, K5 at 72–78 %, K1 at
  67–81 %.
- **`t_ikl`** (batch label stride-1 in all three tensors) went from 10.9 to 2.91 ms, now
  within 5 % of the batch-outermost `t_batch` and faster than `perm+gemmul8` (3.69 ms).
- **Where the rest goes** (see NOTES-M4 "Remaining gaps"): K4 at small k (60–66 % at
  k = 1024: the mod-p epilogue is not overlapped with the main loop), K1 on very long
  rows (55 % on r_bigk), and the fixed per-call costs that dominate `sq1024`: K0's
  small H2D copy (0.03–0.4 ms, slowed by the PCIe link's idle power state) and ten
  launches. NOTES-M4 has the design proposals for K0 and K1.
- The 4090 runs the large cases at its 450 W power cap (`clocks_event_reasons` 0x4), so
  run-to-run differences of a few % in K4 (and in GEMMul8: 259 → 242 ms at 16384³ between
  the two sweeps) are noise.

## First baseline sweep (2026-10-03)

> The M1 library before any M4 optimization (commit `e4d8845` = dev `c1b0bba` + the
> bench). Its observations below describe that state.

RTX 4090 (128 SMs; max clocks 3120/10501 MHz; driver 595.91.07; CUDA runtime 13.4;
cuBLAS 13.8.0; cuTENSOR 2.8.0). Env `immerdink`, commit `e4d8845`. Every case passed
the quiet gate (conditions recorded per record). Total GPU time for the sweep was about
1.5 minutes. Raw data: `bench/results/m4-baseline-e4d8845.jsonl`.


### All cases (N = 14 for the Ozaki-II methods)

| case | spec | m×n×k | method | median ms | spread % | TFLOPS-eq | max normwise err | max rel err |
|---|---|---|---|---|---|---|---|---|
| sq1024 | `ik,jk->ij` | 1024×1024×1024 | dgemm | 1.9 | 0.6 | 1.13 | 1.3e-16 | 2.0e-14 |
|  |  |  | emu dyn | 0.455 | 0.7 | 4.72 | 9.1e-18 | 1.6e-16 |
|  |  |  | emu fix55 | 0.217 | 4.7 | 9.89 | 1.3e-17 | 8.5e-15 |
|  |  |  | int8 | 0.0113 | 18.2 | 191 TOPS (×14: 0.158 ms) | — | — |
|  |  |  | gemmul8 N=14 | 0.175 | 1.7 | 12.26 | 1.3e-16 | 1.2e-13 |
|  |  |  | imd N=14 | 0.266 | 1.2 | 8.07 | 9.0e-17 | 3.2e-14 |
| sq2048 | `ik,jk->ij` | 2048×2048×2048 | dgemm | 15.6 | 12.6 | 1.10 | 1.7e-16 | 1.1e-13 |
|  |  |  | emu dyn | 2.04 | 1.7 | 8.43 | 6.4e-18 | 1.8e-16 |
|  |  |  | emu fix55 | 1.15 | 3.3 | 14.98 | 8.5e-18 | 2.0e-14 |
|  |  |  | int8 | 0.0553 | 4.2 | 310 TOPS (×14: 0.775 ms) | — | — |
|  |  |  | gemmul8 N=14 | 0.965 | 3.1 | 17.80 | 1.6e-16 | 4.5e-13 |
|  |  |  | imd N=14 | 1.23 | 0.8 | 14.01 | 7.8e-17 | 3.6e-13 |
| sq4096 | `ik,jk->ij` | 4096×4096×4096 | dgemm | 113 | 1.3 | 1.22 | 1.5e-16 | 1.1e-13 |
|  |  |  | emu dyn | 10.7 | 5.6 | 12.88 | 4.9e-18 | 2.0e-16 |
|  |  |  | emu fix55 | 6.81 | 9.0 | 20.18 | 7.7e-18 | 1.8e-14 |
|  |  |  | int8 | 0.252 | 3.7 | 545 TOPS (×14: 3.53 ms) | — | — |
|  |  |  | gemmul8 N=14 | 5.98 | 9.5 | 22.99 | 1.5e-16 | 6.0e-14 |
|  |  |  | imd N=14 | 6.38 | 13.2 | 21.55 | 7.1e-17 | 2.2e-13 |
| sq8192 | `ik,jk->ij` | 8192×8192×8192 | bw (D2D copy) | 1.18 | 4.1 | 912 GB/s | | |
|  |  |  | dgemm | 896 | 0.5 | 1.23 | 2.0e-16 | 2.0e-13 |
|  |  |  | emu dyn | 65.9 | 1.9 | 16.68 | 3.5e-18 | 1.8e-16 |
|  |  |  | emu fix55 | 46.4 | 1.6 | 23.71 | 4.8e-18 | 1.2e-14 |
|  |  |  | int8 | 1.78 | 1.8 | 617 TOPS (×14: 25 ms) | — | — |
|  |  |  | gemmul8 N=14 | 36.9 | 2.2 | 29.82 | 1.5e-16 | 3.4e-13 |
|  |  |  | imd N=14 | 39.9 | 3.2 | 27.57 | 5.6e-17 | 1.5e-13 |
| sq16384 | `ik,jk->ij` | 16384×16384×16384 | dgemm | 7449 | 0.5 | 1.18 | 1.7e-16 | 2.0e-13 |
|  |  |  | emu dyn | 518 | 0.3 | 16.99 | 3.4e-18 | 1.9e-16 |
|  |  |  | emu fix55 | 347 | 1.0 | 25.37 | 3.6e-18 | 4.2e-14 |
|  |  |  | int8 | 14.8 | 2.6 | 593 TOPS (×14: 208 ms) | — | — |
|  |  |  | gemmul8 N=14 | 259 | 6.2 | 33.99 | 1.3e-16 | 4.9e-13 |
|  |  |  | imd N=14 | 380 | 0.6 | 23.12 | 6.7e-17 | 2.7e-13 |
| r_lowk | `ik,jk->ij` | 8192×8192×1024 | dgemm | 109 | 0.0 | 1.26 | 1.6e-16 | 6.1e-14 |
|  |  |  | emu dyn | 11.8 | 0.4 | 11.70 | 1.0e-17 | 1.9e-16 |
|  |  |  | int8 | 0.404 | 3.0 | 340 TOPS (×14: 5.66 ms) | — | — |
|  |  |  | gemmul8 N=14 | 11.9 | 3.2 | 11.55 | 1.5e-16 | 1.7e-13 |
|  |  |  | imd N=14 | 8.87 | 3.4 | 15.49 | 8.3e-17 | 2.3e-13 |
| r_bigk | `ik,jk->ij` | 2048×2048×32768 | dgemm | 218 | 0.0 | 1.26 | 4.7e-17 | 1.1e-12 |
|  |  |  | emu dyn | 30.8 | 0.8 | 8.91 | 1.6e-18 | 1.6e-16 |
|  |  |  | int8 | 0.694 | 0.9 | 396 TOPS (×14: 9.71 ms) | — | — |
|  |  |  | gemmul8 N=14 | 11 | 0.4 | 24.97 | 1.3e-16 | 1.2e-12 |
|  |  |  | imd N=14 | 16.2 | 1.3 | 16.95 | 7.3e-17 | 4.8e-13 |
| r_skinny | `ik,jk->ij` | 16384×512×4096 | dgemm | 54.5 | 11.0 | 1.26 | 1.6e-16 | 4.8e-13 |
|  |  |  | emu dyn | 12.9 | 0.1 | 5.32 | 5.7e-18 | 1.7e-16 |
|  |  |  | int8 | 0.138 | 6.2 | 497 TOPS (×14: 1.94 ms) | — | — |
|  |  |  | gemmul8 N=14 | 4.5 | 3.3 | 15.26 | 2.1e-16 | 1.3e-12 |
|  |  |  | imd N=14 | 6.92 | 2.6 | 9.94 | 7.9e-17 | 3.2e-13 |
| r_mm | `ik,jk->ij` | 1000×64×50000 | dgemm | 6.63 | 0.0 | 0.97 | 5.0e-17 | 2.6e-13 |
|  |  |  | emu dyn | 11.4 | 7.6 | 0.56 | 1.3e-18 | 1.9e-16 |
|  |  |  | int8 | 0.172 | 8.4 | 37 TOPS (×14: 2.41 ms) | — | — |
|  |  |  | gemmul8 N=14 | 3.1 | 3.0 | 2.07 | 1.1e-16 | 7.1e-13 |
|  |  |  | imd N=14 | 4.89 | 0.2 | 1.31 | 4.5e-17 | 3.0e-13 |
| t_nn | `ik,kj->ij` | 4096×4096×4096 | dgemm | 109 | 0.0 | 1.26 | 1.5e-16 | 9.2e-14 |
|  |  |  | emu dyn | 10.4 | 0.2 | 13.18 | 4.4e-18 | 1.6e-16 |
|  |  |  | int8 | 0.252 | 14.2 | 545 TOPS (×14: 3.53 ms) | — | — |
|  |  |  | gemmul8 N=14 | 6.12 | 0.4 | 22.47 | 1.9e-16 | 3.3e-13 |
|  |  |  | cutensor | 109 | 0.2 | 1.26 | 1.5e-16 | 9.2e-14 |
|  |  |  | imd N=14 | 6.57 | 3.9 | 20.91 | 5.9e-17 | 4.9e-13 |
| t_batch | `lik,ljk->lij` | 1024×1024×1024 b=16 | dgemm | 27.5 | 9.7 | 1.25 | 2.0e-16 | 4.4e-14 |
|  |  |  | emu dyn | 6.83 | 0.3 | 5.03 | 9.2e-18 | 1.7e-16 |
|  |  |  | int8 | 0.121 | 4.2 | 283 TOPS (×14: 1.7 ms) | — | — |
|  |  |  | gemmul8 N=14 | 2.75 | 0.7 | 12.48 | 1.4e-16 | 7.9e-14 |
|  |  |  | cutensor | 27.3 | 0.1 | 1.26 | 2.0e-16 | 4.4e-14 |
|  |  |  | imd N=14 | 4.07 | 1.6 | 8.43 | 7.4e-17 | 4.0e-14 |
| t_ikl | `ikl,kjl->ijl` | 1024×1024×1024 b=16 | dgemm | 30.3 | 9.9 | 1.14 | 2.1e-16 | 2.8e-13 |
|  |  |  | perm+dgemm | 28.1 | 0.1 | 1.22 | 2.1e-16 | 2.8e-13 |
|  |  |  | emu dyn | 6.82 | 0.3 | 5.04 | 1.0e-17 | 1.7e-16 |
|  |  |  | perm+emu dyn | 7.7 | 0.1 | 4.46 | 1.0e-17 | 1.7e-16 |
|  |  |  | int8 | 0.121 | 6.8 | 283 TOPS (×14: 1.7 ms) | — | — |
|  |  |  | gemmul8 N=14 | 2.76 | 0.4 | 12.46 | 1.9e-16 | 4.8e-13 |
|  |  |  | perm+gemmul8 N=14 | 3.65 | 0.3 | 9.41 | 1.9e-16 | 4.8e-13 |
|  |  |  | cutensor | 28.2 | 0.6 | 1.22 | 2.1e-16 | 2.8e-13 |
|  |  |  | imd N=14 | 10.9 | 0.4 | 3.14 | 9.3e-17 | 5.3e-13 |
| t_df | `acQ,bdQ->acbd` | 4096×4096×2048 | dgemm | 54.5 | 0.0 | 1.26 | 1.3e-16 | 2.3e-13 |
|  |  |  | emu dyn | 5.62 | 0.2 | 12.24 | 7.1e-18 | 2.1e-16 |
|  |  |  | int8 | 0.144 | 4.9 | 476 TOPS (×14: 2.02 ms) | — | — |
|  |  |  | gemmul8 N=14 | 4.02 | 0.7 | 17.11 | 1.4e-16 | 2.5e-13 |
|  |  |  | cutensor | 54.8 | 0.1 | 1.25 | 1.3e-16 | 2.3e-13 |
|  |  |  | imd N=14 | 3.68 | 0.9 | 18.68 | 7.6e-17 | 7.9e-13 |
| t_cc | `kjbc,kiac->abij` | 4096×4096×4096 | dgemm | 109 | 0.2 | 1.26 | 1.2e-16 | 4.0e-13 |
|  |  |  | perm+dgemm | 110 | 0.0 | 1.25 | 1.2e-16 | 4.0e-13 |
|  |  |  | emu dyn | 10.5 | 0.5 | 13.15 | 4.4e-18 | 1.8e-16 |
|  |  |  | perm+emu dyn | 11.3 | 0.1 | 12.12 | 4.4e-18 | 1.8e-16 |
|  |  |  | int8 | 0.25 | 2.8 | 549 TOPS (×14: 3.51 ms) | — | — |
|  |  |  | gemmul8 N=14 | 5.96 | 0.4 | 23.08 | 1.4e-16 | 4.2e-13 |
|  |  |  | perm+gemmul8 N=14 | 6.86 | 0.3 | 20.04 | 1.4e-16 | 4.2e-13 |
|  |  |  | cutensor | 110 | 0.1 | 1.25 | 1.0e-16 | 2.9e-14 |
|  |  |  | imd N=14 | 6.21 | 3.5 | 22.13 | 7.0e-17 | 2.1e-13 |
| t_permout | `abk,kcd->cadb` | 4096×4096×4096 | dgemm | 109 | 0.1 | 1.26 | 1.1e-16 | 1.0e-13 |
|  |  |  | perm+dgemm | 109 | 0.0 | 1.26 | 1.1e-16 | 1.0e-13 |
|  |  |  | emu dyn | 10.4 | 0.2 | 13.16 | 5.7e-18 | 2.0e-16 |
|  |  |  | perm+emu dyn | 10.8 | 0.5 | 12.75 | 5.7e-18 | 2.0e-16 |
|  |  |  | int8 | 0.25 | 4.6 | 550 TOPS (×14: 3.5 ms) | — | — |
|  |  |  | gemmul8 N=14 | 6.11 | 0.4 | 22.48 | 1.7e-16 | 1.8e-13 |
|  |  |  | perm+gemmul8 N=14 | 6.42 | 0.2 | 21.41 | 1.7e-16 | 1.8e-13 |
|  |  |  | cutensor | 110 | 0.2 | 1.25 | 1.1e-16 | 1.0e-13 |
|  |  |  | imd N=14 | 6.59 | 2.8 | 20.86 | 6.1e-17 | 6.2e-14 |

### N sweep

| case | method | N | median ms | TFLOPS-eq | max normwise err |
|---|---|---|---|---|---|
| sq4096 | dgemm | — | 113 | 1.22 | 1.5e-16 |
| sq4096 | emu | dyn | 10.7 | 12.88 | 4.9e-18 |
| sq4096 | emu | fix47 | 6.21 | 22.12 | 1.7e-15 |
| sq4096 | emu | fix55 | 6.81 | 20.18 | 7.7e-18 |
| sq4096 | emu | fix63 | 8.18 | 16.80 | 4.9e-18 |
| sq4096 | gemmul8 | 8 | 3.49 | 39.44 | 1.7e-09 |
| sq4096 | gemmul8 | 12 | 5.09 | 27.00 | 2.7e-14 |
| sq4096 | gemmul8 | 14 | 5.98 | 22.99 | 1.5e-16 |
| sq4096 | gemmul8 | 16 | 6.71 | 20.48 | 3.2e-18 |
| sq4096 | gemmul8 | 20 | 8.44 | 16.28 | 3.2e-18 |
| sq4096 | imd | 8 | 3.41 | 40.33 | 8.1e-10 |
| sq4096 | imd | 12 | 5.52 | 24.91 | 1.1e-14 |
| sq4096 | imd | 14 | 6.38 | 21.55 | 7.1e-17 |
| sq4096 | imd | 16 | 7.13 | 19.29 | 3.2e-18 |
| sq4096 | imd | 20 | 9.14 | 15.04 | 3.2e-18 |
| sq8192 | dgemm | — | 896 | 1.23 | 2.0e-16 |
| sq8192 | emu | dyn | 65.9 | 16.68 | 3.5e-18 |
| sq8192 | emu | fix47 | 41.6 | 26.46 | 1.2e-15 |
| sq8192 | emu | fix55 | 46.4 | 23.71 | 4.8e-18 |
| sq8192 | emu | fix63 | 53.8 | 20.42 | 3.5e-18 |
| sq8192 | gemmul8 | 8 | 21.7 | 50.77 | 1.1e-09 |
| sq8192 | gemmul8 | 12 | 31.6 | 34.81 | 4.0e-14 |
| sq8192 | gemmul8 | 14 | 36.9 | 29.82 | 1.5e-16 |
| sq8192 | gemmul8 | 16 | 41.8 | 26.28 | 1.6e-18 |
| sq8192 | gemmul8 | 20 | 51.5 | 21.34 | 1.6e-18 |
| sq8192 | imd | 8 | 21.1 | 52.01 | 5.5e-10 |
| sq8192 | imd | 12 | 33.9 | 32.40 | 1.7e-14 |
| sq8192 | imd | 14 | 39.9 | 27.57 | 5.6e-17 |
| sq8192 | imd | 16 | 44.9 | 24.46 | 1.6e-18 |
| sq8192 | imd | 20 | 55.8 | 19.72 | 1.6e-18 |

### immerdink phase lower bounds (MEMORY_MODEL §4, N = 14, ms)

| case | N | K1 scale | K2/K3 pack | K4 GEMM | K5 CRT | Σ lower bound | N × cuBLASLt int8 | measured imd | Σ LB / measured |
|---|---|---|---|---|---|---|---|---|---|
| sq1024 | 14 | 0.0166 | 0.0458 | 0.0455 | 0.0229 | 0.131 | 0.158 | 0.266 | 49 % |
| sq2048 | 14 | 0.0666 | 0.183 | 0.364 | 0.0915 | 0.705 | 0.775 | 1.23 | 58 % |
| sq4096 | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.53 | 6.38 | 67 % |
| sq8192 | 14 | 1.07 | 2.93 | 23.3 | 1.46 | 28.8 | 25 | 39.9 | 72 % |
| sq16384 | 14 | 4.26 | 11.7 | 186 | 5.86 | 208 | 208 | 380 | 55 % |
| r_lowk | 14 | 0.133 | 0.366 | 2.91 | 1.46 | 4.88 | 5.66 | 8.87 | 55 % |
| r_bigk | 14 | 1.07 | 2.93 | 5.83 | 0.0915 | 9.91 | 9.71 | 16.2 | 61 % |
| r_skinny | 14 | 0.549 | 1.51 | 1.46 | 0.183 | 3.7 | 1.94 | 6.92 | 53 % |
| r_mm | 14 | 0.422 | 1.16 | 0.74 | 0.0014 | 2.32 | 2.41 | 4.89 | 48 % |
| t_nn | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.53 | 6.57 | 65 % |
| t_batch | 14 | 0.266 | 0.732 | 0.728 | 0.366 | 2.09 | 1.7 | 4.07 | 51 % |
| t_ikl | 14 | 0.266 | 0.732 | 0.728 | 0.366 | 2.09 | 1.7 | 10.9 | 19 % |
| t_df | 14 | 0.133 | 0.366 | 1.46 | 0.366 | 2.32 | 2.02 | 3.68 | 63 % |
| t_cc | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.51 | 6.21 | 69 % |
| t_permout | 14 | 0.266 | 0.732 | 2.91 | 0.366 | 4.28 | 3.5 | 6.59 | 65 % |

The phase table lists the MEMORY_MODEL §4 per-kernel lower bounds. Each is
`max(bytes / 1008 GB/s, ops / 660.6 TOPS)` with g = N, β = 0 and ρ = 1, at datasheet
peaks. The measured `bw` is 912 GB/s and the best measured int8 rate is about 617 TOPS,
so the bounds are slightly optimistic. "N × cuBLASLt int8" is the GEMM phase at the
library's own int8 rate.

## Observations

- **Accuracy at N = 14.** immerdink's max normwise error is 4.5–9.3·10⁻¹⁷, below native
  DGEMM (1–2·10⁻¹⁶). GEMMul8 fast mode is at DGEMM level (1–2·10⁻¹⁶), so at N = 14
  immerdink is about 2× more accurate than GEMMul8, at about 7–10 % more time on squares. cuBLAS `emu dyn`/`fix55` is more
  accurate (~5·10⁻¹⁸), which corresponds to roughly imd N = 15–16.
- **Square matrices.** immerdink M1 is 7–10 % behind GEMMul8 at 4096³ and 8192³:
  - at 8192³ and N = 14: 27.6 vs 29.8 TFLOPS-eq;
  - both beat the cuBLAS emulation (`fix55`: 23.7);
  - native DGEMM runs at 1.23.

  At 16384³ the gap opens: 23.1 vs 34.0 (380 vs 259 ms), and immerdink is at 55 % of
  its lower bound. Something scales badly there, probably the K4 tile config or the CRT
  at this size; worth profiling first. The M4 exit target (≥ 12.85 TFLOPS-eq at 8192³,
  N = 14) is already met by M1 on the desktop 4090. The meaningful bar is GEMMul8's
  29.8 on this machine.
- **Permuted tensors.** imd on `kjbc,kiac->abij` (22.1 TFLOPS-eq) and
  `abk,kcd->cadb` (20.9) runs within 3–4 % of its own 4096³ matrix rate (21.6). It beats
  `perm+gemmul8` on `t_cc` (6.21 vs 6.86 ms). That is the TBLIS point, already visible at
  M1.
- **Weak spot: `ikl,kjl->ijl`** (batch label innermost, stride-1). immerdink takes
  10.9 ms there, against 4.07 ms for the same GEMMs with the batch outermost (`t_batch`)
  and 3.65 ms for `perm+gemmul8`. That is 19 % of its lower bound, so the
  gather-pack/CRT on a stride-1 batch mode is the first M4 target (PLAN M4 item 1).
- **Small or skinny shapes** (`sq1024`, `r_skinny`, `r_mm`, `t_batch`) sit at 48–53 %
  of the bound. Fixed per-call overhead (several kernel launches, the K0 H2D copy)
  and the non-GEMM kernels likely dominate there. GEMMul8 is 1.5× faster on `r_skinny` and `r_mm`.
- **`r_lowk` (k = 1024)**: immerdink is faster than GEMMul8 (8.87 vs 11.9 ms).
- **cuBLAS emulation** is weak at low k and small shapes. `emu dyn` on `r_mm` is slower
  than native DGEMM.
- **cuTENSOR FP64** equals cuBLAS DGEMM (≈ 1.25 TFLOPS) on every tensor case. On the
  4090 the native FP64 baseline is 1/64 rate either way.

## Per-phase timing

The library has opt-in phase marks (M4): built with `-DIMMERDINK_PROFILE=ON` (the bench
build does this; production builds compile them out), `imd_bench` makes `--phase-reps`
extra calls with marks on and records per phase the median time, its MEMORY_MODEL §4
lower bound and the bound as % of the time (`phases` in the JSON record; the tables
above come from `bench/summarize.py`). The phases are K0 vec, K1 scale A/B, K2 pack A,
K3 pack B, K4 gemm (every GEMM launch), K5 crt, DIRECT and K5 beta-only. Still open:
NVTX ranges for `nsys` (not available on this box), achieved bytes per phase exported by
the library, and skipping K0 (NOTES-M4 proposal 1).

## Not covered yet

- No **`emu fix` at equal accuracy** sweep on non-square cases (only `dyn` there).
- **GEMMul8 accurate mode** and the cuBLAS emulation's mantissa offset are not tried.
- **Clock locking**: clocks are recorded, not locked (governor `powersave`, turbo on).
  The spread column shows run-to-run noise. Most variants are under 5 %; a few small
  ones are not.

## Residue-group size g: speed vs workspace (2026-10-03, `bench/results/g-sweep-66af510.jsonl`)

N = 14, RN, vec_mode GENERATE, quiet-gated. Workspace ≈ (g/8)(|A|+|B|) + (N/8)|C|.
Each cell: TFLOPS-eq as % of g = N, workspace as % of g = N.

| case | g=7 | g=4 | g=2 | g=1 |
|---|---|---|---|---|
| sq1024 | 99% / 67% | 86% / 52% | 57% / 43% | 36% / 38% |
| sq2048 | 90% / 67% | 74% / 52% | 61% / 43% | 49% / 38% |
| sq8192 | 99% / 67% | 95% / 52% | 87% / 43% | 70% / 38% |
| r_lowk | 97% / 90% | 93% / 86% | 86% / 83% | 66% / 81% |
| r_bigk | 89% / 52% | 75% / 31% | 61% / 17% | 41% / 10% |
| r_skinny | 84% / 55% | 68% / 36% | 52% / 24% | 33% / 17% |
| t_ikl | 88% / 67% | 72% / 52% | 57% / 43% | 37% / 38% |
| t_df | 95% / 75% | 84% / 64% | 68% / 57% | 52% / 54% |
| t_cc | 94% / 67% | 83% / 52% | 69% / 43% | 50% / 38% |
| s_cc16 | 89% / 67% | 74% / 52% | 57% / 43% | 46% / 38% |

- The cost is almost entirely in the pack (K2/K3): each residue group re-reads the FP64
  operand, so pack traffic goes from (8 + N)·|A| bytes at g = N to about N/g·(8 + g)·|A|.
  Example: sq8192 pack A 2.28 ms (g = 14) → 10.5 ms (g = 1). On small shapes the GEMM also
  loses occupancy (sq1024 K4 0.09 → 0.33 ms) because each launch has fewer moduli.
- The floor is (N/8)|C| (the product planes), so shapes with large C relative to A+B
  (r_lowk) gain little from small g; shapes with large operands (r_bigk, r_skinny) gain most.
- g = 7 halves the residue memory for ≤ 12% (often ≤ 5%) slowdown; g ≤ 2 is a
  last resort.

## A2: K4 epilogue / main-loop overlap (2026-10-03, `bench/sweep_a2.txt`)

K4 is now `K4Gemm` (persistent, register-only mod-p epilogue). Before/after interleaved in one
session: `bench/results/a2-before-ab-c63117a.jsonl` vs `a2-after-3e419b2.jsonl` (the latter
with GEMMul8 and cuBLASLt int8). K4 at k = 1024 went from 60–64 % to 80–86 % of its bound,
whole calls 1.07–1.20× faster at moderate k; full tables, ncu analysis and what was tried:
[NOTES-A2.md](NOTES-A2.md).

## A1 / A4: small-problem overhead, K1 long rows, K5 (2026-10-03)

The fused launch KF (DESIGN v1.4 §4: K0 + K1 + group 1's packs for k̂ ≤ 512), launch-shape
changes for small grids, K5 without local memory and with coalesced stores, and the K1
long-row grid. Before = dev + A2 (`2961ea4`), after = `19335ff`, interleaved per case in one
session (gated, GPU lock): `bench/results/a1-ab2-{before,after}.jsonl` (small cases, 40
calls) and `a1-ab1-{before,after}.jsonl` (every compare case but sq16384). Small shapes at
N = 8 went from 0.34–0.52× GEMMul8 to 0.92–0.96× (s128–s512), s_df / s_perm from 0.7× to
1.27–1.36×; at N = 14 every small shape is at or above GEMMul8; large shapes unchanged
within noise (after/before 0.89–1.03). K5 at N = 8 is at 91–96 % of its bound (was
61–64 %), K1 on r_bigk at 67–69 % (was 55 %). nsys breakdowns, tables and the remaining
floor: [NOTES-A1.md](NOTES-A1.md).

## A3: shape-dependent K4 tiles, K1 long rows, KF for k̂ ≤ 1024 (2026-10-03)

Four K4 configurations (128×256, 128×128, 64×128, 64×64) chosen per plan by a fitted rule
(DESIGN v1.5 §4), K1's long-row grid kept within the L2 (r_mm), and the fused launch KF
extended to k̂ ≤ 1024. Before = dev `fa77ec0`, after = `d3c9a78`, interleaved per case by
`bench/run_ab.sh` with GEMMul8 run first in both processes (the GPU's boost clock settles per
process; same-process GEMMul8 ratios are the robust metric):
`bench/results/a3-{before-fa77ec0,after-d3c9a78}.jsonl`. At N = 8: s128 0.87 → 1.06×
GEMMul8, s256 0.82 → 1.00×, s512 0.85 → 1.11×, s_cc8 0.85 → 1.03×, sq1024 0.94 → 1.04×,
r_mm 0.81 → 0.96× (1.09× at N = 14), t_ikl / t_batch 8–10 % faster; r_skinny 0.94× and r_bigk
0.97× unchanged (K4 at its bound there); large shapes within noise. The optional `k4_tune`
plan option (`bench/sweep_a3_tune.txt`, `a3-tune-ae2ac92.jsonl`) costs 1–24 ms of plan
creation and up to 256 MiB of temporary device memory for no measurable gain over the
rule. Per-configuration K4 times on 84 cases (`a3-configs-{nsys,phases}.jsonl`), the rule
fit, ncu analysis and tables: [NOTES-A3.md](NOTES-A3.md).

## D4 part 1: fused K1 + pack for large operands (KL) (2026-10-03)

One launch per operand (KL, DESIGN v1.6 §4) replaces K1 and group 1's pack whenever KF does
not apply (k̂ > 1024): the operand is read from DRAM once instead of twice before its first
GEMM. Before = dev `a572d85`, after = `197cd0f`, all of `bench/sweep_compare.txt` interleaved
by `bench/run_ab.sh` with GEMMul8 first in both processes:
`bench/results/{d4-before-a572d85,compare-197cd0f}.jsonl`. N = 8 / 14 vs GEMMul8 (same
process): r_skinny 0.93 → 1.18× / 1.08 → 1.27×, r_bigk 0.98 → 1.09× / 1.01 → 1.08×, r_mm
0.97 → 1.02× / 1.07 → 1.10×, sq2048 1.03 → 1.22×, sq4096 1.16 → 1.32×, sq8192 1.12 → 1.20×,
t_cc 1.19 → 1.32×, t_df 1.35 → 1.52×, t_nn 1.19 → 1.30×, t_permout 1.19 → 1.33×; k̂ ≤ 1024
(KF, unchanged) within noise (identical nsys kernel times). ncu: the scale + pack phase reads
half the DRAM bytes at g = N and one operand read less at any g. Tables, the threads-per-CTA
measurement and the g analysis: [NOTES-D4.md](NOTES-D4.md).

## Clock state and warm-up (2026-10-03)

GPU Boost picks the SM clock from power (450 W cap), temperature and recent load, so separate
benchmark processes landed at about 2535–2820 MHz, swinging small-shape timings by ±10–30 %.
`bench/run_sweep.sh` and `bench/run_ab.sh` now run each case through `bench/warm_run.sh`: a
fixed 3 s dense warm-up (`BENCH_WARMUP_S`, 0 disables) inside the same GPU-lock hold, then the
case, then a `{"case", "clock_state"}` line (SM clock, power, temperature, clock-event reasons;
a `nvidia-smi -lgc` lock shows up as `applications_clocks_setting: Active`). Check: three
separate runs of s512 all at 2805 MHz, timings within ~5 %. For publishable small-shape numbers,
lock clocks (`sudo nvidia-smi -lgc <f>,<f>`, reset with `-rgc`); that is a manual, root step.

## D1: packed operands (2026-10-03, `bench/results/d1-packed-323a21c.jsonl`)

One operand packed once with `imd_pack` (`imd_pack_desc_for` orders, so C is bit-identical to
the plain call: checked in every record) and reused by R contractions; `imd_bench --methods
packed` times, per repetition and interleaved in one process, `R × imd_contract` against
`imd_pack + R × imd_contract_packed` (same partner, same C), plus each call alone
(`bench/sweep_d1.txt`, quiet-gated, `warm_run.sh`; clocks 2490–2805 MHz, power-capped in the
long sequences). Ratio = plain / packed (> 1: packing wins); "per call" = the R = 16 sequence
divided by 16, the pack subtracted.

| case (m×n×k) | N | packed (role) | imd_pack ms | one call ms | pack / call | R = 1 | R = 4 | R = 16 | per call at R = 16: plain → packed ms | workspace plain → packed (pack) |
|---|---|---|---|---|---|---|---|---|---|---|
| sq4096 (4096×4096×4096) | 8 | A (A) | 0.322 | 2.730 | 0.12 | 0.94× | 1.12× | 1.14× | 3.098 → 2.700 (13 %) | 0.375 → 0.250 GiB (0.125) |
| sq4096 | 8 | B (B) | 0.323 | 2.718 | 0.12 | 0.95× | 1.14× | 1.13× | 3.106 → 2.731 (12 %) | 0.375 → 0.250 GiB (0.125) |
| sq4096 | 14 | A (A) | 0.438 | 4.916 | 0.09 | 0.98× | 1.04× | 1.07× | 5.049 → 4.680 (7 %) | 0.656 → 0.438 GiB (0.219) |
| sq4096 | 14 | B (B) | 0.439 | 5.290 | 0.08 | 1.09× | 1.05× | 1.09× | 5.096 → 4.643 (9 %) | 0.656 → 0.438 GiB (0.219) |
| sq8192 (8192×8192×8192) | 8 | A (A) | 1.224 | 19.980 | 0.06 | 1.01× | 1.06× | 1.09× | 20.596 → 18.848 (8 %) | 1.500 → 1.000 GiB (0.500) |
| sq8192 | 14 | A (A) | 1.731 | 34.499 | 0.05 | 1.02× | 1.06× | 1.05× | 34.499 → 32.791 (5 %) | 2.625 → 1.750 GiB (0.875) |
| t_cc (4096×4096×4096) | 8 | A (B) | 0.338 | 3.103 | 0.11 | 1.20× | 1.13× | 1.13× | 3.165 → 2.773 (12 %) | 0.375 → 0.250 GiB (0.125) |
| t_cc | 8 | B (A) | 0.334 | 2.987 | 0.11 | 1.16× | 1.12× | 1.13× | 3.112 → 2.725 (12 %) | 0.375 → 0.250 GiB (0.125) |
| t_cc | 14 | A (B) | 0.441 | 4.891 | 0.09 | 1.03× | 1.03× | 1.07× | 5.086 → 4.716 (7 %) | 0.656 → 0.438 GiB (0.219) |
| t_cc | 14 | B (A) | 0.443 | 5.452 | 0.08 | 1.04× | 1.05× | 1.09× | 5.157 → 4.715 (9 %) | 0.656 → 0.438 GiB (0.219) |
| t_df (4096×4096×2048) | 8 | A (A) | 0.171 | 1.521 | 0.11 | 1.00× | 1.14× | 1.13× | 1.821 → 1.601 (12 %) | 0.250 → 0.188 GiB (0.063) |
| t_df | 8 | B (B) | 0.173 | 1.551 | 0.11 | 1.01× | 1.11× | 1.11× | 1.813 → 1.619 (11 %) | 0.250 → 0.188 GiB (0.063) |
| t_df | 14 | A (A) | 0.227 | 2.576 | 0.09 | 0.98× | 1.05× | 1.08× | 2.862 → 2.632 (8 %) | 0.438 → 0.328 GiB (0.109) |
| t_df | 14 | B (B) | 0.225 | 2.483 | 0.09 | 0.85× | 1.06× | 1.08× | 2.892 → 2.652 (8 %) | 0.438 → 0.328 GiB (0.109) |
| r_skinny (16384×512×4096) | 8 | A (A) | 1.208 | 2.316 | 0.52 | 1.00× | 1.69× | **1.98×** | 2.596 → 1.236 (52 %) | 0.578 → 0.078 GiB (0.500) |
| r_skinny | 8 | B (B) | 0.066 | 2.735 | 0.02 | 1.21× | 1.03× | 1.02× | 2.574 → 2.531 (2 %) | 0.578 → 0.563 GiB (0.016) |
| r_skinny | 14 | A (A) | 1.685 | 4.073 | 0.41 | 1.08× | 1.51× | **1.79×** | 3.940 → 2.090 (47 %) | 1.012 → 0.137 GiB (0.875) |
| r_skinny | 14 | B (B) | 0.070 | 3.730 | 0.02 | 1.01× | 1.00× | 1.02× | 4.103 → 4.033 (2 %) | 1.012 → 0.985 GiB (0.027) |

- `imd_pack` costs about what the operand's KL launch costs in a call (it is KL with all N
  planes): 5–12 % of a call for the square / CC / DF shapes, 41–52 % for r_skinny's large A
  (whose KL dominates that call). So R = 1 is break-even (single calls: ±10–20 % noise).
- With reuse each later call saves that operand's KL share: 5–13 % per call for square /
  CC / DF shapes (more at N = 8, where the GEMM share is smaller), **47–52 % on r_skinny
  when the large operand is packed** (1.79–1.98× at R = 16); packing the small operand of a
  skinny shape saves ~2 %. Sustained sequences run power-capped (per call 3.1 ms vs 2.7 ms
  single on sq4096 N = 8), the same for both sides.
- Workspace saved per contraction: the operand's sX, fX, RX (`g/N` of the pack; all of it at
  g = N, here): 25–33 % for square / CC / DF shapes, 86 % for r_skinny's A; the caller keeps
  the pack (`N·b·r·k̂` bytes) instead. Details and per-phase marks: [NOTES-D1.md](NOTES-D1.md).

## Desktop GPU time-slicing (2026-10-03, docs/NOTES-K5-bimodal.md)

The "bimodal K5" was the GPU being time-sliced with desktop renderers (the Claude desktop
app, gnome-shell) whenever they draw: ~2.2 ms slices, 0.4–0.5 ms lost per switch, landing in
whichever kernel is running (K4 before D4, K5 after). Clean medium/large runs have timed-loop
spread ≤ ~2.3 %; contaminated ones ≥ ~8.7 %. Runs since the clock-weighted quiet gate
(fddc944) could be contaminated, including the A3/D4/D1 sweeps. Same-process ratios
against GEMMul8 are only partly protected (both libraries get switched, but at random).
Now: `tools/run_quiet.sh` waits up to 300 s for raw GPU utilization ≤ 1 %, and
`bench/warm_run.sh` retries a case up to 3× while its spread is > 5 % and marks every record
with `attempts` and `contaminated`. **For clean numbers, minimize the Claude desktop window
(or switch to another workspace) while a sweep runs**: an active, streaming conversation
keeps the GPU busy, and every attempt will be flagged.

## Clean re-run in bench mode (2026-10-04, `compare-clean-bb6b568.jsonl`, `d1-clean-bb6b568.jsonl`)

Console-only (no display server; no other GPU clients; persistence mode on; PCIe gen4 x16
after the reboot); warm-up and retries as above. Ratio = immerdink TFLOPS-eq / GEMMul8's
(permute + GEMMul8 where a permute is needed), same process.

| case | N = 8 | N = 14 | | case | N = 8 | N = 14 |
|---|---|---|---|---|---|---|
| sq1024 | 1.06 | 1.19 | | r_lowk | 1.86 | 2.06 |
| sq2048 | 1.23 | 1.29 | | r_bigk | 1.06 | 1.07 |
| sq4096 | 1.31 | 1.38 | | r_skinny | 1.19 | 1.26 |
| sq8192 | 1.16 | 1.17 | | r_mm | 1.05 | 1.12 |
| sq16384 | — | 1.09 | | t_nn | 1.30 | 1.36 |
| s_lowk | 2.82 | 2.76 | | t_batch | 1.13 | 1.25 |
| s_cc16 | 1.25 | 1.42 | | t_ikl | 1.58 | 1.54 |
| s_df | 1.26 | 1.46 | | t_df | 1.51 | 1.65 |
| s_perm | 1.42 | 1.52 | | t_cc | 1.65 | 1.58 |
| s_cc8 | 1.37 | 2.85 | | t_permout | 1.41 | 1.45 |
| s128 / s256 / s512 | 1.40 / 0.90 / 0.92 | ≥ 1.2 (GEMMul8 bimodal at N = 14) | | | | |

- Medium and large cases agree with the D4 sweep within ~3 %, so desktop time-slicing
  mostly hit small cases and individual phases (K5) rather than the whole-call ratios there.
- 36 of 106 records still exceed the 5 % spread threshold even with no desktop: tiny shapes
  (20–110 µs) from launch and clock jitter, plus a few medium cases at 5–6 % (r_lowk, r_mm).
  In bench mode "contaminated" therefore means "noisier than 5 %", not "time-sliced".
- **Below GEMMul8 only s256 and s512 at N = 8** (0.90 / 0.92×): roadmap A5.

**D1 packed reuse (clean):** one `imd_pack` + R contractions vs R plain calls; all
bit-identical.

| case (pack) | pack / call | R = 4 | R = 16 | per-call saving | workspace (GiB) |
|---|---|---|---|---|---|
| sq4096 (A or B), N = 8 / 14 | 12 / 10 % | 1.10 / 1.08 | 1.12 / 1.10 | 11 / 10 % | 0.38 → 0.25 / 0.66 → 0.44 |
| sq8192 (A), N = 8 / 14 | 7 / 6 % | 1.04 / 1.04 | 1.08 / 1.05 | 7 / 6 % | 1.50 → 1.00 / 2.63 → 1.75 |
| t_cc (A or B), N = 8 / 14 | 13 / 10 % | 1.10 / 1.08 | 1.13 / 1.10 | 12 / 10 % | 0.38 → 0.25 / 0.66 → 0.44 |
| t_df (A or B), N = 8 / 14 | 12 / 9 % | 1.09 / 1.07 | 1.11 / 1.09 | 10 / 8 % | 0.25 → 0.19 / 0.44 → 0.33 |
| r_skinny (A), N = 8 / 14 | 53 / 46 % | 1.63 / 1.55 | **1.97 / 1.78** | 52 / 46 % | 0.58 → 0.08 / 1.01 → 0.14 |
| r_skinny (B), N = 8 / 14 | 3 / 2 % | 1.02 / 1.01 | 1.02 / 1.02 | 2 % | ≈ unchanged |

R = 1 breaks even (1.00×) everywhere: a pack costs what the fused scale+pack it replaces costs.
