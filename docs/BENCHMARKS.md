# Benchmarks

M4 benchmark harness and the first baseline sweep.

> **Status: starting point, not a result.** immerdink here is the **M1 library,
> before any M4 optimization** (commit `e4d8845` = dev `c1b0bba` + the bench). The
> numbers set the baselines that M4 work is measured against.

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
| `imd N=…` | immerdink `imd_plan_create` once, then `imd_contract` timed. Settings: RN, g = N, β = 0, workspace from `imd_workspace_size` |
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
python3 bench/summarize.py bench/results/<run>.jsonl     # the tables below
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

## First baseline sweep (2026-10-03)

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

## Phase breakdown: what is missing

The C API has no timing hooks, so the phase table compares the whole call with the sum
of the §4 lower bounds. M4 wants:
1. **Per-kernel event timing.** An opt-in `IMD_PROFILE` env var, or an
   `imd_contract_profiled` taking an `imd_profile_t*`, that records cudaEvents between
   K0/K1/K2-3/K4/K5 and returns per-phase ms. This gives the "% of §4 bound" exit metric
   per phase directly, without Nsight.
2. NVTX ranges per kernel, so `nsys` can attribute time without source changes.
3. Achieved bytes per phase from the plan (`params.workspace_size`-style formulas,
   exposed), so the bench can print achieved GB/s per phase instead of re-deriving §4
   in Python.
4. Plan-time vs call-time separation is already there (only `imd_contract` is timed);
   for small shapes, an option to skip the K0 H2D vector copy when the plan is reused.

## Not covered yet

- No **`emu fix` at equal accuracy** sweep on non-square cases (only `dyn` there).
- **GEMMul8 accurate mode** and the cuBLAS emulation's mantissa offset are not tried.
- **Clock locking**: clocks are recorded, not locked (governor `powersave`, turbo on).
  The spread column shows run-to-run noise. Most variants are under 5 %; a few small
  ones are not.
