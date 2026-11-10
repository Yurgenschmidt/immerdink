# K45 single-warp CRT gate and G2b native comparators (exploration, not product code)

**What this is.** A one-off H200 experiment from the Ozaki-spike exploration (immerdink-eval `analysis/20261110-dataflow/plan.md` §6, steps 2 and 3). Nothing here is built by the immerdink CMake project.

- `k45_gate.cu` asks whether the CRT (K5) part of a proposed fused K4+K5 kernel ("K45") can run fast enough with only one warp per SM sub-partition.
  - It runs immerdink's real `k5f_crt_n` code (`k5_extract.cuh`, copied verbatim from dev 91a5cbe; see `K5_SOURCE` and `extract_k5.sh`).
  - It caps registers at 96.
  - It measures the effective IPC against the gate: 0.375 at r_lowk, 0.23 at t_df.
- `g2b_bench.cu` times the native ways to compute a self-contraction C = A·Aᵀ, both as a plain matrix and in the permuted `aiQ,bjQ->abij` layout. These are the comparators a symmetric (SYRK-style) immerdink path has to beat:
  - cuBLAS DSYRK plus a mirror pass;
  - DGEMM;
  - cuBLAS FP64 emulation;
  - cuTENSOR.
- `k45_run.sh` holds the phases of one PACE embers job. `k45_tab.py` summarises its output locally.
