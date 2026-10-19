# Which GPUs immerdink supports, and how well

This page is for someone who wants to know whether immerdink runs on their GPU, what the library tells
them about it, and whether emulating FP64 with INT8 beats the GPU's own FP64 there.
The design behind it is in [NOTES-18.md](NOTES-18.md).

## What the library reports

immerdink emulates 64-bit floating-point matrix products with 8-bit integer tensor-core products.
Its speed depends on constants that were measured on one card, the RTX 4090. For any other card
the library has to answer two questions at plan creation, and it tells you the answers:

* **Which constants does this card get?** (the *table match*) `exact` means a table fitted on this
  very card. `arch` means a table fitted on another card with the same compute capability.
  `generation` means one from the same major architecture generation. `default` means the RTX
  4090's table, because nothing is closer.
* **What kind of code runs?** (the *code kind*) `native` is machine code built for this exact
  architecture. `compatible` is prebuilt code of an older minor version of the same generation
  (sm_80 code on an sm_86 card). `recompiled` means the library carries only portable code (PTX)
  for this card and the driver compiled it when the first plan was made. `unusable` means the
  library cannot run on this card with this build or driver.

The two combine into one **plan status**:

| status | meaning |
|---|---|
| `native` | tables fitted on this card, code built for it |
| `approximate` | it runs, with constants fitted on another card (or older-minor code); results are exact, speed and the planner's predictions are less reliable |
| `recompiled` | like the above, and the first plan creation was slower because the driver compiled the kernels |
| `unsupported` | no plan can be made; `imd_plan_create` returns `NOT_SUPPORTED` and `imd_device_info` says which kernel failed and why |

Results never depend on the status: the INT8 sums are exact integers, so the output is bit-identical
on every card that runs. Only speed, and the accuracy of the performance predictions, differ.

Ask for it in C with `imd_device_info(handle, &info)` and `imd_plan_info(plan, &info)`, in Python with
`immerdink.device_info()` and `plan.info`. The rule itself, for a card you do not have, is
`imd_device_select` / `immerdink.host.device_select((major, minor), sm_count, name)`.

## The cards

| Card | Compute capability | SMs | L2 | Status with the default build (arch 89) | Status with the release build | K4 path | Tables |
|---|---|---|---|---|---|---|---|
| RTX 4090 | 8.9 | 128 | 72 MiB | `native` (observed, 110 kernels checked in ~80 ms) | `native` | `mma.sync` (sm_80 code path) | exact (sm89, fitted here) |
| H100, H200 | 9.0 | 132 | 50 MiB (H200: assumed, not read from a card) | `recompiled`, or `unsupported` (CUDA error 222) if the driver is older than the CUDA toolkit that built the library. Expected, not yet observed on the card | `approximate` (table: `default`, i.e. the sm89 constants) | `mma.sync`; Hopper's `wgmma` is the subject of issues 16, 17 and 22 | `default` until issues 20 and 23 add sm_90 rows |
| RTX PRO 6000 (Blackwell) | 12.0 | 188 | 128 MiB | as for Hopper | `approximate` (`default`) | `mma.sync`: ptxas 13.4.92 rejects integer `wgmma` / `tcgen05` for sm_120a (checked 2026-10-12) | `default` until issue 24 |
| RTX 5090 (not measured; synthetic profile) | 12.0 | 170 | 96 MiB | not run | `approximate` (`default`; `arch` once an sm_120 row exists) | `mma.sync` | `default` |
| A100, RTX 3090 and other sm_80 / sm_86 cards (not measured) | 8.x | varies | | `unsupported` (the default build's PTX is for architecture 89, which an older card cannot compile); `compatible` if sm_80 machine code is built (the release build), `recompiled` if PTX of architecture 80 is | `approximate` (`generation`: the sm89 row) | `mma.sync` | `generation` |
| anything below compute capability 8.0 | | | | `unsupported` | `unsupported` | - | - |

"Release build" is `-DIMMERDINK_CUDA_ARCHS="80-real;89;90-real;120-real"`: machine code for each
shipped architecture plus portable code of sm_89. Its build time and size are about three to four times
those of the default (four K4 configurations for each architecture).
Rows marked "expected" follow from the rules in [NOTES-18.md](NOTES-18.md) and the measurements in it; the
first campaign run (R1) prints the real status of each cluster card.

Native FP64 is never used as a fallback. cuTENSOR and cuBLAS cover that, and calling them is the
caller's choice.

## Where emulation beats the card's own FP64

"Crossover": the smallest square size at which immerdink at DGEMM-level accuracy (N = 14 moduli,
max normwise error at or below cuBLAS DGEMM's) is faster than cuBLAS DGEMM on the same card.
All numbers are TFLOPS-equivalent (2mnk / time) from the issue 11 cluster runs
(`scratch/runs/20261011-180837` in generalWork, commit a52bd9b of the benchmark harness; `*` marks a
record whose own timing spread was above the limit, so read it as a hint).

| Card | 256³ | 512³ | 1024³ | 2048³ | 4096³ | 8192³ | Crossover |
|---|---|---|---|---|---|---|---|
| H100 | 1.35* vs 2.46* (0.55×) | 6.66* vs 14.1* (0.47×) | 13.9* vs 40.9* (0.34×) | 28* vs 48* (0.58×) | 42.8 vs 50 (0.86×) | 54.4* vs 45.6* (1.19×) | between 4096³ and 8192³ (only the 8192³ point is above 1, and noisy) |
| H200 | 1.36* vs 2.51* (0.54×) | 6.72* vs 14.3* (0.47×) | 14.3* vs 41.1* (0.35×) | 28.1 vs 48.7 (0.58×) | 43 vs 50* (0.86×) | 53.1* vs 46.8* (1.13×) | as H100 |
| RTX PRO 6000 | 1.79* vs 0.495* (3.61×) | 8.37* vs 1.06* (7.93×) | 20.7* vs 1.43* (14.5×) | 30.8 vs 1.58 (19.5×) | 44 vs 1.58 (27.8×) | 50.5* vs 1.58* (31.9×) | none: faster at every size measured (its FP64 rate is about 1.6 TFLOPS) |
| RTX 4090 | not in the issue 11 runs | | | | | | not measured here; its datasheet FP64 rate is 1/64 of its FP32 rate (about 1.3 TFLOPS), as low as the RTX PRO 6000's, so expect none |

Each cell is "immerdink vs cuBLAS DGEMM (ratio)". Rows for cards other than the three above are
"not measured". Cross-card comparisons are not made: each card is different hardware.
On Hopper the best other emulator (mostly cuBLAS's own FP64 emulation) is faster than immerdink
today in 12 of 17 benchmark cases (tracking issue 14); closing that gap is the point of issues 16 to 22.

## Limits new K4 configurations must fit

* At most **4 K4 configurations** (`IMD_K4_CONFIG_COUNT`), and each one's launch parameters must fit the
  **256-byte parameter block** (`kK4BlobBytes`, `src/k4/k4_ops.hpp`). Both are checked by the test
  `device_kernel_sites`; a wgmma configuration (issue 22) either fits them or changes them together with
  the stored plan layout.
* Every kernel launched anywhere in the library must appear in the device-status check
  (`src/contract.cu` `kernel_occupancy_w` / `extra_kernel_versions`, `src/k4/k4_cfg.cuh`); the same test
  fails otherwise.

## Notes

* **Driver and toolkit.** Portable code (PTX) written by CUDA toolkit X can only be compiled by a driver
  that supports X. On the development machine (toolkit 13.4, driver 595.99, CUDA 13.2) a PTX-only build is
  refused with error 222. A build with machine code for the card does not need this.
* **Latency.** The first plan on a device loads and checks every kernel of the library, about 0.1 s when
  the code is native. With `recompiled` code the driver compiles all of them then, which can take seconds.
  Later plans on that device are not affected.
* **`CUDA_MODULE_LOADING=EAGER`** loads the kernels at the first CUDA call; an unusable build then fails
  there, and the library reports it the same way (`NOT_SUPPORTED`, with the CUDA error).
