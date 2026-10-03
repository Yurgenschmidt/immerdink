# A1 / A4 notes: small-problem overhead, K1 / K5 leftovers

ROADMAP A1 (the ~50–70 µs per-call floor; small shapes at 0.35–0.8× GEMMul8) and A4 (K1 on
very long rows, K5). Measure first, then fix by measured payoff. Design changes: DESIGN v1.4
(§4 KF, moduli parts, §6a v2), `spec/Pipeline.tla` (`Fused`, `MParts`).

All timings through `tools/run_bench.sh` (quiet gate) and, from 12:10 on 2026-10-03, also
under the shared GPU lock (`flock /home/ef/repos/ehrenfest-ipdps/_wip/gpu.lock`).

## 1. Where the time goes before (dev `22496a8`)

`nsys profile -t cuda` of `imd_bench --methods imd` (30 timed calls after 5 warm-up calls; the
median of the last 25), analysed from the sqlite export per timed call: the host time between
the bench's two `cudaEventRecord`s (all of `imd_contract`'s host work), the launch API time
inside it, and the GPU kernels it launched: the span from the first kernel start to the last
kernel end = kernel time + gaps between consecutive kernels. µs.

| case | N | launches | host (µs) | GPU span | kernels | gaps | per kernel (µs) |
|---|---|---|---|---|---|---|---|
| s128 | 8 | 7 | 14.3 | 53.9 | 49.8 | 4.1 | K0 1.5, K1 3.6 + 3.6, pack 11.9 + 11.5, K4 10.8, K5 6.8 |
| s128 | 14 | 7 | 14.1 | 66.6 | 62.6 | 4.1 | K0 1.5, K1 3.6 + 3.6, pack 17.2 + 16.8, K4 10.8, K5 9.1 |
| s256 | 8 | 7 | 14.6 | 54.4 | 50.3 | 4.1 | K0 1.6, K1 3.0 + 2.9, pack 11.9 + 11.5, K4 12.6, K5 6.8 |
| s256 | 14 | 7 | 17.0 | 67.5 | 63.3 | 4.2 | K0 1.6, K1 3.0 + 3.0, pack 17.1 + 16.9, K4 12.7, K5 9.0 |
| s512 | 8 | 7 | 14.3 | 65.8 | 61.7 | 4.1 | K0 1.5, K1 4.4 + 4.4, pack 11.8 + 11.5, K4 16.1, K5 11.9 |
| s512 | 14 | 7 | 14.3 | 79.3 | 75.2 | 4.1 | K0 1.5, K1 4.4 + 4.4, pack 16.3 + 15.9, K4 16.8, K5 15.8 |
| s_df | 8 | 7 | 14.3 | 97.9 | 93.7 | 4.0 | K0 1.5, K1 5.3 + 5.3, pack 11.8 + 11.5, K4 25.5, K5 32.9 |
| s_df | 14 | 7 | 14.4 | 138.5 | 134.3 | 4.2 | K0 1.5, K1 5.2 + 5.2, pack 17.2 + 16.9, K4 48.3, K5 40.0 |
| s_perm | 8 | 7 | 14.5 | 100.0 | 96.0 | 4.0 | K0 1.6, K1 7.2 + 5.4, pack 11.7 + 11.5, K4 25.6, K5 32.9 |
| s_perm | 14 | 7 | 14.7 | 139.9 | 135.8 | 4.1 | K0 1.6, K1 7.2 + 5.3, pack 17.1 + 16.9, K4 48.3, K5 39.5 |
| s_cc8 | 8 | 7 | 20.4 | 67.5 | 62.6 | 4.9 | K0 1.6, K1 4.6 + 4.4, pack 11.8 + 11.5, K4 16.1, K5 12.6 |
| s_cc8 | 14 | 7 | 15.3 | 80.2 | 76.2 | 4.0 | K0 1.6, K1 4.6 + 4.4, pack 16.3 + 16.0, K4 16.8, K5 16.6 |
| sq1024 | 8 | 7 | 14.4 | 134.6 | 130.4 | 4.1 | K0 1.5, K1 9.5 + 9.5, pack 15.6 + 15.4, K4 46.0, K5 32.9 |
| sq1024 | 14 | 7 | 14.0 | 197.2 | 193.1 | 4.1 | K0 1.4, K1 9.7 + 10.2, pack 22.3 + 21.9, K4 88.1, K5 39.6 |

What this says:
- **The host is not the bottleneck.** `imd_contract` takes 14 µs of host time for its 7
  launches (≈ 1.7 µs per `cudaLaunchKernel`), and the GPU span is 54–200 µs: the host is
  always far ahead. Cutting host-side work in `imd_contract` (argument checks, parameter
  copies) cannot buy anything measurable.
- **The gaps are small:** 0.64–0.70 µs per kernel boundary, ≈ 4 µs per call (6 boundaries).
- **The floor is the kernels themselves, at small sizes latency-bound with few blocks:**
  - the pack takes **11.5 µs per operand at N = 8 (17 µs at N = 14) whatever the size**, from
    s128 (4 blocks) to s512 (64 blocks). ncu (s128, N = 8): 4 blocks of 256 threads,
    IPC 0.98, 8.2 cycles per issued instruction: each thread converts 16 values for every
    modulus of the group (≈ 4 k instructions in one dependent stream), and only 4 SMs work;
  - K4 for 128³: 8 CTAs (one 128×256 tile per modulus, half of it padding), 2,550
    instructions per warp at 10 cycles each: the epilogue of a 128×256 tile dominates;
  - K5 (s128): 16 blocks; at s_df / sq1024 (1 M elements) 33–40 µs, ≈ 1/3 of its bound. ncu
    (s_df, N = 8): **26 % of the L1 sectors are local memory** (a dynamically indexed limb
    array in `round_scaled`), C stores use 8 of 32 bytes per sector (4 consecutive j̄ per
    thread, one store instruction per j̄), 23.6 cycles per issued instruction;
  - K1: two launches (A, B) of 3–5 µs each at small sizes.

GEMMul8 does s128 at N = 8 in 19.5 µs end to end, so it is reachable only by removing
launches *and* making each remaining kernel use the GPU.

A2 (persistent K4, merged into this branch from dev `2961ea4`) changes the K4 column only;
the same profile of dev + A2 is the baseline every "before" below refers to:

| case | N | launches | host (µs) | GPU span | kernels | gaps | per kernel (µs) |
|---|---|---|---|---|---|---|---|
| s128 | 8 | 7 | 13.2 | 49.0 | 44.8 | 4.2 | K0 1.5, K1 3.6 + 3.6, pack 11.9 + 11.5, K4 5.9, K5 6.8 |
| s128 | 14 | 7 | 15.1 | 61.8 | 57.7 | 4.1 | K0 1.5, K1 3.6 + 3.6, pack 17.3 + 16.8, K4 5.9, K5 9.0 |
| s256 | 8 | 7 | 14.0 | 49.7 | 45.5 | 4.1 | K0 1.6, K1 3.0 + 2.9, pack 12.0 + 11.5, K4 7.8, K5 6.8 |
| s512 | 8 | 7 | 13.2 | 60.8 | 56.8 | 4.1 | K0 1.5, K1 4.4 + 4.4, pack 11.9 + 11.5, K4 11.2, K5 11.9 |
| s512 | 14 | 7 | 13.4 | 74.1 | 70.1 | 4.0 | K0 1.5, K1 4.4 + 4.4, pack 16.4 + 16.0, K4 12.0, K5 15.5 |
| s_df | 8 | 7 | 22.4 | 88.5 | 83.4 | 5.2 | K0 1.5, K1 5.3 + 5.2, pack 11.8 + 11.5, K4 15.4, K5 32.6 |
| s_df | 14 | 7 | 13.7 | 116.4 | 112.3 | 4.1 | K0 1.5, K1 5.2 + 5.2, pack 17.2 + 16.9, K4 27.0, K5 39.3 |
| s_cc8 | 8 | 7 | 14.1 | 61.6 | 57.4 | 4.1 | K0 1.6, K1 4.4 + 4.4, pack 11.8 + 11.5, K4 11.3, K5 12.3 |
| sq1024 | 8 | 7 | 18.1 | 124.6 | 119.8 | 4.9 | K0 1.4, K1 9.6 + 9.7, pack 15.6 + 15.3, K4 35.7, K5 32.5 |
| sq1024 | 14 | 7 | 13.2 | 177.5 | 173.4 | 4.1 | K0 1.4, K1 10.0 + 10.2, pack 22.4 + 22.1, K4 68.0, K5 39.2 |

## 2. Changes (in order of measured payoff)

| # | change | where | kind |
|---|---|---|---|
| 1 | **KF**: K0 + K1 + group 1's packs in **one launch** for k̂ ≤ 512 | `contract.cu` `kf_scale_pack`, `launch_kf`; plan-time choice `analysis.hpp fused_path` | DESIGN v1.4 §4, Pipeline.tla `Fused` |
| 2 | KF with **4 values per thread** (u32 stores; rows of up to 128 lanes, shared-memory combine across a row's warps) when the 16-per-thread grid would have < 16 blocks per SM after the moduli split | `kf_scale_pack<Cap, 4>` | launch choice |
| 3 | **Moduli parts**: a pack launch (K2/K3, KF) with fewer than 2 tiles per SM splits the group's slots over `gridDim.y` | `pack_split` | launch choice, Pipeline.tla `MParts` |
| 4 | **K5**: one element per thread for ≤ 2048·SMs outputs; four per thread otherwise, with a warp's results transposed through shared memory so C stores and `c_col` loads are coalesced | `k5_crt<L, E>` | launch choice |
| 5 | **No local memory** in K5 and KF: `round_scaled` selected `x[s/32 + j]` (compiled into a dynamically indexed local copy: 26–38 % of K5's L1 sectors), `shl256` indexed `x.w[i − ws]` | `kernels/numerics.cuh` | same integers |
| 6 | Residue emit **branch-free** (p = 256 through the same byte-weight dot + umulhi64 with magic 2^56), the rare eps > 0 factor hoisted | `PackEmitterT` | same values |
| 7 | 32-bit division when operands fit (K5 index split, descriptor offsets) | `divmod` | — |
| 8 | **K1 long rows** (A4): a capped 1024-thread grid is rounded to a multiple of the SM count (no second partial wave of grid-strided blocks); with more rows than the cap, two 512-thread blocks per SM | `k1_blocks`, `launch_k1` | launch choice |

Not adopted (measured):
- KF moduli split targeting 4 blocks per SM: s128 N = 14 KF 6.0 → 8.5 µs (every part
  recomputes σ), so the split keeps targeting 2 blocks per SM.
- KF 4-per-thread only for tk ≤ 128 (the first version): s256 / s512 / s_df stayed at
  13–16 µs; extending it to every tk with the cross-warp combine gave 7–13 µs.
- Host-side trimming of `imd_contract`: nothing to gain (§1: the host is far ahead).
- K4 for small shapes (A3, tile selection) was not touched: K4 is 5.6 µs at s128 and
  10.8 µs at s512 (one 128×256 tile per modulus at s128, half of it padding).

Bit-exactness: every change keeps the §3 integers (σ is the same exact computation, only
reduced across lanes; residues the same congruence class; K5 the same chain). Every
contract test case runs three times per vec mode (plan-time path; unfused with the
4-per-thread K5; fused without the small-problem variants), each bit-exact vs the oracle
and bit-identical to the others, and the kernel launches of every call are counted (the
LD_PRELOAD shim) and checked against the plan and `imd_lowering`.

## 3. Where the time goes after (`19335ff`, nsys as in §1)

| case | N | launches | host (µs) | GPU span | kernels | gaps | per kernel (µs) |
|---|---|---|---|---|---|---|---|
| s128 | 8 | 3 | 6.0 | 17.1 | 15.7 | 1.4 | KF 6.1, K4 5.6, K5 4.0 |
| s128 | 14 | 3 | 6.3 | 18.0 | 16.6 | 1.4 | KF 6.1, K4 5.6, K5 4.8 |
| s256 | 8 | 3 | 6.1 | 20.5 | 19.1 | 1.4 | KF 7.1, K4 7.5, K5 4.6 |
| s256 | 14 | 3 | 19.1 | 22.5 | 21.2 | 1.3 | KF 8.0, K4 7.5, K5 5.6 |
| s512 | 8 | 3 | 6.0 | 33.9 | 32.5 | 1.4 | KF 12.3, K4 10.8, K5 9.4 |
| s512 | 14 | 3 | 6.0 | 40.2 | 38.8 | 1.3 | KF 14.9, K4 11.5, K5 12.3 |
| s_df | 8 | 3 | 6.0 | 47.6 | 46.1 | 1.4 | KF 12.3, K4 15.1, K5 18.8 |
| s_df | 14 | 3 | 6.1 | 75.3 | 73.9 | 1.4 | KF 16.2, K4 26.7, K5 31.1 |
| s_perm | 8 | 3 | 6.0 | 48.4 | 46.8 | 1.4 | KF 12.9, K4 15.2, K5 18.9 |
| s_perm | 14 | 3 | 6.1 | 76.1 | 74.7 | 1.3 | KF 16.5, K4 26.9, K5 31.3 |
| s_cc8 | 8 | 3 | 6.0 | 34.7 | 33.3 | 1.4 | KF 12.9, K4 10.9, K5 9.4 |
| s_cc8 | 14 | 3 | 6.3 | 41.2 | 39.8 | 1.4 | KF 15.5, K4 11.6, K5 12.8 |
| sq1024 | 8 | 7 | 13.6 | 108.5 | 104.3 | 4.2 | K0 1.5, K1 9.4 + 9.4, pack 14.9 + 14.6, K4 35.6, K5 18.8 |
| sq1024 | 14 | 7 | 13.4 | 164.2 | 159.9 | 4.2 | K0 1.5, K1 9.5 + 9.5, pack 20.6 + 20.3, K4 67.6, K5 30.9 |

s128 at N = 8: 49.0 → 17.1 µs of GPU time (7 → 3 launches, gaps 4.2 → 1.4 µs; KF 6.1 µs
does what K0 + K1 + the packs did in 32 µs). sq1024 (k̂ = 1024 > 512) keeps the unfused
sequence; its gain is K5 (32.5 → 18.8 µs at N = 8).

## 4. Results vs GEMMul8 (same session, quiet-gated, `bench/results/a1-ab2-*.jsonl`)

Each case ran the baseline bench (dev + A2 `2961ea4`: GEMMul8 and imd) and the A1 bench
(`19335ff`: imd) back to back, each in its own gated process, 40 timed calls. Median µs;
"vs G8" = GEMMul8 time / imd time (> 1: imd faster). For the permuted cases (s_cc8,
s_cc16, s_perm) GEMMul8 runs on pre-permuted operands; the fair comparison is perm +
GEMMul8 (in parentheses).

| case | N | GEMMul8 | imd before | imd after | after / before | before vs G8 | **after vs G8** | after TFLOPS-eq |
|---|---|---|---|---|---|---|---|---|
| s128 | 8 | 18.4 | 54.0 | 19.2 | 0.36 | 0.34× | **0.96×** | 0.22 |
| s128 | 14 | 92.4 ¹ | 71.7 | 23.6 | 0.33 | 1.29× | **3.9×** | 0.18 |
| s256 | 8 | 21.5 | 51.2 | 23.4 | 0.46 | 0.42× | **0.92×** | 1.43 |
| s256 | 14 | 111.6 ¹ | 73.6 | 24.6 | 0.33 | 1.52× | **4.5×** | 1.37 |
| s512 | 8 | 33.8 | 65.5 | 36.1 | 0.55 | 0.52× | **0.94×** | 7.43 |
| s512 | 14 | 91.7 ¹ | 75.8 | 49.4 | 0.65 | 1.21× | **1.86×** | 5.43 |
| s_lowk | 8 | 303.0 | 191.6 | 106.5 | 0.56 | 1.58× | **2.84×** | 5.04 |
| s_lowk | 14 | 483.7 | 239.4 | 181.9 | 0.76 | 2.02× | **2.66×** | 2.95 |
| s_batch | 8 | 4432 | 182.1 | 110.6 | 0.61 | 24× | **40×** | 1.21 |
| s_batch | 14 | 6897 | 237.6 | 186.9 | 0.79 | 29× | **37×** | 0.72 |
| s_cc8 | 8 | 33.7 (46.0) | 68.6 | 39.9 | 0.58 | 0.49× (0.67×) | **0.84× (1.15×)** | 6.72 |
| s_cc8 | 14 | 72.3 (138.8) ¹ | 82.9 | 47.1 | 0.57 | 0.87× (1.7×) | **1.54× (2.9×)** | 5.70 |
| s_cc16 | 8 | 284.7 (369.3) | 334.8 | 307.1 | 0.92 | 0.85× (1.10×) | **0.93× (1.20×)** | 23.60 |
| s_cc16 | 14 | 486.4 (576.4) | 455.7 | 437.2 | 0.96 | 1.07× (1.26×) | **1.11× (1.32×)** | 16.58 |
| s_df | 8 | 67.6 | 95.2 | 52.4 | 0.55 | 0.71× | **1.29×** | 10.24 |
| s_df | 14 | 110.8 | 121.8 | 81.6 | 0.67 | 0.91× | **1.36×** | 6.58 |
| s_perm | 8 | 67.6 (150.0) | 96.3 | 53.2 | 0.55 | 0.70× | **1.27× (2.8×)** | 10.08 |
| s_perm | 14 | 109.6 (118.8) | 146.5 | 88.1 | 0.60 | 0.75× (0.81×) | **1.24× (1.35×)** | 6.10 |
| sq1024 | 8 | 102.2 | 123.9 | 107.8 | 0.87 | 0.83× | **0.95×** | 19.93 |
| sq1024 | 14 | 187.4 | 183.3 | 169.0 | 0.92 | 1.02× | **1.11×** | 12.71 |
| sq2048 | 8 | 596.7 | 628.2 | 576.5 | 0.92 | 0.95× | **1.04×** | 29.80 |
| sq2048 | 14 | 984.1 | 874.5 | 844.8 | 0.97 | 1.13× | **1.16×** | 20.34 |
| r_bigk | 8 | 7270 | 8056 | 7537 | 0.94 | 0.90× | **0.96×** | 36.47 |
| r_bigk | 14 | 11175 | 11456 | 11052 | 0.96 | 0.98× | **1.01×** | 24.87 |

¹ GEMMul8 at N = 14 on the smallest shapes is bimodal in this session (spread 25–54 %;
s128 min 81.5 µs; an earlier session measured 49 µs). Its N = 8 times are stable (spread
6–9 %), so N = 8 is the comparison that matters for the small-shape floor.

Large shapes (`bench/results/a1-ab1-*.jsonl`, 10 timed calls, every case of
`bench/sweep_compare.txt` except sq16384, commit `14497c8` = this work minus the last two
commits): after / before is 0.89–1.03 (no regression beyond noise; K5 gives 1–10 %):
sq4096 1.15× / 1.29× GEMMul8 (N = 8 / 14), sq8192 1.12× / 1.15×, r_lowk 1.83× / 2.04×,
t_df 1.35× / 1.53×, t_cc 1.15× / 1.27×, t_batch 1.06× / 1.18×, t_nn 1.16× / 1.29×,
r_skinny 0.92× / 1.07×, r_mm 0.89× / 1.06×.

## 5. A4: per-phase before / after (bench phase marks, ms; % of the MEMORY_MODEL §4 bound)

| case | N | phase | before (dev + A2) | after | after % of bound |
|---|---|---|---|---|---|
| r_bigk | 14 | K1 scale A | 1.001 (55 %) | 0.801 / 0.770 ² | 67–69 % |
| r_bigk | 14 | K1 scale B | 0.978 | 0.770 / 0.778 ² | 69 % |
| r_bigk | 8 | K1 scale A / B | 0.986 / 0.962 | 0.794 / 0.773 | 67 / 69 % |
| sq8192 | 8 | K5 crt | 1.743 (61 %) | 1.165 | 91 % |
| sq8192 | 14 | K5 crt | 1.995 (74 %) | 1.789 | 82 % |
| r_lowk | 8 | K5 crt | 1.664 (64 %) | 1.165 | 91 % |
| r_lowk | 14 | K5 crt | 1.957 (75 %) | 1.740 | 84 % |
| sq4096 | 8 | K5 crt | 0.432 (62 %) | 0.281 | 95 % |
| sq4096 | 14 | K5 crt | 0.494 (74 %) | 0.442 | 83 % |
| t_cc | 14 | K5 crt | 0.493 (74 %) | 0.443 | 83 % |
| r_bigk | 8 | K5 crt | 0.114 | 0.070 | 96 % |

² first value: the grid rounding alone (`a1-ab1`), second: plus the 512-thread variant
(one run; its 4 % is at the noise level).

K5 at N = 14 (4 limbs) is now instruction-bound, not memory-bound: ncu at s_df, N = 14:
26 M warp instructions for 1 M outputs (≈ 830 per output: 14 CRT steps of ≈ 20
instructions on 4 limbs, the centering and the rounding), 53 % SM throughput, no local
memory. Going further needs a different CRT realization (DESIGN §3.6; out of scope here).
K1 for long rows is at 67–69 % of the one-DRAM-read bound; its second pass re-reads from
L2 by design (the two-launch split was measured slower in v1.2).

## 6. The remaining floor

- **s128 at N = 8 (0.96× GEMMul8):** 17.1 µs of GPU time = KF 6.1 + K4 5.6 + K5 4.0 + two
  gaps 1.4, plus ≈ 2 µs from the first launch to the first kernel. Three dependent kernels
  (scale + pack → GEMM → CRT) are the minimum for this pipeline: K4 needs every residue of
  both operands (a grid-wide dependency) and K5 every plane, and each kernel costs ≈ 2–3
  µs of launch, fill and drain even when tiny. Going lower needs the CRT inside the GEMM
  for small tiles (all N moduli per CTA; DESIGN §7 rejected it for useful tiles because of
  registers) or K4 and K5 in one persistent launch.
- **s256 / s512 at N = 8 (0.92× / 0.94×):** K4 (7.5 / 10.8 µs: one 128×256 tile per
  modulus, few CTAs: A3 tile selection), KF at s512 (12.3 µs, instruction-bound: σ plus 8
  residues per element) and K5 (9.4 µs for 262 K outputs, instruction-bound CRT).
- **N = 14:** at or above GEMMul8 on every small shape.
- **k̂ > 512 (sq1024, s_cc16)** keeps K0 + 2 K1 + 2 packs (5 launches, ≈ 50 µs at sq1024,
  N = 8); a KF for k̂ ≤ 1024 (8192-value tiles) is the next step there, with A3 for K4.
