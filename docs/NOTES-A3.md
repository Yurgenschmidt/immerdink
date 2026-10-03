# A3 notes: shape-dependent K4 tile selection

ROADMAP A3: K4 runs one configuration (128×256×64, 3 stages, 8 warps of 64×64, 1 CTA/SM)
for every m:n:k. Where immerdink trails GEMMul8 in the same-session sweep
(`bench/results/compare-b2c45cd.jsonl`, N = 8): s128 / s256 / s512 0.84–0.86×, sq1024
0.94×, r_skinny (16384×512×4096) 0.94×, r_bigk (2048²×32768) 0.98×, r_mm (1000×64×50000)
0.81×; at N = 14 ≥ 1× nearly everywhere (r_mm 0.99×). This file has the measurements first
(dev `fa77ec0`, before any change), then the design, the changes and the before/after
numbers.

Tooling as in NOTES-A2: every profile and timing through `tools/run_bench.sh` (quiet gate,
memcap, the shared GPU lock). ncu 2026.3.1 (counters enabled) on the first K4 launch of a
call (`imd`, N = 8, g = N: one store-mod launch per call), `--set full` plus
`sm__pipe_tensor_cycles_active` and `dram__bytes_read`; ncu locks the SM clock at 2.5 GHz,
so its durations compare with each other, not with event timings. nsys 2026.5.1 for kernel
times per call; the bench's phase marks (median of 5 profiled calls) for the per-phase
split and the MEMORY_MODEL §4 bound.

## 1. Measurements before any change (dev `fa77ec0`)

### K4 on the trailing shapes (ncu, N = 8, one launch = the whole K4 phase)

Tiles = ⌈m/128⌉·⌈n̂/256⌉·N; the persistent grid is min(tiles, 128 SMs × 1). "Per tile" is
the average SM-active time of an SM that has work (SM active cycles × 128 / CTAs / tiles
per CTA). "Tensor (active)" is `sm__pipe_tensor_cycles_active` over the cycles the SMs
were active, "tensor (elapsed)" over the whole kernel and all 128 SMs.

| case (view m×n̂×k̂) | tiles | grid | waves | K4 µs | per tile | tensor (elapsed) | tensor (active) | DRAM | L2 |
|---|---|---|---|---|---|---|---|---|---|
| s128 (128×128×128) | 8 | 8 | 0.06 | 6.8 | 5.7 µs | 1.5 % | 29 % | 5 % | 3 % |
| s256 (256×256×256) | 16 | 16 | 0.12 | 8.0 | 7.7 µs | 4.6 % | 43 % | 14 % | 7 % |
| s512 (512×512×512) | 64 | 64 | 0.5 | 12.2 | 11.3 µs | 25 % | 57 % | 36 % | 26 % |
| sq1024 (1024³) | 256 | 128 | 2 | 34.5 | 17 µs | 72 % | 76 % | 50 % | 62 % |
| s_cc16 (1536³) | 576 | 128 | 4.5 | 112.7 | 25 µs | 76 % | 86 % | 34 % | 60 % |
| r_lowk (8192²×1024) | 16384 | 128 | 128 | 1969 | 15.4 µs | 84 % | 84 % | 33 % | 72 % |
| r_skinny (16384×512×4096) | 2048 | 128 | 16 | 947 | 59 µs | 94 % | 95 % | 66 % | 69 % |
| r_bigk (2048²×32768) | 512 | 128 | 4 | 3442 | 860 µs | 99 % | 99 % | 33 % | 74 % |
| r_mm (1000×64×50000) | **64** | 64 | 0.5 | 583 | 583 µs | **49 %** | **99 %** | 75 % | 35 % |

(All with 256 threads, 248 registers, 72 KB of shared memory: 1 CTA and 8 warps per SM,
occupancy 16.7 %.)

What this says, per shape class:
- **Small (s128–s512): K4 is one partial wave of 128×256 tiles; each tile is a long
  dependent chain on one SM.** s128 runs 8 CTAs on 8 of 128 SMs; each takes 14.3 K cycles
  for 2 k-iterations, 1,795 issued instructions per warp at 7.7 cycles each (2 warps per
  scheduler): the register epilogue of a 128×256 tile (128 accumulators per thread, ~10
  instructions each) dominates the 2–8 k-iterations of main loop. Half of s128's tile is
  padding (n̂ = 128 < 256). Smaller tiles give more CTAs (more SMs) and a shorter epilogue
  per thread (a 32×64 warp tile holds 64 accumulators per thread, a 64×64 one 128).
- **sq1024 / s_cc16 (k̂ = 1024–1536, 2–4.5 waves): 72–76 % tensor.** 2 tiles per CTA at
  sq1024, so the exposed first prologue and last epilogue of each CTA are a large share;
  s_cc16 has a half wave of tail (576 tiles on 128 CTAs). Moderate-k epilogue cost as in
  NOTES-A2 (no second accumulator set per SM at 128×256).
- **r_skinny: K4 is at its bound.** 2·8·16384·512·4096 ops / 660 TOPS = 0.83 ms (compute;
  the residue reads are 0.55 ms), measured 0.93–0.95 ms in the bench = 89 % of the bound,
  tensor pipe 94 %. Tile selection cannot buy anything here.
- **r_bigk: K4 is at its bound** (99 % tensor, 98 % of the §4 bound in the bench).
- **r_mm: K4 is compute-bound on padding with half the SMs idle.** n̂ = 64 fills a quarter of
  the 256-wide tile; 8 M-tiles × 1 N-tile × 8 moduli = 64 tiles on 128 SMs. Each active SM
  runs its tensor pipe at 99 % for 782 k-iterations of a 128×256×64 tile of which 3/4 are
  zero columns: 0.58 ms (ncu) / 0.66 ms (events) against a §4 bound of 0.42 ms, which is
  the *memory* bound (reading the 400 MB of A residues once; the useful MMA work is
  0.08 ms). A tile with n = 64 (and more of them) makes it memory-bound.

### r_mm and r_skinny: which phase (bench phase marks, N = 8, `compare-b2c45cd`)

| case | phase | ms | §4 bound | % of bound |
|---|---|---|---|---|
| r_mm | K1 scale A | 0.864 | 0.397 | **46 %** |
| r_mm | K1 scale B | 0.067 | 0.025 | 38 % |
| r_mm | K2 pack A | 0.962 | 0.794 | 83 % |
| r_mm | K3 pack B | 0.080 | 0.051 | 64 % |
| r_mm | K4 gemm | 0.664 | 0.423 | **64 %** |
| r_mm | K5 crt | 0.014 | 0.001 | 7 % |
| r_mm | whole call | 2.635 (GEMMul8 2.144) | | 0.81× |
| r_skinny | K1 scale A | 0.635 | 0.533 | 84 % |
| r_skinny | K2 pack A | 1.226 | 1.065 | 87 % |
| r_skinny | K4 gemm | 0.932 | 0.832 | 89 % |
| r_skinny | K5 crt | 0.135 | 0.133 | 99 % |
| r_skinny | whole call | 2.989 (GEMMul8 2.812) | | 0.94× |

(nsys of the same build agrees: r_mm K1 A 862 µs, K4 660 µs; r_skinny K4 926 µs.)

- **r_mm: K1 A is the largest loss, then K4.** ncu of K1 A: `k1_scale<512>` with 256 blocks
  (two per SM, the A4 long-row variant), DRAM throughput 95 % of peak but **L2 hit rate
  34 %**: 1000 rows of 50000 doubles (400 KB each); 256 rows in flight are 102 MB, more
  than the 72 MB L2, so the second pass (S, re-reading the row after γ) misses and K1
  reads A twice from DRAM (867 µs × 930 GB/s ≈ 2 × 400 MB). r_bigk's rows (256 KB, 64 MB
  in flight) hit (L2 70 %), r_skinny's (32 KB rows, 1280 blocks of 256) too (L2 49 %).
  This is cheap to fix (task 4): keep the rows in flight within the L2 budget and walk the
  second pass backwards, so the most recently read part of a row (the part still in L2) is
  re-read first; S and γ are exact and order-independent (u128 sum, max), so the results
  do not change.
- **r_skinny: neither K4 nor anything cheap.** Every phase is at 84–99 % of its bound; the
  gap to GEMMul8 (~6 %) is the sum of K1 A (84 %) and the pack of A (87 %) streaming the
  537 MB operand three times (K1 two passes, one mostly from L2; the pack). Closing it needs
  fewer passes over A (K1 fused into the pack for long rows: ROADMAP D4-type fusion, a
  kernel-sequence change), not tile selection.

## 2. Design (DESIGN v1.5 §4, §6, §6a)

- **Configurations** (`src/host/k4_config.hpp kK4Configs`, one K4Gemm instantiation each in
  `src/k4/k4_cfg<i>.cu`): 128×256 (8 warps of 64×64, 3 stages, 1 CTA/SM: the A2 kernel),
  128×128 (4 warps of 64×64, 3 stages, 2 CTAs/SM), 64×128 (4 warps of 32×64, 4 stages,
  2 CTAs/SM), 64×64 (2 warps of 32×64, 4 stages, 3 CTAs/SM). The register epilogue's in-quad
  transpose needs 64-column warp tiles, so the small tiles use 32×64 warp tiles (64
  accumulators per thread instead of 128: half the per-thread epilogue chain). All share
  K4Gemm (persistent, swizzle-8, register epilogue) and the CUTLASS main loop; every W byte
  is the same exact reduction, so the results are bit-identical (tested on every contract
  suite and on `cutlass_mod_epilogue`, every configuration).
- **One launch structure.** The n̂ split stays in units of 65535·256 columns for every
  configuration, so launches, `gemm_launches`, `kernel_launches`, the workspace and the
  Pipeline spec do not depend on the choice; the plan's parameter sets are a fixed-size
  blob (K4Blob), so neither do the plan host bytes.
- **Selection** `k4_select` (libimmerdink_host, reported by `imd_lowering` v3 `k4_config`
  for a caller-given `sm_count`): a per-configuration time estimate from the tile count,
  the busiest SM's rounds and concurrency, and two fitted constants per configuration and
  concurrency (§3); a smaller tile must beat 128×256's estimate by 5 %. Integer arithmetic
  (saturating u128), transcribed independently in `tests/test_host.py`.
- **Autotuning** (`imd_options_t.k4_tune`, default off): times group 1's K4 launches once
  per configuration at plan creation in `k4_tune_bytes` of temporary device memory (0 and no
  tuning above 256 MiB). Cost in §4.

## 3. Configurations measured, and the rule fit

Every configuration, forced through the benchmark hook (`imd_bench --k4 0,1,2,3`, one
process per (case, N), g = N), on 42 GEMM views × N = 8, 14 (84 cases): nsys K4 kernel
medians (30 timed calls per configuration; `bench/results/a3-configs-nsys.jsonl`) for 31
small and medium views, and the bench's K4 phase marks for the large ones
(`bench/sweep_a3.txt`, `bench/results/a3-configs-phases.jsonl`; phase marks below
~0.1 ms are too coarse, so the nsys runs replaced them there). K4 µs, the per-case best in
bold; "rule" is `k4_select`'s choice (fitted on these data, 128 SMs) and its time relative
to 128×256:

| case | N | view | 128×256 | 128×128 | 64×128 | 64×64 | rule | rule / 128×256 |
|---|---|---|---|---|---|---|---|---|
| b16_512x64 | 8 | 512×64×512 b=16 | 38.1 | 21.7 | 21.9 | **18.8** | 64x64 | 0.49 |
| b16_512x64 | 14 | 512×64×512 b=16 | 68.3 | 45.8 | 46.8 | **44.8** | 64x64 | 0.66 |
| b4_256 | 8 | 256×256×512 b=4 | 9.6 | **6.7** | 7.7 | 10.0 | 128x128 | 0.69 |
| b4_256 | 14 | 256×256×512 b=4 | **10.2** | 10.8 | 12.1 | 15.3 | 128x256 | 1.00 |
| b8_128 | 8 | 128×128×128 b=8 | 5.6 | 4.4 | **3.4** | 3.5 | 64x128 | 0.61 |
| b8_128 | 14 | 128×128×128 b=8 | 5.5 | 4.2 | **4.1** | 5.1 | 64x128 | 0.74 |
| r_bigk | 8 | 2048×2048×32768 | **3421.2** | 3932.2 | 5780.5 | 8058.9 | 128x256 | 1.00 |
| r_bigk | 14 | 2048×2048×32768 | **5945.3** | 6963.2 | 10777.6 | 15727.6 | 128x256 | 1.00 |
| r_lowk | 8 | 8192×8192×1024 | **1990.7** | 2069.5 | 2694.1 | 3591.2 | 128x256 | 1.00 |
| r_lowk | 14 | 8192×8192×1024 | **3393.5** | 3614.7 | 4738.1 | 6144.0 | 128x256 | 1.00 |
| r_mm | 8 | 1000×64×50000 | 659.8 | 553.5 | 520.8 | **517.8** | 64x64 | 0.78 |
| r_mm | 14 | 1000×64×50000 | 905.1 | 897.1 | **877.6** | 878.2 | 128x128 | 0.99 |
| r_skinny | 8 | 16384×512×4096 | **925.7** | 1001.5 | 1398.8 | 1859.6 | 128x256 | 1.00 |
| r_skinny | 14 | 16384×512×4096 | **1607.7** | 1740.8 | 2413.6 | 3157.0 | 128x256 | 1.00 |
| s128 | 8 | 128×128×128 | 5.0 | 3.8 | 2.7 | **2.5** | 64x64 | 0.50 |
| s128 | 14 | 128×128×128 | 5.6 | 4.3 | 3.0 | **2.9** | 64x64 | 0.52 |
| s256 | 8 | 256×256×256 | 7.5 | 5.1 | **3.6** | 3.7 | 64x64 | 0.49 |
| s256 | 14 | 256×256×256 | 7.5 | 5.2 | **4.0** | 4.3 | 64x128 | 0.53 |
| s512 | 8 | 512×512×512 | 9.9 | **6.9** | 7.8 | 10.1 | 128x128 | 0.70 |
| s512 | 14 | 512×512×512 | **11.6** | 12.3 | 13.6 | 17.1 | 128x256 | 1.00 |
| s_batch | 8 | 64×64×64 b=256 | 48.3 | 24.7 | 15.2 | **11.4** | 64x64 | 0.24 |
| s_batch | 14 | 64×64×64 b=256 | 85.0 | 43.0 | 26.4 | **19.2** | 64x64 | 0.23 |
| s_cc16 | 8 | 1536×1536×1536 | 107.1 | **105.2** | 138.7 | 183.9 | 128x256 | 1.00 |
| s_cc16 | 14 | 1536×1536×1536 | **183.0** | 188.3 | 256.5 | 342.6 | 128x256 | 1.00 |
| s_cc8 | 8 | 512×512×512 | 9.8 | **6.8** | 7.8 | 10.0 | 128x128 | 0.69 |
| s_cc8 | 14 | 512×512×512 | **10.4** | 11.1 | 12.3 | 15.6 | 128x256 | 1.00 |
| s_lowk | 8 | 2048×2048×64 | 26.5 | **24.4** | 27.4 | 35.9 | 128x256 | 1.00 |
| s_lowk | 14 | 2048×2048×64 | 51.5 | 52.4 | **49.4** | 63.5 | 128x256 | 1.00 |
| sq1024 | 8 | 1024×1024×1024 | **31.8** | 32.7 | 41.9 | 56.9 | 128x256 | 1.00 |
| sq1024 | 14 | 1024×1024×1024 | 60.3 | **55.8** | 76.1 | 99.2 | 128x256 | 1.00 |
| sq2048 | 8 | 2048×2048×2048 | **240.6** | 246.8 | 346.1 | 471.0 | 128x256 | 1.00 |
| sq2048 | 14 | 2048×2048×2048 | **415.7** | 437.2 | 603.1 | 812.0 | 128x256 | 1.00 |
| sq4096 | 8 | 4096×4096×4096 | **1757.2** | 1864.7 | 2684.9 | 3592.2 | 128x256 | 1.00 |
| sq4096 | 14 | 4096×4096×4096 | **3036.2** | 3272.7 | 4771.8 | 6492.2 | 128x256 | 1.00 |
| t_df | 8 | 4096×4096×2048 | **903.2** | 978.9 | 1317.9 | 1727.5 | 128x256 | 1.00 |
| t_df | 14 | 4096×4096×2048 | **1583.1** | 1695.7 | 2312.2 | 3036.2 | 128x256 | 1.00 |
| t_ikl | 8 | 1024×1024×1024 b=16 | **516.1** | 556.0 | 712.7 | 948.2 | 128x256 | 1.00 |
| t_ikl | 14 | 1024×1024×1024 b=16 | **895.0** | 965.6 | 1244.2 | 1624.1 | 128x256 | 1.00 |
| x1280_1k | 8 | 1280×1280×1024 | 65.9 | **59.3** | 71.5 | 95.1 | 128x256 | 1.00 |
| x1280_1k | 14 | 1280×1280×1024 | **91.3** | 92.7 | 116.5 | 150.7 | 128x256 | 1.00 |
| x128_16k | 8 | 128×128×16384 | 211.3 | 107.4 | **54.9** | 54.9 | 64x64 | 0.26 |
| x128_16k | 14 | 128×128×16384 | 190.3 | 98.0 | 51.6 | **51.0** | 64x64 | 0.27 |
| x1536_512 | 8 | 1536×1536×512 | 49.2 | **44.7** | 55.2 | 73.1 | 128x256 | 1.00 |
| x1536_512 | 14 | 1536×1536×512 | **70.5** | 72.4 | 94.7 | 119.7 | 128x256 | 1.00 |
| x16k_256 | 8 | 16384×256×1024 | **191.9** | 192.1 | 199.1 | 241.6 | 128x256 | 1.00 |
| x16k_256 | 14 | 16384×256×1024 | 338.2 | **337.5** | 344.6 | 427.0 | 128x256 | 1.00 |
| x1k_16k | 8 | 1024×1024×16384 | **470.0** | 506.9 | 699.4 | 947.2 | 128x256 | 1.00 |
| x1k_16k | 14 | 1024×1024×16384 | 892.9 | **869.4** | 1225.7 | 1629.2 | 128x256 | 1.00 |
| x1k_256 | 8 | 1024×1024×256 | 15.4 | **14.8** | 16.4 | 20.8 | 128x256 | 1.00 |
| x1k_256 | 14 | 1024×1024×256 | 24.1 | **20.4** | 24.1 | 30.9 | 128x128 | 0.85 |
| x1k_2k | 8 | 1024×1024×2048 | **55.5** | 62.5 | 83.0 | 111.1 | 128x256 | 1.00 |
| x1k_2k | 14 | 1024×1024×2048 | 113.9 | **108.2** | 147.3 | 198.4 | 128x256 | 1.00 |
| x1k_4k | 8 | 1024×1024×4096 | **114.7** | 129.0 | 175.1 | 226.3 | 128x256 | 1.00 |
| x1k_4k | 14 | 1024×1024×4096 | 231.4 | **225.3** | 312.3 | 399.4 | 128x256 | 1.00 |
| x256_64k | 8 | 256×256×65536 | 834.9 | 485.0 | 377.1 | **358.4** | 64x64 | 0.43 |
| x256_64k | 14 | 256×256×65536 | 858.3 | 640.4 | 599.5 | **580.2** | 64x128 | 0.70 |
| x256_8k | 8 | 256×256×8192 | 97.5 | 50.1 | **26.4** | 29.3 | 64x64 | 0.30 |
| x256_8k | 14 | 256×256×8192 | 97.8 | 52.2 | **38.9** | 49.3 | 64x128 | 0.40 |
| x2k_192 | 8 | 2048×192×8192 | **199.6** | 202.6 | 201.2 | 202.8 | 128x256 | 1.00 |
| x2k_192 | 14 | 2048×192×8192 | 343.1 | **338.3** | 344.1 | 350.6 | 128x256 | 1.00 |
| x2k_1k | 8 | 2048×2048×1024 | **133.1** | 134.1 | 178.2 | 239.6 | 128x256 | 1.00 |
| x2k_1k | 14 | 2048×2048×1024 | **230.4** | 231.4 | 308.2 | 414.7 | 128x256 | 1.00 |
| x2k_256 | 8 | 2048×2048×256 | 45.8 | **43.5** | 50.4 | 65.7 | 128x256 | 1.00 |
| x2k_256 | 14 | 2048×2048×256 | **80.0** | 80.2 | 97.6 | 123.4 | 128x256 | 1.00 |
| x384 | 8 | 384×384×384 | 8.2 | **5.5** | 5.7 | 6.3 | 64x64 | 0.77 |
| x384 | 14 | 384×384×384 | 8.5 | **5.9** | 6.8 | 8.4 | 64x128 | 0.80 |
| x3k_1k | 8 | 3072×3072×1024 | **290.8** | 292.9 | 391.2 | 529.4 | 128x256 | 1.00 |
| x3k_1k | 14 | 3072×3072×1024 | **513.0** | 530.4 | 681.0 | 917.5 | 128x256 | 1.00 |
| x4k_128 | 8 | 4096×128×4096 | 182.6 | 179.6 | **175.9** | 177.2 | 128x128 | 0.98 |
| x4k_128 | 14 | 4096×128×4096 | 328.1 | 307.4 | **305.8** | 309.5 | 128x128 | 0.94 |
| x512_32k | 8 | 512×512×32768 | 438.4 | **365.1** | 381.9 | 484.4 | 128x128 | 0.83 |
| x512_32k | 14 | 512×512×32768 | 612.0 | **580.3** | 657.7 | 812.1 | 128x256 | 1.00 |
| x512_4k | 8 | 512×512×4096 | 51.0 | **29.7** | 41.7 | 56.1 | 128x128 | 0.58 |
| x512_4k | 14 | 512×512×4096 | **57.7** | 58.9 | 80.2 | 106.1 | 128x256 | 1.00 |
| x640_1k | 8 | 640×640×1024 | **16.6** | 17.1 | 20.2 | 26.3 | 128x256 | 1.00 |
| x640_1k | 14 | 640×640×1024 | 30.8 | **24.9** | 30.9 | 39.2 | 128x128 | 0.81 |
| x64_8k | 8 | 64×8192×4096 | 338.5 | 337.5 | 336.3 | **336.3** | 64x128 | 0.99 |
| x64_8k | 14 | 64×8192×4096 | 571.2 | **567.9** | 568.5 | 569.0 | 64x128 | 1.00 |
| x768 | 8 | 768×768×768 | 26.6 | **21.5** | 23.0 | 28.5 | 128x128 | 0.81 |
| x768 | 14 | 768×768×768 | **25.7** | 26.5 | 32.7 | 44.2 | 128x256 | 1.00 |
| x8k_512 | 8 | 8192×512×512 | **71.3** | 77.2 | 95.2 | 119.3 | 128x256 | 1.00 |
| x8k_512 | 14 | 8192×512×512 | **135.7** | 137.1 | 167.1 | 219.6 | 128x256 | 1.00 |
| x8k_64 | 8 | 8192×64×4096 | 343.2 | 337.9 | **335.5** | 338.4 | 64x64 | 0.99 |
| x8k_64 | 14 | 8192×64×4096 | 580.8 | 568.7 | **566.8** | 569.5 | 64x64 | 0.98 |

- **Large problems: 128×256 stays best** (sq2048–sq4096, r_lowk, r_bigk, r_skinny, t_df,
  t_ikl, x2k_1k, x3k_1k): 128×128 is 0–17 % slower (L2 traffic, as in NOTES-A2), 64×128
  34–81 % and 64×64 80–165 % slower (32×64 warp tiles: more shared-memory fragment loads per
  MMA, and 2.7× the operand traffic per MAC at 64×64).
- **Few tiles: smaller tiles win by up to 4×.** s128 / s256 (0.5×), s_batch (64×64×64,
  b = 256: 0.23×, the 128×256 tile is 8× padding), x128_16k / x256_8k / x256_64k (2–4
  128×256 tiles per modulus: 0.26–0.70×), b16_512x64 and r_mm (n̂ = 64: 0.49× / 0.79×).
- **Moderate sizes (1–4 waves of 128×256): 128×128 often 5–15 % faster** (x384, s512, x768,
  x512_4k, x512_32k, x640_1k, x1k_256, sq1024 at N = 14): two CTAs per SM overlap one
  tile's epilogue with the other's main loop, which beats its extra L2 traffic while the
  problem is small.
- **Memory-bound skinny views** (x8k_64, x4k_128, x2k_192, x64_8k, x16k_256): all four
  within 1–7 %.

**Model.** For configuration c (tile tm×tn, o CTAs/SM), the busiest SM gets
j = ⌈tiles/128⌉ tiles, run as ⌊j/o⌋ rounds of o concurrent CTAs and one round of the rest; a
round of q CTAs costs `round_ns + q·tm·tn·k̂·mac_fs[q]`. Linear least squares in the
relative error on the 84 cases gave (DESIGN §4 table) round_ns 2716 / 2550 / 1424 / 1333
and mac_fs 418 | 453, 456 | 505, 612 | 944, 1088, 802 (128×256 | 128×128 | 64×128 |
64×64; mac_fs[q] by concurrency). A first model with one round size (q = min(j, o) for
every round) misjudged the odd last round of the 2- and 3-CTA configurations (worst pick
1.32× the best); splitting the last round brought it to 1.15×. A per-tile term on top
fitted negative constants (over-fitting) and was dropped. The margin (another tile must
beat 128×256's estimate by 5 %) costs 0.1 % of the total here and protects against model
error on shapes not measured.

**Result on the fitted cases:** K4 total 38.04 ms with the rule against 39.67 ms always
128×256 (−4.1 %) and 37.88 ms per-case best (+0.4 %); the rule never picks a configuration
slower than 128×256; worst pick 1.15× the best (x384: 64×64 or 64×128 instead of 128×128,
still 0.77–0.80× of 128×256). Where it leaves time (5–11 %) is the 128×128 sweet spot at
1–4 waves (s_lowk, sq1024 N = 14, x1280_1k, x1536_512, x2k_256), which the model scores
within the margin.

## 4. Beyond K4: K1 on long rows (task 4) and KF for k̂ ≤ 1024 (task 5)

### K1 long rows (r_mm)

K1 A of r_mm read its operand twice from DRAM (§1). Variants, each in its own quiet-gated
process, interleaved, N = 8, two rounds (K1 A ms, % of its one-read bound; whole call ms):

| case (rows × row bytes) | before (fa77ec0) | 1024/SM, forward | 2×512/SM, backward | 1024/SM, backward |
|---|---|---|---|---|
| r_mm (1000 × 400 KB) | 0.865 (46 %), 2.55–2.63 | **0.584–0.607 (65–68 %), 2.18–2.24** | 0.675–0.690, 2.27–2.32 | 0.637 (62 %), 2.27 |
| x1k_64k (1024 × 512 KB) | 1.148 (46 %), 3.46 | **0.817 (65 %), 2.95** | 0.950, 3.01 | 0.818–0.845, 2.89–2.98 |
| x300_100k (300 × 800 KB) | 0.612–0.626 (39 %), 2.57–2.74 | 0.547–0.552, 1.82–1.87 | 0.555–0.569, 1.82–1.88 | **0.456 (52 %), 1.76** |
| r_bigk (2048 × 256 KB) | 0.761–0.777 (70 %) | (2×512 forward = before) | 0.792–0.796 | (n/a: 2×512) |
| r_skinny (16384 × 32 KB) | 0.642 (83 %) | = | = | = |

- The A4 variant (two 512-thread blocks per SM) kept 2 × 128 rows in flight whatever their
  length; above the L2 it re-reads from DRAM. Now it is used only while 2 × SMs rows fit a
  64 MiB budget (r_bigk's 256 × 256 KB = 64 MiB still do), else one 1024-thread block per
  SM (r_mm: 128 × 400 KB = 51 MB in flight; ncu: L2 hit 34 → 73 %).
- Walking pass 2 backwards helps only when even those rows exceed the budget (x300_100k:
  102 MB in flight, 0.55 → 0.46 ms) and costs 3–8 % when they fit (r_mm, r_bigk), so it is
  a launch choice: backwards iff rows in flight × row bytes > 64 MiB.
- Same exact integers in every variant (max, u128 sum); `contract_bigk` covers all three
  launch shapes (170 × 32768: 2 × 512; 130 × 40000: 1024 forward; 60 × 150000: 1024
  backward).
- r_mm's K1 A is now at 65–68 % of its bound (was 46 %); what is left is the dependent
  gather (`X[v_k[h]]`) of a 1024-thread block per SM; r_mm's B side (64 rows: 64 blocks) is
  small.

### KF for k̂ ≤ 1024

NOTES-A1 left sq1024-class shapes on the unfused sequence (K0 + 2 K1 + 2 packs). With
k̂ ≤ 1024 fused (DESIGN v1.5 §4; same outputs; Pipeline's `Fused` is k̂-independent),
interleaved A/B against fa77ec0 (whole call µs; this build also has the first K4 rule):

| case | N | before | after | after / before | KF ms vs K0+K1+K2+K3 before |
|---|---|---|---|---|---|
| sq1024 | 8 | 100.4 | 89.1 | 0.89 | 0.041 vs 0.074 |
| sq1024 | 14 | 154.6 | 155.1 | 1.00 | 0.065 vs 0.079 |
| x768 | 8 | 91.1 | 71.7 | 0.79 | 0.039 vs 0.061 |
| x768 | 14 | 99.3 | 95.2 | 0.96 | 0.051 vs 0.063 |
| x640_1k | 8 | 86.0 | 63.5 | 0.74 | 0.040 vs 0.062 |
| x640_1k | 14 | 120.8 | 95.2 | 0.79 | 0.052 vs 0.081 |
| t_ikl (1024³, b = 16) | 8 | 1861 | 1678 | 0.90 | 0.874 vs 1.062 |
| t_ikl | 14 | 2583 | 2381 | 0.92 | 1.047 vs 1.258 |
| r_lowk (8192²×1024) | 8 | 3604 | 3574 | 0.99 | 0.415 vs 0.514 |
| r_lowk | 14 | 5651 | 5722 | 1.01 | 0.492 vs 0.595 |
| x2k_1k | 8 | 294 | 305 | 1.04 | 0.121 vs 0.120 |
| x2k_1k | 14 | 471 | 470 | 1.00 | 0.139 vs 0.152 |

KF does the work of K0 + K1 + both packs in 55–85 % of their time from k̂ = 640 to 1024;
at x2k_1k it is even, and the whole-call differences there and at r_lowk are within the
±3 % noise of this power-capped card (sq4096, untouched, moved +2 % in the same run).
