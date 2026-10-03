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
