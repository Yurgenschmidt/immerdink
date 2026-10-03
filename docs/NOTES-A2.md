# A2 notes: K4 epilogue / main-loop overlap

ROADMAP A2: K4 (the batched int8 GEMMs with the mod-p → u8 epilogue) ran at 60–66 % of its
bound at k ≈ 1024 (NOTES-M4). This file has the measurements first (before any change), then
what changed and the before/after numbers. Everything is kernel-internal: same launches
(`src/host/k4_launches.hpp`), same §3.5 arithmetic, bit-identical results (all 29 tests,
incl. `cutlass_mod_epilogue`, `mod_exhaustive_cpu`, `device_constants`, `host_vs_plan`); no
cert, spec or Apalache change.

Tooling. Every profile and timing went through `tools/run_bench.sh` (quiet gate, memcap)
inside the shared GPU lock. ncu 2026.3.1, `--set full`, the first K4 launch of a call
(`imd`, N = 14, g = N; each case is a single K-chunk, so one store-mod launch), and the
cuBLASLt int8 TN GEMM (int32 out, one modulus) of the bench's `int8` method. ncu locks the
clocks (2.5 GHz SM here), so ncu durations are comparable with each other, not with the
event timings. The int8 tensor-pipe utilization `sm__pipe_tensor_cycles_active` at this
clock is the fraction of the 660 TOPS bound: duration × tensor % ≈ the §4 math bound
(r_lowk: 2.92 ms vs 2.91). The main-loop / epilogue split is from warp-state sampling per
SASS instruction (Source Counters): main loop = the IMMA loop body, prologue = before it,
epilogue = after it (for K4Gemm this includes issuing the next tile's prologue).

## 1. Measurements before any change (dev 22496a8 + sweep file, `c63117a`)

### Whole kernel (ncu, N = 14)

| case (m×n×k, b) | K4 ncu ms | tensor pipe | L2 | cuBLASLt int8 (1 modulus) tensor pipe |
|---|---|---|---|---|
| r_lowk 8192²×1024 | 4.80 | 61.3 % | 53.7 % | 53.7 % (0.39 ms; int32 out) |
| t_df 4096²×2048 | 1.93 | 75.7 % | 61.9 % | 73.9 % |
| t_ikl 1024³, b = 16 | 1.21 | 60.5 % | 52.9 % | 57.6 % |
| sq8192 | 25.2 | 92.3 % | 70.8 % | 93.2 % |
| r_bigk 2048²×32768 | 6.09 | 97.0 % | 73.5 % | 56.8 % (128 tiles: one partial wave) |

- cuBLASLt picks `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3` — the same CUTLASS 2.x
  main loop at 256×128 — and is *not* a better roofline at moderate k: its int32 output
  (4 B per element) costs it more than our u8 epilogue. At large k both sit at 92–97 %.
- Occupancy: 1 CTA (8 warps) per SM, limited by both registers (226 / thread) and shared
  memory (72 KB). Achieved 16.6 % of 48 warps = 2 warps per scheduler.
- Stalls (whole kernel, r_lowk): math-pipe throttle 45 %, wait 23 %, short scoreboard 11 %,
  selected 9 %, barrier 4 %.

### Where the time goes (warp-state samples by region)

| case | prologue | main loop | epilogue |
|---|---|---|---|
| r_lowk (k = 1024) | 6.4 % | 61.9 % | **31.7 %** |
| t_df (k = 2048) | 4.1 % | 76.3 % | 19.6 % |
| t_ikl (k = 1024) | 7.3 % | 61.4 % | **31.3 %** |
| sq8192 | 1.2 % | 92.7 % | 6.1 % |
| r_bigk | 0.4 % | 97.9 % | 1.8 % |

- **Main loop**: stalls are 72 % math-pipe throttle (tensor pipe saturated), 18 % wait. It is
  at the hardware rate; nothing to gain there.
- **Epilogue** (r_lowk: 39 % of all executed instructions): stalls wait 36 %, short
  scoreboard 33 % (shared-memory loads), barrier 8 %. Two causes:
  1. *The mod-p reduction re-read its constants per element.* `VisitorModP` called
     `mod_p(v, l)`, which reads `p`, `bias`, `magic` from the `__constant__` table; the
     compiler re-issued two indexed `LDC`s and the `p == 256` branch for every element
     (2 LDC + 2 BRA + 3 IMAD.WIDE + carry adds ≈ 12 SASS instructions per element; 14
     per element in all, including the staging).
  2. *CUTLASS's epilogue structure.* Accumulators go through shared memory in 8 row
     groups with two barriers each, and at threadblock N = 256 CUTLASS uses the
     non-"mixed" iterators for 8-bit outputs ("still has bank conflicts", their comment):
     88 M of 147 M epilogue shared-memory wavefronts are bank-conflict excess (2.5×).
- **Prologue**: the first two cp.async stages of every tile are exposed (long-scoreboard
  stalls), because the next CTA on an SM starts only after this one exits.
- At k ≥ 8192 the epilogue is ≤ 6 % and K4 is already at 92–97 %.

### Configuration probes (ncu, same code with fix 1 below applied, "e1")

| variant | r_lowk ms (tensor) | t_df ms (tensor) | note |
|---|---|---|---|
| e1: 128×256 s3, CUTLASS EVT | 4.12 (71.0 %) | 1.78 (82.8 %) | reference |
| `.satfinite` removed (own MMA specialization) | 4.11 (71.5 %) | 1.77 (82.9 %) | **saturation is free** |
| EVT epilogue stages 2 | 4.03 (72.4 %) | 1.75 (83.6 %) | +2 % |
| 256×128 s3 (conflict-free "mixed" iterators) | 3.84 (76.2 %) | 1.70 (86.4 %) | +7 % / +4 % |
| 128×128 s3, 2 CTAs/SM | 3.57 (86.5 %) | 1.70 (91.3 %) | overlap via the 2nd CTA, but L2 at 93 % |

- `OpMultiplyAddSaturate` costs nothing measurable (CUTLASS has no non-saturating sm80
  int8 MMA; a bench-only specialization without `.satfinite`, `-DIMD_K4_NOSAT_MMA`, gives
  the same time). It stays (DESIGN §2: K_c is sized for `.satfinite`).
- Two CTAs per SM overlap one CTA's epilogue with the other's main loop — the effect A2
  wants — but at 128×128 the A/B panel traffic per MAC is 1.33× that of 128×256 and L2
  throughput reaches 91–93 %: the 2-CTA configuration becomes L2-bound. Two 128×256 CTAs
  cannot be co-resident (each needs 128 accumulator registers per thread for 256 threads:
  the whole register file).
- CUTLASS 3.x warp-specialized epilogues (TMA, `wgmma`, producer/consumer warp groups) are
  sm_90+ and are not reachable on sm_89; a hand-built ping-pong (two consumer warp groups
  alternating main loop and epilogue) needs two accumulator sets, i.e. the same register
  file as two co-resident CTAs (above), with the same L2 traffic.

## 2. What changed

1. **Constants once per CTA** (`635532f`, `kernels/mod_epilogue.cuh`, `device/modp.hpp`):
   the callbacks load `p`, `bias`, `magic` in their constructor; the `p = 256` test is
   uniform per fragment; `umulhi64_u32` computes the same umulhi64(magic, n) from 32-bit
   words. r_lowk 4.80 → 4.12 ms (ncu).
2. **K4Gemm** (`54f6902`, `kernels/k4_gemm.cuh`, `gemm_k4.cu`): our own kernel around
   CUTLASS's `MmaMultistage` main loop (unchanged: same tile 128×256×64, 3 stages, 8 warps of
   64×64, same int8 `.satfinite` MMA):
   - **persistent CTAs** (grid = SMs × occupancy; tiles in the M4 swizzle-8 order, so a
     wave still covers a compact 16 × 8 block of tiles and its A/B panels stay in L2); a CTA
     issues the cp.async prologue of its *next* tile before running the epilogue of the
     current one, so that latency overlaps the epilogue;
   - **register-only epilogue**: each thread reduces its own accumulators (PTX m16n8k32
     layout: rows g, g + 8 of each 16-row tile, columns 2t, 2t + 1 of each 8-column tile),
     packs the bytes, and a 4 × 4 transpose of 32-bit words inside each lane quad (two
     `__shfl_xor_sync` butterfly stages) gives each thread 16 consecutive bytes of a row:
     one 128-bit store per row and thread. No shared memory, no barriers, so the next
     tile's prologue can use all of it. Add-mod chunks load the old 16 bytes per row,
     run the same shuffle backwards and add them to the accumulators before the reduction
     (the DESIGN §3.5 `(W + acc) mod p`, unchanged).
3. **Epilogue instruction diet** (`3e419b2`): the first K4Gemm still had ~1900 epilogue
   instructions per warp (a per-element `p == 256 ? : ` select compiled to two branches per
   element, `−p` re-negated per element, the 64-bit high product with zero moves, four
   64-bit divisions per tile). Now: one `p = 256` branch per tile, `−p` hoisted
   (`mod_p_barrett_neg`), `umulhi(al, n)` + one `mad.lo.cc`/`madc.hi` carry chain, and
   `cutlass::FastDivmod` for the tile → (z, m, n, modulus) mapping when the launch's indices
   fit 31 bits (64-bit division otherwise). ~1300 epilogue instructions per warp, ~5 integer
   instructions per element for the reduction.

Probed and not adopted (ncu, K4Gemm): 4 stages (96 KB, possible now that the epilogue
needs no shared memory) 1–3 % slower; 256×128 within noise; 128×128 with 2 CTAs/SM
(r_lowk 3.66 ms, L2 89 %) slower than 128×256 with 1 CTA/SM.

### Exactness

- The reduction computes the same values: `umulhi64_u32(a, n) = floor(a n / 2^64)` for
  n < 2^32 (proof in the comment: dropping the fraction of `al n / 2^32` cannot cross a
  multiple of 2^64), so cert_barrett's statement about umulhi64 applies unchanged. The
  device path (PTX carry chain) is checked over **every** reachable input of every modulus
  by `device_constants` (through `mod_p` → `mod_p_epilogue` → `mod_p_barrett` →
  `mod_p_barrett_neg`), the host path by `mod_exhaustive_cpu`.
- The kernel: `cutlass_mod_epilogue` now runs each case (all 20 moduli with b = 2 and 3
  chunks; a middle group with 2 × 2 partial tiles; the K_c' extremes `v_min`/`v_max`)
  through the CUTLASS EVT kernel *and* K4Gemm, the latter twice: one CTA per SM, and only 3
  persistent CTAs with the batch count split into launches of 2 (every CTA loops over many
  tiles, every launch has a nonzero `batch0`). All bit-exact. The contract suites
  (oracle-exact, incl. the forced chunk / batch / n-hat splits of `contract_stages`) pass.
- Plan bytes: the parameter-map entries are 280 bytes now (`host/analysis.hpp`
  `kK4StoreEntryBytes` / `kK4AddEntryBytes`, static_asserted in `gemm_k4.cu`);
  `host_vs_plan` passes.

## 3. Results

### ncu (same clocks), N = 14

| case | before | after | tensor pipe before → after | epilogue share of samples |
|---|---|---|---|---|
| r_lowk | 4.80 ms | 3.50 ms (−27 %) | 61.3 → 83.7 % | 31.7 → 15.1 % |
| t_df | 1.93 ms | 1.62 ms (−16 %) | 75.7 → 90.5 % | 19.6 → 8.5 % |
| t_ikl | 1.21 ms | 0.92 ms (−24 %) | 60.5 → 83.3 % | 31.3 → 15.2 % |
| sq8192 | 25.2 ms | 24.3 ms (−4 %) | 92.3 → 97.1 % | 6.1 → 2.4 % |
| r_bigk | 6.09 ms | 6.06 ms | 97.0 → 98.6 % | 1.8 → 0.8 % |

What is left at k = 1024 is the epilogue's own integer work (~8 instructions per element:
reduction, packing, quad transpose) executed by the same warps that run the MMAs; with one
128×256 CTA per SM nothing else can issue tensor instructions meanwhile. Hiding it would
need a second accumulator set (a second CTA or warp group), which at this tile size does not
fit the register file and at 128×128 hits L2 bandwidth (§1).

### Timings (CUDA events, `bench/sweep_a2.txt`, quiet-gated)

Before (`c63117a`, `build/bench-base`) and after (`3e419b2`) ran **interleaved per case**
in one session (`bench/results/a2-before-ab-c63117a.jsonl`, `a2-after-3e419b2.jsonl`; the
after file also has GEMMul8 and the cuBLASLt int8 GEMM of the same session). The 4090 runs
these at its power cap, so differences below ~3 % are noise. An earlier, non-interleaved
before run is `a2-before-c63117a.jsonl` (same picture; K4 63.7 / 71.3 / 64.5 % of bound on
r_lowk / t_df / t_ikl at N = 14).

### K4 phase: ms and % of its §4 bound (event marks, median of 5 profiled calls)

| case | N | before ms | before % | after ms | after % | K4 speedup |
|---|---|---|---|---|---|---|
| r_lowk | 8 | 2.690 | 62 % | 1.944 | 86 % | 1.38× |
| r_lowk | 14 | 4.546 | 64 % | 3.548 | 82 % | 1.28× |
| t_df | 8 | 1.108 | 75 % | 0.930 | 90 % | 1.19× |
| t_df | 14 | 1.942 | 75 % | 1.626 | 90 % | 1.19× |
| t_ikl | 8 | 0.691 | 60 % | 0.521 | 80 % | 1.33× |
| t_ikl | 14 | 1.208 | 60 % | 0.903 | 81 % | 1.34× |
| t_cc | 8 | 1.952 | 85 % | 1.754 | 95 % | 1.11× |
| t_cc | 14 | 3.287 | 89 % | 3.102 | 94 % | 1.06× |
| t_permout | 8 | 1.952 | 85 % | 1.738 | 96 % | 1.12× |
| t_permout | 14 | 3.377 | 86 % | 3.066 | 95 % | 1.10× |
| sq4096 | 8 | 1.953 | 85 % | 1.751 | 95 % | 1.12× |
| sq4096 | 14 | 3.378 | 86 % | 3.071 | 95 % | 1.10× |
| sq8192 | 8 | 14.275 | 93 % | 13.786 | 97 % | 1.04× |
| sq8192 | 14 | 25.794 | 90 % | 24.739 | 94 % | 1.04× |
| r_bigk | 8 | 3.438 | 97 % | 3.406 | 98 % | 1.01× |
| r_bigk | 14 | 6.040 | 96 % | 5.957 | 98 % | 1.01× |
| s_cc16 | 8 | 0.141 | 62 % | 0.119 | 74 % | 1.19× |
| s_cc16 | 14 | 0.231 | 67 % | 0.186 | 83 % | 1.24× |

### Whole call: TFLOPS-eq (median), immerdink before / after vs GEMMul8 (after session)

| case | N | imd before | imd after | speedup | GEMMul8 | after / GEMMul8 (before) |
|---|---|---|---|---|---|---|
| r_lowk | 8 | 28.01 | 33.52 | 1.20× | 20.39 | 1.64 (1.37) |
| r_lowk | 14 | 19.00 | 22.55 | 1.19× | 11.46 | 1.97 (1.66) |
| t_df | 8 | 33.34 | 36.57 | 1.10× | 29.26 | 1.25 (1.14) |
| t_df | 14 | 22.50 | 25.16 | 1.12× | 16.69 | 1.51 (1.35) |
| t_ikl | 8 | 16.03 | 17.84 | 1.11× | 18.13 | 0.98 (0.88) |
| t_ikl | 14 | 11.32 | 12.85 | 1.14× | 11.56 | 1.11 (0.98) |
| t_cc | 8 | 40.49 | 43.49 | 1.07× | 39.06 | 1.11 (1.04) |
| t_cc | 14 | 27.09 | 29.03 | 1.07× | 22.69 | 1.28 (1.19) |
| t_permout | 8 | 38.97 | 42.27 | 1.08× | 37.17 | 1.14 (1.05) |
| t_permout | 14 | 26.18 | 28.43 | 1.09× | 21.99 | 1.29 (1.19) |
| sq4096 | 8 | 40.54 | 43.18 | 1.07× | 39.10 | 1.10 (1.04) |
| sq4096 | 14 | 26.90 | 29.18 | 1.08× | 22.69 | 1.29 (1.19) |
| sq8192 | 8 | 55.30 | 56.29 | 1.02× | 52.68 | 1.07 (1.05) |
| sq8192 | 14 | 33.84 | 34.84 | 1.03× | 30.89 | 1.13 (1.10) |
| r_bigk | 8 | 33.84 | 34.44 | 1.02× | 37.88 | 0.91 (0.89) |
| r_bigk | 14 | 23.96 | 24.15 | 1.01× | 24.77 | 0.97 (0.97) |
| s_cc16 | 8 | 19.88 | 21.40 | 1.08× | 25.55 | 0.84 (0.78) |
| s_cc16 | 14 | 14.27 | 15.70 | 1.10× | 14.78 | 1.06 (0.97) |

### cuBLASLt int8 (one modulus, int32 out), after session

| case | ms | TOPS | × N=14 ms | imd K4 N=14 ms |
|---|---|---|---|---|
| r_lowk | 0.405 | 339 | 5.68 | 3.55 |
| t_df | 0.155 | 445 | 2.16 | 1.63 |
| t_ikl | 0.118 | 292 | 1.65 | 0.90 |
| t_cc | 0.258 | 533 | 3.61 | 3.10 |
| t_permout | 0.262 | 525 | 3.66 | 3.07 |
| sq4096 | 0.257 | 535 | 3.60 | 3.07 |
| sq8192 | 1.798 | 612 | 25.17 | 24.74 |
| r_bigk | 0.743 | 370 | 10.41 | 5.96 |
| s_cc16 | 0.035 | 208 | 0.49 | 0.19 |

- **K4 at moderate k**: 60–64 % → 80–86 % of the bound at k = 1024 (r_lowk, t_ikl), 75 →
  90 % at k = 2048 (t_df), 85–89 → 94–96 % at k = 4096 (t_cc, t_permout, sq4096). Large k
  (sq8192, r_bigk) moves 1–4 %. K4 is now faster than N × the cuBLASLt int8 GEMM on every
  case (that one writes int32), by 1.2–2× at moderate k.
- **Whole call**: 1.07–1.20× faster at moderate k, 1.01–1.03× at large k. Against GEMMul8
  (same session): r_lowk 1.64× / 1.97× (N = 8 / 14), t_df 1.25× / 1.51×, the 4096-class
  cases 1.10–1.14× / 1.28–1.29×, t_ikl 0.98× / 1.11× (was 0.88× / 0.98×), s_cc16 0.84× /
  1.06× (was 0.78× / 0.97×). r_bigk stays at 0.91× / 0.97×: its K4 was already at 96–98 %;
  the gap there is K1 / the pack on 256 KB rows (A4) and tile selection (A3).
- What remains of K4's gap at k = 1024 is the epilogue's integer work itself (see the ncu
  results above).
