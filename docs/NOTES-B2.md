# B2 notes: the performance model (DESIGN v1.8 §6c; v1.10 fidelity follow-up in §6)

ROADMAP B2. A CUDA-free model in libimmerdink_host (host API version 6): `imd_perf_predict`
(per-phase and total time, TFLOPS-eq, MEMORY_MODEL §4 bounds), `imd_perf_predict_pack`,
`imd_perf_rank` (N × g × vec mode × packed operands, ranked by predicted time inside a memory
budget, with exact footprints), and the calibration (`imd_perf_calib_*`, a `key = value` text
file; the RTX 4090 one is embedded). Design reviewed 2026-10-05 (roofline K4 accepted; eps
path as a data input; effective L2 and the fitter's seed calibration). All numbers below are
immerdink's own, on the RTX 4090 at stock clocks (not locked; SM 2460–2820 MHz after calls,
450 W cap), through the quiet gate and the shared GPU lock.

## 1. Model in one paragraph

Each phase's time is linear in calibrated per-kernel parameters over features computed exactly
from the plan's lowering (no Python copy of a formula): ns per launch, per DRAM byte, per
residue on the eps path, per L2 re-read byte, per row; K4 is a roofline, max(A3 round model with
a partial-K-tile term, DRAM bytes beyond an effective L2); K5 per byte and per CRT limb
operation; a signed per-call term maps the sum of the event-marked phases onto the timed call
(launch gaps the phases miss, minus the marks' own cost: −0.48 µs per launch, +0.19 µs per
call). 12 classes, 40 parameters in use.

## 2. Calibration (bench/sweep_calib.txt, tools/calibrate.sh)

- 44 cases, 269 timed calls with per-phase data (normal data and `--data dominant`), every K4
  configuration forced, N = 4…20, g < N, RESIDENT, packed operands, K-empty and DIRECT.
- GPU time ≈ 12 min of runs (6.2 + 5.0 min across two sessions; quiet-gate waits for other
  work on the shared GPU not counted).
- Fit: per class NNLS in relative error; K4 roofline with a DRAM-rate scan and an
  effective-L2 scan. In-sample: phase medians 4–11 % per class; whole calls median 4.3 %, p90
  11.6 %.
- Fitted values worth knowing (effective rates):

| term | value |
|---|---|
| KL DRAM | 872 GB/s (one read of the operand + group 1's residues) |
| pack (K2/K3, groups 2..) | 923 GB/s |
| KF | 1.16 TB/s (small operands partly in L2) |
| K4 DRAM term, 128×256 / 128×128 / 64×128 | 814 GB/s (one shared rate: the scan made them equal) |
| K4 DRAM term, 64×64 | 603 GB/s (narrower tiles re-read the operands more) |
| K4 effective L2 | 0.40 × 72 MiB (with the nominal 72 MiB the memory-bound g < N points were up to 48 % low) |
| K4 128×256 | 3.97 µs per round + 0.389 ps per MAC (A3 had 2.72 µs / 0.418 ps without launch marks) |
| eps path | +0.39 ps per residue in KL, +1.4 ps in KF, +0.46 ps in K2/K3 |
| launches (with phase marks) | K0 3.3 µs, K5 4.9 µs, KF 6.6 µs, KL 7.3 µs |

- **Review question: shared K4 DRAM rate?** One rate for all four configurations raised the
  64×64 class error from 7.3 % / 19 % (median / p90) to 11.0 % / 34 %. The three larger tiles
  converge on one rate by themselves (814 GB/s), 64×64 does not, so the rates stay per
  configuration.
- **Review question: is op_eps more than N?** It is identified on the same plans with
  different data. With the eps fraction for dominant-entry data (512 / k), KL / KF at N = 15 and
  16 fit within +3 … +8 % mean bias for both data kinds, and KF at N = 20 has +1 % mean bias.
  So no separate register-pressure term was needed: the N = 20 "jump" was the eps path. The
  typical-data default fraction was simulated for the bench generator: a 512-element tile is
  on the path with probability 0.03 / 0.23 / 0.65 / 0.96 at margin D = −2.5 / −2 / −1.5 / −1,
  the same for k = 1024 and 8192; it is a property of the data, so callers can pass their own
  fraction, and `eps_s` gives the exact bracket.
- **Found on the hold-out set: the partial K tile.** K-chunks whose length is not a multiple of
  64 (k̂ = 1008, 5008, …) run 20–40 % slower in K4 than the round model. Of the criteria
  k̂ mod 64 / 128 / 256, mod 64 fits best (hold-out K4 64×128 median 22 → 9.5 %). It became the
  `mac_ktail` feature (one per configuration).

## 3. Hold-out validation (bench/sweep_perf_validate.txt; shapes disjoint from the fit)

114 timed calls: the regression sweep's cases, tensor cases (t_cc, t_df, t_perm, t_ikl,
abcd,cdef->abef), odd sizes, small shapes, N = 8 / 10 / 14 / 15 / 16 / 17, g < N on
memory-bound shapes, RESIDENT, packed operands and dominant-entry data. Relative error of the
predicted total vs the measured timed median:

| measured call | calls | median \|err\| | p90 \|err\| | max \|err\| | target (median / p90) | met |
|---|---|---|---|---|---|---|
| < 100 µs | 29 | 11.7 % | 19.3 % | 22.0 % | 15 % / 30 % | yes |
| 100 µs – 1 ms | 26 | 5.1 % | 17.5 % | 22.7 % | 10 % / 20 % | yes |
| ≥ 1 ms | 59 | 3.2 % | 8.4 % | 20.1 % | 5 % / 15 % | yes |

N = 15…17 only (ehrenfest's range): ≥ 1 ms, 20 calls: median 3.9 %, p90 9.5 % (met). 100 µs – 1 ms,
6 calls: median 13.2 %, p90 22.5 % (missed; KF at N ≥ 15 on 640³-class shapes is predicted
20–30 % low, open). < 100 µs: 1 call, 16.8 %.

Per phase on calls ≥ 1 ms (median / p90): KL A 3.7 / 13.4 %, KL B 7.5 / 22.7 %, K2 5.2 / 15.2 %,
K3 18.7 / 37.0 %, K4 5.4 / 18.0 %, K5 8.5 / 26.0 %. K0 (3–9 µs) is noisy (33 % median) but
negligible in these calls.

Examples (TFLOPS-eq measured / predicted, time measured / predicted):

| call | TFLOPS-eq | time |
|---|---|---|
| sq8192, N = 14 | 37.05 / 37.78 | 29.68 / 29.10 ms |
| sq8192, N = 8 | 62.27 / 63.92 | 17.66 / 17.20 ms |
| t_cc (32 / 128), N = 14 | 32.23 / 31.45 | 4.27 / 4.37 ms |
| r_mm 1000 × 64 × 50000, N = 14 | 2.45 / 2.42 | 2.62 / 2.65 ms |
| 24576 × 384 × 3072, N = 16, g = 7 | 9.61 / 10.04 | 6.03 / 5.78 ms |
| kjbc,kiac->abij (24 / 112), N = 17 | 20.04 / 20.25 | 1.94 / 1.92 ms |
| 3000³, N = 15 | 25.09 / 24.06 | 2.15 / 2.24 ms |
| 5000³, N = 16 | 21.42 / 20.75 | 11.67 / 12.05 ms |
| s512, N = 8 | 10.21 / 8.88 | 26.3 / 30.2 µs |

Worst hold-out calls: 640³ at N = 16 / 17 (−21 / −23 %: KF low), kjbc,kiac->abij (12 / 48) at
N = 8 (−22 %), s512 with g = 4 (+21 / +22 %), 768 × 768 × 100000 at N = 16 (+20 %: long KL rows
on the eps path), ikl,kjl->ijl (1024, l = 16) at N = 8 (−18 %). Before the K-tail term sq5000
and 3000 × 700 × 5000 were −21 / −22 %; now +5 % / −6 %.

## 4. Open

- **Closed (2026-10-05, branch b2-kf-n15): KF at N = 15–17 on mid-size shapes.** More data alone
  did not fix it: on 10 new fit cases (448–960³, non-square, batched, a CC tensor; bench
  generator and dominant-entry data; ≈ 4.5 min of GPU time with 8 new hold-out cases), KF was
  predicted up to 45 % low on the shapes with 512 < k̂ ≤ 1024 at N ≥ 15 (576³: 62 µs measured,
  36 µs predicted) and up to 20 % high below. The cause is KF's launch shape: above k̂ = 512 at
  these N the E = 16 kernel runs (16 values per thread; 288 blocks at 576³, 2 resident per SM),
  and each thread emits g residues for each of its 16 values in sequence, so time follows the
  busiest SM's chain, not the bytes (448³ runs the E = 4 kernel with 4× the blocks: 20 µs). The
  new KF feature `chain` = ⌈blocks / (SMs × resident)⌉ × E × ⌈g / parts⌉ × (1 + f) (DESIGN §6c.1),
  from the launch rules now shared with contract.cu (`kf_lg_tk`, `kf_operand_tiles`,
  `kf_small`, `pack_moduli_parts`), fitted at 23.6 ns per step. Offline, of the resident counts
  (1, 2, 3) × (2, 4, 6) and with or without the eps weight, (2, 4) with the weight gave the best
  hold-out (KF phase at N = 15–17: median / p90 19.6 / 33.6 % → 13.6 / 17.9 %).
- K3 (B packs of groups 2..) still has a 37 % p90 on ≥ 1 ms calls (K2: 15 %). No obvious cheap
  fix: the B side has the zero-rows launch and often few rows (r_mm-like n = 64–384), i.e. a
  small pack grid, which the PACK features do not see; a wave term like KF's would be the
  candidate.
- The K4 partial-K-tile term overpredicts k̂ = 800 (K4 +35 % on 800³ at N = 15 / 16) while it
  fixed k̂ = 1008 / 5008; whether the cost is the predicated residue tile or the 128-byte row
  alignment (800 mod 128 = 32, 1008 mod 128 = 112) is not settled with the data at hand.
- Possible kernel follow-up, out of B2's scope: a branch-free eps path in the pack emitter
  (apply the 2^ε factor unconditionally, with ε = 0 giving 1) would remove the data dependence
  from the packs. Then `eps_fraction` would no longer matter and `eps_s` would become a constant
  cost, at the price of the multiply on every element at N ≤ 14.
- Recalibrate after any kernel change that moves per-phase times (`tools/calibrate.sh`).

Raw records: the local evaluation workspace, `results/b2/` (calib-1/2/3, validate-1/2/3,
validation-final.txt and validation-kf-n15.txt).

## 5. After the KF chain term (2026-10-05, shipped calibration refit on 297 calls)

Hold-out, 136 calls (the §3 set plus 8 KF cases at N = 15–17):

| measured call | calls | median \|err\| | p90 \|err\| | max \|err\| | target (median / p90) | met |
|---|---|---|---|---|---|---|
| < 100 µs | 34 | 10.1 % | 15.4 % | 20.1 % | 15 % / 30 % | yes |
| 100 µs – 1 ms | 43 | 5.4 % | 14.4 % | 29.1 % | 10 % / 20 % | yes |
| ≥ 1 ms | 59 | 2.8 % | 7.8 % | 20.2 % | 5 % / 15 % | yes |

N = 15–17 only:

| measured call | calls | median \|err\| | p90 \|err\| | max \|err\| | target | met |
|---|---|---|---|---|---|---|
| < 100 µs | 6 | 8.1 % | 12.2 % | 12.6 % | 15 % / 30 % | yes |
| 100 µs – 1 ms | 23 | 8.0 % | 15.1 % | 29.1 % | 10 % / 20 % | yes |
| ≥ 1 ms | 20 | 4.0 % | 7.3 % | 12.4 % | 5 % / 15 % | yes |

The largest N = 15–17 errors: 800³ with dominant-entry data (+26 / +29 %, K4's partial-K-tile
term, see §4), the 12 / 56 CC tensor at N = 15 (+15 %), 800³ and 960³ (+13 … +15 %).
Examples (TFLOPS-eq measured / predicted, time measured / predicted): 800³ at N = 15 8.06 / 7.06
(127 / 145 µs); 1200 × 400 × 700 at N = 16 5.61 / 5.93 (120 / 113 µs); batched 4 × 640² × 384 at
N = 17 5.41 / 5.69 (232 / 221 µs).

## 6. Fidelity follow-up (DESIGN v1.10, host API 7, calibration format 2; branch b2-fidelity)

Four items: wave / occupancy terms for every kernel, a SUSTAINED (power-capped) regime,
prediction intervals with tie-aware ranking, and COLD / WARM cache state. **All numbers below
come from data collected on 2026-10-05 13:20–14:00, after the BIOS update** (0809 → 1836: Intel
default CPU power limits, GPU persistence mode on, PCIe Gen4 x16). That run took ≈ 40 min of GPU
time: 363 calibration records (bench/sweep_calib.txt with the new fine-k̂ and small-pack cases,
plus the occupancy table), 136 hold-out records, 14 sustained windows and 33 cache-pair records.
Data from before the update, and from the period when PCIe was stuck at Gen1, only guided the
choice of features. Raw records are in `~/repos/immerdink-eval/results/b2/fidelity-postbios/`
(`calib-`, `validate-`, `sustained-` and `cache-20261005-1320.jsonl`), with the fits and
validation logs next to them.

### 6.1 Waves and occupancy

- Every launch rule now lives in `src/host/analysis.hpp` (`launch_blocks`, `pack_launch`,
  `kl_launch`, `pack_kl_lg_rows_for`, `k5_launch`, plus the existing `kf_*`). `contract.cu`
  launches with these rules, so the launches are unchanged and the bit-exact GPU tests pass.
  The model computes each launch's waves `⌈B / (SMs · r)⌉` and per-CTA steps from the same rules.
- CTAs per SM, measured with `imd_bench --methods occupancy`
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaFuncGetAttributes`):
  - K0, zero-rows, DIRECT and beta: 6.
  - KF E = 16: 2 (47.5 KB smem, 128 regs); KF E = 4: 4.
  - KL with 512 threads: 2; with 1024 threads: 1.
  - K2 / K3: 2 (40.8 KB, 114 regs).
  - K5: 4–6 depending on N and variant.
  - K4: 1 / 2 / 2 / 3, equal to the configuration table.

  The v1.8 built-in KF residencies (2 / 4) were right.
- Which terms survived (refit on the v1.8 data, confirmed on the post-BIOS data):
  - **memory waves** for KL (`kl.wave` = waves × chunks per row) and K2 / K3 (`pack.wave` = tile
    loads of the busiest SM, ~5.7 µs per wave: a launch of a few hundred tiles runs at ~0.35
    TB/s). For KL the compute chain nc · E · g fitted to 0 and the wave replaced it;
  - KF's compute chain, unchanged;
  - `k5.dram`, the W plane bytes beyond the effective L2 (K5 at 8192², N = 8 was 24 % low).

  The compute chains of K0, K2 / K3 and DIRECT fit to 0; K5's chain is small (5 ns per step).
- **K4 partial K tile, resolved.** The fine-k̂ sweep covered 2048² × k̂ with the rule and
  128×256, and 4864–5120 with 128×128 as well. K4 phase in ps per MAC:

  | k̂ | mod 32 / 64 / 128 | ps/MAC (N = 8) | vs aligned |
  |---|---|---|---|
  | 768, 832, 896, 960, 1024, 1088 | 0 / 0 / 0–64 | 0.0036–0.0039 | — |
  | 800 (also with 2000² rows) | 0 / 32 / 32 | 0.0040 (0.0042) | +3 % |
  | 784, 1040 | 16 / 16 / 16 | 0.0048–0.0049 | +28 … +33 % |
  | 1008 | 16 / 48 / 112 | 0.0050–0.0051 | +35 % |
  | 4864, 4928, 4992, 5056, 5120 | 0 | 0.0032–0.0033 | — |
  | 4960 | 0 / 32 / 96 | 0.0033 (128×128: 0.0040) | +2 % (+14 %) |
  | 4880, 5008 | 16 | 0.0044–0.0045 (128×128: 0.0057) | +38 % (+60 %) |
  | 65552 vs 65536 / 65568 | 16 vs 0 | 0.0063 vs 0.0046 / 0.0049 | +37 % |

  The slow case is **k̂ mod 32 = 16**. Then the int8 residue rows of k̂ bytes are not 32-byte
  aligned: every row starts mid-sector. The partial 64-wide K tile is not the cause (k̂ = 800
  and 4960 have one and are not slow), and neither is the row count (2000 and 2048 rows behave
  the same). In format 2, `mac_ktail` counts chunks with k̂ mod 32 ≠ 0; format 1 keeps mod 64.
  This removes the v1.8 over-prediction at 800³ (+26 … +35 % in K4) and keeps 1008 and 5008
  correct. Possible kernel follow-up, out of B2's scope: padding k̂ to a multiple of 32 would
  make these shapes ~30 % faster in K4.

  **After review (2026-10-05):**
  - The criterion is now applied **per chunk**. K4 reads chunk c of a row at `row·k̂ + c·K_c'`,
    and K_c' = 131056 ≡ 16 (mod 32), so for an aligned k̂ the odd chunks are misaligned. This
    is identical to k̂ mod 32 for single-chunk calls.
  - The two-chunk check measured 512² × 262144 (chunk 1 misaligned) against 262160 (all
    misaligned), N = 8 / 14. K4 took 3.62 / 5.74 ms and 3.33 / 4.64 ms, about 0.006 ps per MAC,
    so it is DRAM-bound. The power-of-two row stride is the slower one, which points to a
    DRAM-channel effect, not alignment. The check is **inconclusive**, and both calls are
    predicted 12–22 % low (like 768² × 100000: long k with few CTAs).
  - Evidence strength: 128×256 has the full sweep. 128×128 is partly counter-evidence (4960 is
    +14 % there). 64×128 and 64×64 are untested.

  **Correction (2026-10-05, DESIGN v1.13 §4b, NOTES-K4ALIGN).** Later measurements found the
  mechanism, and two statements above are wrong.
  - **The "DRAM-channel effect" claim is wrong.** CUTLASS 2.x loads a chunk's residue tile
    (`kc mod 64`) first, so row r's main-loop loads sit at phase `(r·k̂ + chunk end) mod 64`.
    Under this rule:
    - in 262144, chunk 0 (131056 long, ending at 48 mod 64) has every row misaligned, and
      chunk 1 (ending at k̂) is aligned. It is chunk 0, not chunk 1, that is misaligned;
    - in 262160, both chunks have half of the rows misaligned.

    The two calls had equal misaligned fractions, so this check could not separate the
    criteria. The aligned power-of-two stride was slower because of the residue-first
    phase, not because of a DRAM-channel effect. 196608 (aligned, chunk 0 fully misaligned)
    measured worst of the long-k cases: 2048² × 196608, N = 8, 128×256 ran at 23.4
    TFLOPS-eq (70.5 ms; K4 45.7 ms), against 27.5 TFLOPS-eq (60.0 ms) at 196624.
  - **k̂ ≡ 16 (mod 32) misaligns half of the rows, not every row.**
  - The "per chunk" criterion above (`c·K_c' mod 32`) used the chunk start. It happened to
    give the same slow-chunk count. Since v1.13, K_c' = 131008 ≡ 0 (mod 64), so the
    criterion is exactly "k̂ mod 32 ≠ 0".

### 6.2 Hold-out, before and after (post-BIOS data, 136 calls, BURST, REPEAT state)

"Before" is the v1.8 model form refitted on the same post-BIOS calibration data (`perf_fit
--format1`). The old shipped (pre-BIOS) calibration on the same hold-out gives similar numbers:
2.1 / 7.6 / 20.1 % on ≥ 1 ms calls, 6.8 / 15.5 / 32.1 % at 100 µs – 1 ms.

| measured call | calls | before: median / p90 / max | after: median / p90 / max | target |
|---|---|---|---|---|
| < 100 µs | 34 | 6.5 / 15.3 / 30.8 % | 6.8 / 13.4 / 23.6 % | 15 / 30 % |
| 100 µs – 1 ms | 49 | 7.6 / 15.8 / 24.5 % | 6.3 / 15.2 / 19.2 % | 10 / 20 % |
| ≥ 1 ms | 63 | 2.6 / 8.3 / 16.5 % | **1.6 / 6.2 / 14.4 %** | 5 / 15 % |

N = 15–17 (ehrenfest's range):

| measured call | calls | before | after | target |
|---|---|---|---|---|
| < 100 µs | 6 | 10.8 / 16.0 / 16.6 % | 8.3 / 10.2 / 10.7 % | 15 / 30 % |
| 100 µs – 1 ms | 25 | 9.2 / 14.9 / 24.5 % | 8.6 / 15.6 / 17.2 % | 10 / 20 % |
| ≥ 1 ms | 22 | 3.1 / 7.7 / 11.2 % | 2.2 / 6.9 / 14.4 % | 5 / 15 % |

Per phase on calls ≥ 1 ms (median / p90, before → after):

| phase | before | after |
|---|---|---|
| KL B | 6.0 / 18.3 % | 3.9 / 14.2 % |
| K2 | 5.2 / 10.9 % | 4.5 / 9.6 % |
| **K3** | 16.4 / 31.1 % | **11.1 / 23.8 %** |
| K4 | 4.6 / 14.2 % | 4.1 / 15.7 % |
| **K5** | 8.2 / 23.4 % | **4.1 / 17.6 %** |
| KF | 17.5 / 25.5 % | 8.8 / 13.7 % |

All targets are met. Two variants were tried and not adopted:
- without the chain / wave features: ≥ 1 ms median / p90 2.4 / 6.7 %, < 100 µs max 30.7 %;
- with per-record clock normalization: ≥ 1 ms 2.9 / 7.3 % (see §6.3).

Examples (TFLOPS-eq measured / predicted; time measured / predicted):

| call | TFLOPS-eq | time |
|---|---|---|
| sq8192, N = 14 | 37.07 / 37.87 | 29.66 / 29.03 ms |
| sq8192, N = 8 | 63.06 / 63.88 | 17.44 / 17.21 ms |
| t_cc, N = 14 | 32.10 / 31.63 | 4.28 / 4.34 ms |
| 3000³, N = 15 | 24.97 / 23.77 | 2.16 / 2.27 ms |
| 24576 × 384 × 3072, N = 16, g = 2 | 5.88 / 5.81 | 9.86 / 9.98 ms |
| 800³, N = 15 (was +14 … +29 %) | 8.16 / 8.75 | 126 / 117 µs |
| 2048 × 1536 × 816, N = 16 | 11.37 / 11.41 | 452 / 450 µs |
| s512, N = 8 | 10.47 / 10.46 | 25.6 / 25.7 µs |

Largest remaining errors:
- 768 × 768 × 100000, N = 15: −14 % (long-k K4, 3 CTA rounds);
- 640³ at N = 16–17 and 1200 × 400 × 700: −16 … −18 % (KF);
- 1024³, N = 14: +20 % (K4 at 4 rounds).

### 6.3 SUSTAINED regime

- **The windows.** `imd_bench --methods sustained` runs each case with a 15 s pre-heat, then
  measures for 6–10 s, reading NVML power and clock every 20 ms; the energy counter is never
  read. Every tensor-core-heavy window sits at 449–450 W and 2308–2424 MHz (sq2048 … sq12k,
  t_cc, t_df, 2048² × 32768). The small ones (512³, 1024³, r_mm) sit at 390–449 W and
  2680–2790 MHz.
- **Clock reference.** Clock normalization of the burst fit did not help (§6.2), so the
  parameters θ are the costs of the burst state as measured. `power.clock_ref_mhz` is the
  effective clock they represent, scanned on the windows: 2530 MHz. The sustained
  leave-one-case-out median error falls from 4.7 % at 2805 MHz to 1.5 % at 2530 MHz.
- **Power fit,** with the awake idle fixed at 60 W: e_tc 0.67 pJ per int8 op, e_byte 335 pJ per
  DRAM byte (an effective value), e_alu 0, γ = 1.
- **Result, leave-one-case-out:**

  | window | TFLOPS-eq measured | time measured / predicted | error | clock measured / predicted |
  |---|---|---|---|---|
  | sq8192, N = 14 | 35.44 | 31.02 / 30.62 ms | −1.3 % | 2312 / 2377 MHz |
  | sq8192, N = 16 | | | −1.2 % | |
  | sq12288, N = 14 | 37.40 | 99.2 / 97.6 ms | −1.7 % | |
  | sq4096, N = 8 / 14 | | | +1.2 / +1.9 % | |
  | sq2048 | | | +2.3 % | |
  | t_cc | | | +0.5 % | |
  | t_df | | | +1.2 % | |
  | 2048² × 32768 | | | −0.2 % | |
  | small windows (512³, 1024³) | | | +1 … +10 % | |

  The small windows are high because back-to-back calls hide part of the launch overhead that
  the call term charges. **Median |error| 1.5 %, max 10.5 %; capped windows within ±2.3 %,
  settled clock within ±5 % (median 2.8 %).** For comparison, the BURST prediction of the same
  windows has a median error of 4.3 %, and the capped ones come out 3–6 % too fast.
- The default regime is **SUSTAINED**, as proposed; below the cap it equals BURST. Design
  question for the reviewer. Examples, SUSTAINED vs BURST at N = 14: 4096³ ×1.061 (2339 MHz),
  8192³ ×1.063 (2357 MHz), 16384³ ×1.056 (2385 MHz, where long runs were seen at ~1950 MHz:
  thermal history beyond a 15 s pre-heat is not modeled), 1024³ ×1.000.

### 6.4 Prediction intervals and ties

- Bands come from the hold-out residuals (measured / predicted): per size class of the
  predicted call, and per kernel class for phases. Coverage was checked by 2-fold
  cross-validation over the cases with 20 random halvings:

  | quantiles | covered | < 100 µs | 100 µs – 1 ms | ≥ 1 ms |
  |---|---|---|---|---|
  | 10 / 90 (in-sample percentiles) | 72 % | | | |
  | 5 / 95 | 80 % | | | |
  | **2.5 / 97.5** | **84 %** | 88 % | 78 % | 87 % |

  The stored band uses the 2.5 / 97.5 quantiles (rule: the narrowest pair that reaches 80 %
  cross-validated). Shipped call bands, as factors of the predicted time: < 100 µs 0.874–1.212,
  100 µs – 1 ms 0.864–1.173, ≥ 1 ms 0.914–1.064.
- These bands already contain the clock spread:

  | size class | NVML clock after the timed loop (p10–p90) | timed-loop spread, median (p90) |
  |---|---|---|
  | < 100 µs | 2790–2805 MHz | 28 % (153 %, single slow calls; medians are used) |
  | 100 µs – 1 ms | 2790–2805 MHz | 5.8 % (13.8 %) |
  | ≥ 1 ms | 2550–2805 MHz | 0.9 % (2.7 %) |

  The per-call probe reads 2294–2561 MHz (p10–p90) at ≥ 1 ms: ~11 % below NVML and noisy, so
  it is only a relative indicator.
- **Ranking ties, changed from the design after the data** (the ratio rule was accepted in
  review; the default order is **TIME**, and TIES_BY_FOOTPRINT is opt-in, with `tie` set in every
  order). "Bands overlap the best" tied 236 of
  360 candidates on a CC tensor, up to 1.36× slower, and the footprint tie-break then picked
  slow ones. Candidates of one contraction share most of the model error: across pairs of
  hold-out calls of the same case, the error of the time ratio is ±7.5 % at the 80th
  percentile. So a tie is now `time_s ≤ best × rank.tie_ratio`, with tie_ratio = 1.075 fitted
  by `perf_validate --bands`. Ties come first, ordered by footprint, and `raw_rank` keeps the
  time order. On the same CC tensor 28 of 360 candidates are now ties, and 16 of 176 on 2048 ×
  1000 × 3000. Design question for the reviewer.

### 6.5 Cache state

| case (N = 14 / 16) | REPEAT | COLD vs REPEAT, measured | COLD vs REPEAT, model | WARM vs REPEAT, measured |
|---|---|---|---|---|
| 256³ | 16.5 / 19.4 µs | +11.6 / +10.4 % | +7.6 / +6.8 % | +3.1 / +0.5 % |
| 512³ | 32.0 / 46.1 µs | +14.1 / +15.6 % | +17.8 / +13.5 % | +2.4 / +0.6 % |
| 1024³ | 125 / 173 µs | +11.3 / +4.2 % | +7.7 / +1.2 % | −1.6 / −3.9 % |
| 2048³ | 722 / 843 µs | −1.9 / −0.5 % | 0 % | −2.5 / −0.9 % |
| 4096 × 768 × 768 | 329 / 435 µs | −5.3 / −2.5 % | 0 % | −7.4 / −4.6 % |
| 256 × 3000 × 3000 (N = 14) | 457 µs | −2.0 % | 0 % | −0.9 % |

- COLD (an evicted L2) costs +4 … +16 % when the call's footprint fits the L2, and nothing
  beyond that.
- WARM (the previous call wrote this call's A) is within −7 … +3 % of the repeat state, so it is
  modeled as REPEAT. The design's `min(I, L_w)` would have predicted 10–20 % savings on large
  calls.
- The parameters are COLD costs: the fitter adds the repeat records' resident-input savings
  back to their phases, and a REPEAT prediction subtracts them again. Without this, the COLD
  prediction lost monotonicity in N: a larger footprint keeps less resident, so the cold
  add-back shrank.
- Fitted: L_w = 0.8 × 72 MiB, 1.46 ps per resident input byte.
- The default `cache = COLD` predicts small calls 4–18 % slower than v6 did, because v6 was
  fitted on the repeat state and predicted it. Design question.

### 6.6 Planner hot path

The shipped calibration is now parsed once in C (a function-local static), so passing a NULL
calibration costs nothing extra, and `immerdink.planner` now passes NULL. The v7 fields do not
change per-query costs: in `planner_overhead`, predict takes 1.6–1.8 µs and rank 8–18 µs.

### 6.7 Review decisions (2026-10-05)

- Accepted: the SUSTAINED default (with its two biases documented), the COLD default
  (recommend WARM when an input is the previous output), WARM = REPEAT (provisional), and the
  ratio tie rule.
- Changed:
  - the default rank order is TIME (`IMD_PERF_ORDER_TIME = 0`; `IMD_PERF_ORDER_TIES_BY_FOOTPRINT
    = 1` is opt-in);
  - `tie_ratio` must be ≤ 1.25;
  - an explicit `clock_mhz` is clamped to the effective clock, renamed `clock_eff_mhz`;
  - the intervals are named `time_lo_s` / `time_hi_s` and `phase_lo_s` / `phase_hi_s`, and the
    bands are `{lo, hi}`;
  - a version-6-sized calibration is a true format-1 view (format-2 parameters zeroed);
  - the `mac_ktail` test is per chunk;
  - the planner takes regime, cache and order by name.
- The shipped parameters are unchanged: no fit or hold-out case changes under the per-chunk
  rule.

### 6.8 Later (ROADMAP)

- **K4 alignment fix in the kernel.** Set K_c' = 131040 (≡ 0 mod 32) and pad k̂ to a multiple
  of 32. Half of all k̂ values pay 28–38 % in K4. This is a layout change, so it needs DESIGN
  and TLA+ work first. **Superseded by DESIGN v1.13 §4b.** Class 32 is also slow on the
  smaller tiles, so the target is K_c' = 131008 (done in v1.13 stage 1) and k̂ padded to 64 as
  both the stride and the K extent (stage 2).
- A **settled-clock / hot calibration point** for long thermal runs (16384³ at ~1950 MHz).
- A **clock-scaled e_byte** in the power model.

### 6.9 Open

- KF on 640–1200-class shapes at N = 16–17 is predicted 15–18 % low. K4 at 3–4 CTA rounds
  (768² × 100000, 1024³) is off by ±15–20 %.
- The power coefficients are effective values: e_byte absorbs the SM power of memory-bound
  work. A sequence-level solve for mixed sequences is left to the caller, from `power_w` per
  call. Thermal history beyond the 15 s pre-heat is not modeled: 16384³ long runs were seen at
  ~1950 MHz, while the model predicts 2385 MHz.
- In small sustained windows, back-to-back calls overlap launch latency that the call term
  charges (+10 % at 512³). A "queued" call term would fix this if it matters.

## 7. Accuracy-aware ranking (DESIGN v1.12 §6f, host API 8; branch b2-accuracy)

Eric approved this item. `rank` answers **the cheapest configuration that meets an error target**,
using the §3.8 bounds and the operand statistics the caller supplies. The formulas and API are in
DESIGN §6f. This section holds the evidence and the open questions.

### 7.1 Design in brief

- **Rigorous.** PROOFS L3a's ℓ1 form, with the caller's optional ℓ1/ℓ2 ratios (√k by default
  gives §3.8 exactly), composed with the final rounding per metric:
  - NORMWISE needs nothing from the caller;
  - ELEMENTWISE needs ρ ≥ max ‖a‖‖b‖/|x|;
  - ABSOLUTE needs a bound on max ‖a‖‖b‖.

  DIRECT uses the native γ_k.
- **Typical.** Hoeffding under the standard probabilistic model of RN quantization errors
  (independent, zero-mean, |e| ≤ ½), with a union bound over the M = b·m·n outputs at failure
  probability p (default 2⁻⁴⁰, review). No fitted constants. TRUNC and DIRECT report the rigorous bound
  as their typical value.
- **One implementation.** The math is in C (`src/host/accuracy.cpp`). `spec/params.py`
  (`error_estimate`, exact rational enclosure) is the reference, and `cert_error_estimate`
  checks the bound on emulated rows. Python only marshals.

### 7.2 Validation on the GPU (rigorous bound; typical-estimate coverage)

**Calibration set: the M5 records.** These are 520 runs: 4 CC fixtures × 5 data families ×
N = 8…20 × RN / TRUNC, with the maximum taken over a sampled grid of 4096 or 512 outputs. On
them, `host.err_estimate` gives:

- rigorous: max measured / bound 0.908;
- typical (RN), estimated over the grid's outputs, with p = 2⁻⁴⁰ (the default after the
  review): covers 260 / 260, with max measured / estimate 0.91 overall and 0.21 where the
  emulation error dominates (N ≤ 13);
- TRUNC's typical value is its bound, so its 260 / 260 is true by construction.

The estimate has no fitted constant. This set was only checked; nothing was tuned on it.

**Held out: `bench/accuracy/err_api.py`, 300 random cases.** The cases cover:
- 10 families: φ = 0.5, 1, 2, 4; rows and columns scaled by 2^U(−300, 300); uniform; same
  sign; M5's decay_opp and outliers; and "constant" rows, which are outside the typical model;
- k = 64…93 564 (log-uniform);
- N = 5…20, RN (236 cases) and TRUNC (64);
- the GEMM `ik,kj->ij` and the two-label `akp,bkp->ab`;
- sampled grids of 16² to 32² outputs with an exact reference.

The statistics come from the data, as a caller would supply them. Raw data:
`~/repos/immerdink-eval/results/b2/accuracy-20261005/` (grids, records, report).

Results:
- **Rigorous: 0 violations** in every metric (NORMWISE with ν = √k and with the measured
  ℓ1/ℓ2 ratios, ELEMENTWISE with the measured ρ, ABSOLUTE). The maximum measured / bound is
  0.964, at N ≥ 16, where the final-rounding term u is attained.
- **Supplying the ℓ1/ℓ2 ratios** tightens the rigorous bound at N ≤ 13 (median over a family):
  - decay_opp: 0.31× (about ½ modulus);
  - outliers: 0.13×;
  - φ = 4: 0.11× (about ¾ modulus);
  - constant rows: 1× (ν = √k).
- **Typical estimate, RN.** Coverage is over the grid's outputs, by family. "Emulation regime"
  means the typical coefficient is > 8u, so the emulation error rather than the final rounding
  is measured.

| family | cases | covered | max meas / typ | median | emulation regime: covered, max |
|---|---|---|---|---|---|
| φ = 0.5 | 29 | 29 | 0.18 | 0.12 | 23 / 23, 0.18 |
| φ = 1 | 25 | 25 | 0.35 | 0.12 | 16 / 16, 0.16 |
| φ = 2 | 27 | 27 | 0.34 | 0.12 | 16 / 16, 0.14 |
| φ = 4 | 26 | 26 | 0.32 | 0.10 | 20 / 20, 0.15 |
| scaled 2^±300 | 23 | 23 | 0.21 | 0.12 | 16 / 16, 0.19 |
| uniform | 22 | 22 | 0.23 | 0.11 | 14 / 14, 0.19 |
| same sign | 22 | 22 | 0.84 | 0.13 | 19 / 19, 0.16 |
| decay_opp | 21 | 21 | 0.17 | 0.03 | 14 / 14, 0.17 |
| outliers | 21 | 21 | 0.69 | 0.10 | 16 / 16, 0.12 |
| **constant rows** | 20 | **2** | **19.6** | 4.7 | **1 / 19**, 19.6 |

This is with p = 2⁻⁴⁰. With the first default, 2⁻²⁰, the constant rows were covered 1 / 20,
with a maximum of 25.5, and every other row stayed covered.

  - On data with random mantissas the estimate covered **all 216 RN cases**. In the emulation
    regime (154 cases) the maximum measured / estimate is 0.19, so the estimate is about 5×
    above the observed maximum.
  - It is a median 0.17× of the rigorous bound: about ⅔ modulus at these k, and more at large
    k (the bound grows as √k, the estimate as √ln M).
  - Rows of equal, non-representable entries violate it by up to 20×, as designed. Their
    errors add coherently and stay within the rigorous bound.
  - **So the typical estimate is reliable for data whose quantization errors behave
    randomly.** That is why it ships opt-in (`basis = TYPICAL`) and is labelled "not a
    guarantee". A caller with structured operands (constant or low-entropy rows, e.g.
    uniform initial states) must use the rigorous bound.

**GPU time:** about 11 s of contractions for the 300 cases plus 0.2 s for the quick run, under
the shared GPU lock. The check phase runs on the CPU (126 s, re-run after the review in 124 s
on the same grids), with the lock not held.

### 7.3 Tests and cost

- **`host_accuracy` (new, CPU):**
  - 20 000 random `imd_err_estimate` cases vs `params.error_estimate`'s exact enclosure
    (18 585 valid, 661 without a bound, 1415 rejected requests): the bound lies within
    [exact, exact·(1 + 2⁻³⁹)], and the typical estimate is within 1e−11;
  - 1000 `imd_err_min_moduli` cases equal a scan of the C and params estimates;
  - 400 rank / predict cases (9480 candidates): feasibility, infeasible bits, order and best
    candidate as specified;
  - v7-sized structs behave as v7.
- **`params.py --check`:** 27 certificates, including the new `cert_error_estimate` (640
  emulated and native elements, 3 metrics each).
- **`host_c_api`:** the v8 sizes and offsets, the estimates, min_moduli, predict and rank with
  a target.
- **`host_planner`:** parity with an Accuracy in 3000 single queries, 600 rank cases and the
  min_moduli queries.
- **`accuracy_err_api_quick` (GPU):** 20 small cases, exit 1 on any violation.
- **Cost.** `planner.Query.min_moduli` takes about 1.7 µs on a non-quiet machine (one ctypes
  call: one lowering plus at most 19 closed forms). `bench/planner_overhead.py` and
  `imd_planner_floor` now include it.

### 7.4 Design questions (reviewed 2026-10-05)

**Rulings.** The independent reviewer accepted the design: the bound matches the proofs, and
the ε path, g, the vec mode, packing and K4 were confirmed exact. Changes:
- **Q4.** `fail_prob` defaults to **2⁻⁴⁰**, about 1.22× at M = 10⁶. For a probability P over a
  run of C calls, pass p = P/C.
- **Q5.** The bound stays on D. The α / β composition is documented: α ≠ 1 turns n into
  n(1+u) + u, and ABSOLUTE also folds |α| into `norm_product`; β ≠ 0 adds u·|βC + αD|.
- **Q6.** `rank(...)[0]` meets the target only if it is `.feasible`; otherwise it is the
  fastest infeasible candidate. This is documented.
- **Q7.** `min_moduli`'s two NOT_SUPPORTED cases are told apart by `err_status`: MISSED means
  the target is unattainable; UNCHECKED means the lowering's own status (a repeated or
  broadcast C label). Both are tested.
- **Q11.** `reserved` ≠ 0 is INVALID_VALUE, so it can become flags (e.g. an α-rounding flag).
- The other questions were accepted as proposed.

**Must-fixes, all done:**
- **Underflow.** The ABSOLUTE bound has an absolute term: 2⁻¹⁰⁷⁵ in the specification (it
  also covers a gradual underflow of D) and 2⁻¹⁰⁷¹ in C. DIRECT ABSOLUTE has 2k·2⁻¹⁰⁷⁵
  (in C, (k + 16)·2⁻¹⁰⁷⁴). `host_accuracy` draws subnormal norm products: 2000 cases, none
  below the exact enclosure.
- **PROOFS L3c:** the ℓ1/ℓ2 refinement, the per-metric composition, DIRECT's γ_k, and
  underflow.
- **Doc numbers:**
  - E/E_typ = √(k / (2 ln(2M/p)));
  - E_typ ≈ 18.4/T at M = 10⁶ and p = 2⁻⁴⁰.

**Cheap fixes, also done:**
- `compose` no longer multiplies by (1 + u), which rounds to 1.0;
- the size constants are explicit;
- `min_moduli` checks the lowering only, not N = 20's workspace;
- `cert_error_estimate` includes constant rows and k = 600 and 1200 at N = 3–5;
- the risk cases are named in the TYPICAL docstring;
- §6d.4 says `min_moduli` does not raise;
- the M5 TRUNC coverage is marked as true by construction;
- a v8 request with v7-sized candidates still marks accuracy misses infeasible.

**The questions as asked (conservative choices taken):**

1. **The typical estimate comes from a model and is not fitted.** Hoeffding gives a
   probabilistic bound with stated assumptions.
   - On the M5 data it is about 4× above the observed maxima (about half a modulus).
   - A constant fitted to the data would recover that, but it would have no model behind it
     and no coverage guarantee outside the fitted families.
   - Chosen: the model. The coverage is reported, not tuned.
2. **TRUNC and DIRECT have no typical estimate** (`err_typical` = `err_bound`). TRUNC's errors
   are biased by the data's signs (§3.3), so the zero-mean model does not apply. For DIRECT a
   probabilistic γ̃_k is possible but brings little.
3. **The default basis is the rigorous bound.** The typical estimate is opt-in
   (`basis = TYPICAL`) and always reported.
4. **`fail_prob` defaulted to 2⁻²⁰ per call** (ruled: 2⁻⁴⁰). This is a union bound over every output of the
   call. A caller making 10⁹ calls should lower it; the cost is about √(ln) and grows slowly.
5. **The bound is on the product D** (α = 1, β = 0). With other α and β, the §3.7 roundings of
   the epilogue (≤ u|αD| and one fma) are the caller's to add. Folding them in would need α,
   β and |C| statistics.
6. **Rank keeps v6's order.**
   - Candidates that miss the accuracy target sort with the budget misses, as infeasible
     SUCCESS by time; there is no separate group.
   - No candidate is pruned before prediction. The ones the target rules out are the cheap
     low-N ones, and the recommended hot-path pattern (`min_moduli`, then `rank([n, n+1])`)
     avoids predicting them.
7. **`imd_err_min_moduli` returns NOT_SUPPORTED** (with `num_moduli` = 0 and the estimates at
   `n_hi`) when no N meets the target. For DIRECT dispatch it returns `n_lo`, since N does not
   matter there.
8. **One scalar ρ (ELEMENTWISE) and one `norm_product` (ABSOLUTE) per call.** These are the
   worst case over the outputs the caller cares about. Per-row or per-column vectors were not
   taken: they would mean data-sized arguments on a hot path.
9. **A v8 predict or rank always reports the NORMWISE estimates**, even without a request.
   This costs one closed-form evaluation per candidate.
10. **The C bound is made rigorous by an inflation factor (1 + 2⁻⁴⁰)** rather than by directed
    rounding modes, which are portable only with a change of the FP environment. The test
    checks both directions against an exact enclosure.
11. **Statistics validation:**
    - INVALID_VALUE for `l1_ratio` < 1 (but ≠ 0), ρ < 1, a non-positive `norm_product` for its
      metric, `fail_prob` outside (0, 1), and a negative or non-finite target;
    - an `l1_ratio` above √k is clamped (it is a valid but loose bound).
12. **The ε path and the K4 configuration do not enter the error.** They are exact, so
    accuracy depends only on (N, rounding, dispatch, k, M) and the statistics. g, the vec mode
    and packing have no effect.
