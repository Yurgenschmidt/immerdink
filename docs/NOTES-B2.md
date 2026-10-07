# B2 notes: the performance model (DESIGN v1.8 §6c; v1.10 fidelity follow-up in §6; v1.18 in §12)

ROADMAP B2. A CUDA-free model in libimmerdink_host (host API version 6): `imd_perf_predict`
(per-phase and total time, TFLOPS-eq, MEMORY_MODEL §4 bounds), `imd_perf_predict_pack`,
`imd_perf_rank` (N × g × vec mode × packed operands, ranked by predicted time inside a memory
budget, with exact footprints), and the calibration (`imd_perf_calib_*`, a `key = value` text
file; the RTX 4090 one is embedded). Design reviewed 2026-10-05 (roofline K4 accepted; eps
path as a data input, retired in DESIGN v1.15 after the branch-free pack emitter, §10;
effective L2 and the fitter's seed calibration). All numbers below are
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
  **Done differently in DESIGN v1.14 §3.4.1** (2026-10-06): no multiply at all — |a'| is held
  as 2 or 3 u32 words and reduced by byte weights, so the pack cost is data-independent and
  N ≤ 13 is unchanged. The model's eps terms are still in place until a recalibration retires
  them.
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
    Under this rule (both calls had three chunks under 131056, since 2·131056 = 262112 < k̂,
    so the "two-chunk check" above was really a three-chunk check):
    - in 262144, chunk 0 (131056 long, ending at 131056 ≡ 48 mod 64) is in class 48, every
      row misaligned; chunk 1 (ending at 262112 ≡ 32 mod 64) is in class 32, every row
      off the 64-byte phase but not misaligned in the format-2 sense; chunk 2 (32 long, ending
      at k̂) is aligned. It is chunk 0, not chunk 1, that is misaligned;
    - in 262160, every chunk (131056, 131056, 48 long) has half of the rows misaligned.

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
- A **settled-clock / hot calibration point** for long thermal runs (16384³ at ~1950 MHz). **Measured in §9:** no THERMAL regime (the thermal part is ≈ 1–1.8 %; documented as +1–2 % drift for minute-scale runs).
- A **clock-scaled e_byte** in the power model. **Proposed in §9 / DESIGN §6c.7.**

### 6.9 Open

- KF on 640–1200-class shapes at N = 16–17 is predicted 15–18 % low. K4 at 3–4 CTA rounds
  (768² × 100000, 1024³) is off by ±15–20 %.
- The power coefficients are effective values: e_byte absorbs the SM power of memory-bound
  work. A sequence-level solve for mixed sequences is left to the caller, from `power_w` per
  call. Thermal history beyond the 15 s pre-heat is not modeled: 16384³ long runs were seen at
  ~1950 MHz, while the model predicts 2385 MHz. **Superseded by §9:** measured settled 2240–2270 MHz, time −3.7 % vs SUSTAINED.
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

## 8. Refit after K4 alignment stage 2 (2026-10-06, branch `k4-align-s2-refit`)

DESIGN v1.13 §4b.6, review ruling "no calibration format bump; refit θ". Stage 2 pads k̂ to
64 for k > 96, so the format-2 slow-chunk term (`mac_ktail`, "k̂ mod 32 ≠ 0") is reached
only by k ≤ 96 plans under AUTO. The shipped θ (2026-10-05) had been fitted on A = 16 data
with K_c' = 131056.

**Data** (RTX 4090, stock clocks, quiet gate and shared GPU lock, `ehrenfest_busy.sh` FREE
before every run; profiling `imd_bench` at f9c4950):
- calibration: `tools/calibrate.sh` over bench/sweep_calib.txt, 01:07–01:29 (22 min), 447
  timed calls. The sweep keeps every earlier case, which now runs at AUTO (the `t_k*` fine-k̂
  cases measure the aligned rate at the padded k̂). It adds 14 cases (`a_*`):
  - A = 16 forced through `--k-align 16` at k > 96 (k̂ mod 32 = 16) on all four K4
    configurations: 2048² × {784, 1008, 1040, 4880, 5008}, 256 × 3000 × 1000,
    512² × 1000, 2000 × 128 × 30000 and 768² × 65552;
  - AUTO k ≤ 96 (k̂ = 48 and 80 are slow, 64 and 96 are not) at 2048² and 512².
- sustained and cache records: reused from 2026-10-05 13:20 through the new
  `tools/calibrate.sh --sustained FILE --cache FILE`. This saves about 13 GPU minutes, and
  is exact because every case of those two sweeps has k ≡ 0 (mod 64), so its plans are
  unchanged by stage 2.
- hold-out: bench/sweep_perf_validate.txt, 01:29–01:44 (15 min), 178 calls. It adds 16 cases
  (`z_*`) with k across the AUTO range: 1536² × {80, 112, 144, 208, 528, 1000, 2032, 4080},
  12288 × 384 × {144, 1040}, 400² × 200, 600³, a DF `acQ,bdQ->acbd` 56⁴ with Q = 2000, a
  CC `kjbc,kiac->abij` with o = 19, v = 112, and two A = 16-forced cases (1536² × 1008 and
  3000 × 700 × 5000).
- Raw records, fits and logs are in immerdink-eval `results/k4align/refit/`.

**Fitted K4 terms** (ps per MAC at full occupancy; slow chunk = `mac_ktail` / that rate):

| configuration | round µs, before → after | MAC rate, before → after | slow-chunk extra, before → after |
|---|---|---|---|
| 128×256 | 3.56 → 2.91 | 0.389 → 0.404 | +36 % → +26 % |
| 128×128 | 3.69 → 3.64 | 0.394 → 0.390 | +71 % → +80 % |
| 64×128 | 3.09 → 3.00 | 0.562 → 0.557 | +54 % → +86 % |
| 64×64 | 4.23 → 3.57 | 0.373 → 0.472 | +263 % → +166 % |

**Fitter change: monotone K4 rounds.** The first refit gave 64×64 mac1 = 0.86 ps and
mac2 = 0, so a round of two concurrent CTAs cost less than a round of one. host_perf then
failed its monotonicity check: 3142 × 128 × 4434 predicted faster than 3142 × 64 × 4434 under
64×64 at N = 6, g = 2. The calibration's 64×64 cases have no two-CTA rounds, so mac2 was
unidentified. `bench/perf_fit.py` (`k4_round_monotone`) now fits the per-round MAC costs
c_q = q·mac_q as cumulative sums of non-negative increments, so c_1 ≤ c_2 ≤ c_3 with NNLS
unchanged. Fit quality is the same or better: 64×128 phases 9.0 / 21.2 → 7.7 / 20.5 %, and
hold-out K4 p90 on ≥ 1 ms 14.3 → 13.0 %. The constraint is active for 64×128 (c_1 = c_2) and
64×64 (c_2 = c_3).

Every `mac_ktail` stays determined (non-zero) with the forced cases. The size ordering of the
slow-chunk penalty (128×256 smallest) matches NOTES-K4ALIGN §2. The other changes:
- K4 effective L2: still 0.40 × 72 MiB.
- K4 DRAM rates: 775 / 775 / 737 / 667 GB/s.
- `power.clock_eff_mhz`: 2530 → 2455 MHz (sustained leave-one-out median 1.6 %).
- `rank.tie_ratio`: 1.075 → 1.078.
- Bands: the 5 / 95 hold-out quantiles, cross-validated coverage 82 %. Call bands are
  < 100 µs 0.906–1.251, 100 µs – 1 ms 0.890–1.108 and ≥ 1 ms 0.926–1.028.

In sample, on the new calibration records, the ≥ 1 ms max error falls from 29.5 % (old θ) to
16.1 %. The old θ predicted 256² × 150000 at g < N (`c_rkl2g`, K4 memory-bound, k̂ now
150016 instead of the slow 150000) up to 21–30 % too fast; the refit has it within −11 … +4 %.

**Hold-out, before (shipped θ of 2026-10-05) and after (the shipped refit), same 178 calls,
BURST, REPEAT** (median / p90 / max |error| of the predicted call time):

| subset | measured call | calls | before | after | target |
|---|---|---|---|---|---|
| all | < 100 µs | 44 | 8.9 / 16.0 / 23.1 % | 7.2 / 17.3 / 24.6 % | 15 / 30 % |
| all | 100 µs – 1 ms | 68 | 4.1 / 14.6 / 19.6 % | 4.5 / 14.8 / 19.7 % | 10 / 20 % |
| all | ≥ 1 ms | 66 | 1.9 / 6.5 / 15.8 % | 1.9 / 7.3 / 15.3 % | 5 / 15 % |
| N = 15–17 | < 100 µs | 6 | 8.9 / 9.9 / 9.9 % | 8.9 / 10.5 / 10.5 % | |
| N = 15–17 | 100 µs – 1 ms | 26 | 8.3 / 15.7 / 19.6 % | 8.0 / 15.8 / 19.7 % | |
| N = 15–17 | ≥ 1 ms | 24 | 2.8 / 7.0 / 15.8 % | 2.7 / 7.4 / 15.3 % | |
| new AUTO-range cases (`z_*`) | < 100 µs | 9 | 12.0 / 16.8 / 19.0 % | 6.0 / 19.3 / 20.3 % | |
| new AUTO-range cases | 100 µs – 1 ms | 20 | 2.8 / 8.3 / 16.2 % | 2.5 / 7.4 / 16.4 % | |
| new AUTO-range cases | ≥ 1 ms | 3 | 3.8 / 5.7 / 6.2 % | 2.9 / 6.4 / 7.3 % | |
| k mod 64 ≠ 0 (A matters) | < 100 µs | 17 | 10.4 / 16.8 / 19.0 % | 7.4 / 19.6 / 20.3 % | |
| k mod 64 ≠ 0 | 100 µs – 1 ms | 39 | 3.5 / 11.6 / 19.6 % | 3.2 / 11.3 / 19.7 % | |
| k mod 64 ≠ 0 | ≥ 1 ms | 18 | 2.9 / 9.4 / 15.8 % | 3.9 / 7.8 / 15.3 % | |
| k mod 64 = 0 | < 100 µs | 27 | 6.9 / 15.2 / 23.1 % | 6.9 / 14.7 / 24.6 % | |
| k mod 64 = 0 | 100 µs – 1 ms | 29 | 6.2 / 15.0 / 16.2 % | 6.8 / 15.1 / 16.1 % | |
| k mod 64 = 0 | ≥ 1 ms | 48 | 1.4 / 5.9 / 10.3 % | 1.8 / 6.7 / 11.4 % | |
| k ≤ 96 or A = 16 forced | all | 8 | 0.3–17.7 %, median 3.6 % | 0.5–20.3 %, median 4.0 % (the max is 96³ at N = 14, KF; 1536² × 80 +13 / +14 → +6 / +8 %) | |

Per phase on calls ≥ 1 ms (median / p90, before → after):
- K4: 3.3 / 14.5 → 4.7 / 13.0 %
- K0: 7.8 / 14.6 → 5.3 / 17.4 %
- KL A: 4.1 / 13.5 → 4.1 / 14.1 %
- K3: 11.7 / 22.0 → 11.9 / 22.2 %
- K5: 4.3 / 17.0 → 4.4 / 17.5 %

All targets are met before and after.

What the refit buys is on the shapes stage 2 changed:
- The small-k AUTO calls lose the over-prediction: the old θ priced A = 64 as if it were
  still partly slow. Before → after (call time measured / predicted with the refit):
  - 1536² × 80: +13 / +14 % → +6 / +8 % (63.2 / 66.8 µs at N = 8; 5.98 / 5.65 TFLOPS-eq);
  - × 112: +12 / +11 % → +6 / +6 %;
  - × 144: +6 / +3 % → +1 / −1 %;
  - × 208: +7 / +7 % → +2 / +3 %.
- The two A = 16-forced hold-out cases: 1536² × 1008 −1.8 / +3.5 % → −4.8 / +0.5 %, and
  3000 × 700 × 5000 +3.8 / +1.6 % → +1.5 / −0.8 %.

On aligned shapes the refit is neutral to slightly worse (≥ 1 ms p90 5.9 → 6.7 %, within the
spread of two calibration days). The largest remaining errors are unchanged by the refit:
- KF on 600³ / 400² × 200 / 96³: −13 … −20 %;
- 12288 × 384 × 1040: −7 … −10 %;
- DF 56⁴ at N = 16: +7 %.

Examples, refit (TFLOPS-eq measured / predicted; time measured / predicted):

| call | TFLOPS-eq | time |
|---|---|---|
| sq8192, N = 14 | 36.25 / 36.88 | 30.34 / 29.82 ms |
| t_cc (32 / 128), N = 14 | 31.85 / 31.05 | 4.32 / 4.43 ms |
| DF 56⁴, Q = 2000, N = 8 | 39.45 / 38.67 | 0.997 / 1.017 ms |
| CC o = 19, v = 112, N = 16 | 16.94 / 17.44 | 1.138 / 1.105 ms |
| 1536² × 4080, N = 8 | 34.72 / 34.21 | 0.555 / 0.563 ms |
| 1536² × 80, N = 8 | 5.98 / 5.65 | 63.2 / 66.8 µs |

**`k4_select` re-check (no change).** The K4 rule constants (DESIGN §4, NOTES-A3 §3) come from
84 A3 cases with mostly power-of-two k. Under stage 2, A = 64 gives every k > 96 the aligned
rate, so the question was whether the smaller tiles now win where the rule keeps 128×256.
- Grid: 15 views × N = 8 / 14, each with every configuration forced plus the rule, call
  medians of 20–300 reps, all with k ≢ 0 (mod 64) so that k̂ is padded. The views were 384²,
  512² and 1024² × 1000, 512² × 4080, 768² × 760, 1024² × 250, 1536² × 500, 2048² × 1000,
  256² × 8100, 128² × 16300, 1000 × 64 × 50000, 4096 × 128 × 4000, 4 × 256² × 500,
  256 × 64² × 100 and 512² × 32700.
- Ran at 01:44–01:48 (3.5 min). Raw data and scripts: `results/k4align/refit/k4grid*`.
- **Result:** the rule's calls sum to 11.42 ms, against 11.35 ms for the per-view fastest
  configuration (+0.65 %) and 12.12 ms for always 128×256 (−5.8 %).
- The rule is the fastest configuration or within 1.5 % in 27 of 30 groups.
  - The three exceptions are 4 × 256² × 500 at N = 14 (128×256 instead of 128×128,
    1.048×), 768² × 760 at N = 8 (128×128 instead of 64×128, 1.034×) and 256² × 8100 at
    N = 8 (64×64 instead of 64×128, 1.066×).
  - These are 45–100 µs calls. Measured with the rule's own record instead of the forced
    run of the same configuration, the worst pick is 1.014×, so the gap is within
    cross-process noise.
- Offline (`k4margin.py`, the rule transcribed; it agrees with `k4_rule` in 30 / 30), the
  margin changes almost nothing:
  - margin 0 or 25 ‰: +0.52 %;
  - 50 ‰ (shipped): +0.53 %;
  - 75–150 ‰: +0.58 %, worst pick 1.09×.
- The calibration's own forced-configuration cases agree: 32 groups, the rule +0.04 % over
  the per-group fastest. Its worst picks are on calls of 15–25 µs (128³, 384³), where the
  noise is larger.
- So the rule is unchanged: no alternative is clearly better.
- Caveat: an A = 16-forced plan (test hook only) is estimated over its unpadded k̂ and may
  pick another configuration than the AUTO plan of the same k. This affects calibration
  and tests, not production.

## 9. Long hot runs and the power model's e_byte (2026-10-06, branch `followup-hotclock`; DESIGN §6c.7 draft)

The question: the SUSTAINED regime predicted 2385 MHz for long 16384³ runs, where §6.3 and
§6.9 had ~1950 MHz. And is e_byte's missing clock scaling what makes the clock bisection
optimistic (solved clock too high)?

**Method.**
- `imd_bench --methods sustained --sustain-warm-s 0 --sustain-trace-s 2` (new option: a
  time series in the record).
- Windows: 240 s / 120 s on 16384³, 180 s / 120 s on 8192³, and 150 s / 150 s on a DF
  `acQ,bdQ->acbd` with a = b = c = d = 96 and Q = 3000. All at N = 14, stock clocks.
- Each group started from a card at ≤ 45 °C idle; the second window of a group followed the
  first.
- NVML power, clock, temperature, clock-event reasons and fan were sampled every 20 ms. The
  energy counter was never read (immerdink-eval findings/energy-20261004.md §2).
- Two mixed calls ran with a 15 s pre-heat: 8192³ at g = 2, and 12288 × 384 × 1040.
- GPU time: about 23 min, ehrenfest FREE throughout, through the quiet gate and the shared
  GPU lock. Raw records and analysis: immerdink-eval `results/followups/hotclock/`; report:
  `findings/followup-hotclock.md`.

**Settled state** (last 30 s; TFLOPS-eq, time per call, NVML SM clock, die temperature):
- 16384³: 37.7 TFLOPS-eq, 233.3 ms, 2251 MHz, 74.4 °C. The hot repeat: 233.8 ms, 2241 MHz.
- 8192³: 35.0 TFLOPS-eq, 31.42 ms, 2280 MHz, 75.1 °C.
- DF: 31.7 TFLOPS-eq, 16.08 ms, 2306 MHz, 75.5 °C.

The cap (reason 0x4) is active throughout, with no thermal reason. The fan ends at 57–63 %,
still rising slowly.

- **Heat-up:** +0.15 … +0.22 % time per °C, clock −3.4 … −5 MHz per °C. From the first seconds
  (60–64 °C) to settled that is +1.3 … +2.7 %; from the calibration windows' 67–74 °C
  (end-of-window readings) it is ≈ 1–1.8 %.
- **~1950 MHz is not reproduced.** It was most likely a single NVML read after a loop: the
  16384³ hot window's post-loop read was 2085 MHz against a 2251 MHz mean.

**Shipped SUSTAINED vs settled:** 16384³ −3.7 / −3.9 %, 8192³ −1.8 / −1.4 %, DF −0.9 /
−0.6 %. For 8192³ and DF this is mostly the thermal ≈ 1–1.8 %; for 16384³ roughly half
thermal and half a fit bias of the unscaled form: it fits e_tc 0.650 pJ/op (vs 0.691
clock-scaled) and e_byte 335 pJ/B (vs 296), so a tensor-core-heavy call gets too little
power, the solved clock is too high and the prediction is optimistic.
- Hold-out caveat: none of the six long windows is in the fit, but 8192³ N = 14 is a
  calibration shape and the DF call is in the `u_tdf` family; only 16384³ is a new shape.
- A clock-scaled e_byte, fitted on the same 14 calibration windows (e_tc 0.691 pJ/op,
  e_byte 296 pJ/B × f/f_ref, γ = 1), predicts the six settled windows with a median |error|
  of 0.72 %, against 1.58 % shipped (16384³ −1.8 %). At γ = 2 it is 0.30 %.
- On the calibration windows' own leave-one-case-out check, the capped-time median goes from
  1.63 to 0.50 % at γ = 1.
- A leakage term a·(T − 70) adds about +0.8 % to the settled predictions (a ≈ 1.5–2 W per °C).
- Mixed calls are neutral: 8192³ g = 2 −0.4 → −0.2 %. 12288 × 384 × 1040 −7.7 → −9.2 %; that
  call is dominated by the time model's −7 … −10 % on this shape (§8).

**Proposal** (DESIGN §6c.7): `power.byte_f_pj`, the SM/L2-side energy per modelled DRAM
byte (memory clock fixed at 10251 MHz); host API 11 (followup-hygiene takes 10); optional
format-2 key (absent = 0); v9/v10 callers get `energy_byte_pj := byte_pj + byte_f_pj`; fit
with byte_pj fixed (collinear) and tc, alu, byte_f free. No THERMAL regime. **Decided**
(independent reviewer, 2026-10-06): Q1 split key with byte_pj fixed, Q2 keep γ = 1, Q3 no
THERMAL regime with the drift documented and the calibration windows on `--sustain-warm-s
60`. The form comparison and γ are redone on those windows in the combined refit, so the
refit needs GPU. Implemented in v1.15 (§9.1).

### 9.1 Implementation and the 60 s pre-heat windows (2026-10-06, branch `hotclock`; DESIGN v1.15 §6c.7)

- Host API 11: `imd_perf_calib_t.energy_byte_f_pj`, optional `power.byte_f_pj` (absent = 0,
  bit-identical predictions, written only when non-zero), the folded view for v7–v10-sized
  structs, `perf_fit.py --power-form scaled` (default; `--byte-pj`, γ = 1). Tests:
  `host_perf` suite v1.15, `host_c_api`.
- New windows: `tools/calibrate.sh --sustained-only` (sweep_sustained.txt with
  `--sustain-warm-s 60 --sustain-trace-s 2`), 14 windows, about 20 min of GPU including the
  quiet waits, ehrenfest FREE. Raw: immerdink-eval
  `results/followups2/hotclock/sustained-20261006-1305.jsonl`; analysis `compare.py` /
  `compare.log`; report `findings/hotclock-implementation.md`.
- The windows end at 73–77 °C (median 75 °C; the 15 s windows of 2026-10-05: 67–74, median
  73). Capped calls are 0.8–1.5 % slower than in those windows (8192³ N = 14: 31.40 vs
  31.02 ms, 2282 vs 2312 MHz; 12288³: 100.69 vs 99.23 ms).
- Fit on the new windows, scaled form, γ = 1, byte_pj 0: e_tc 0.707 pJ/op, e_byte^f 308 pJ/B;
  pipeline LOO capped time 0.64 % (unscaled: 1.98 %); settled hold-out (six 2–4 min windows)
  median / max 0.69 / 1.0 % (unscaled: 0.62 / 2.9 %, with 16384³ −2.7 / −2.9 %). γ = 2 is
  worse on every check (1.37 % LOO, 1.01 % settled). γ = 1 and byte_pj = 0 confirmed.
- No refit shipped here: the combined refit (§10) uses the N ≤ 14 windows of this run.

## 10. Combined refit: bf-eps kernels, eps path retired, clock-scaled byte energy (2026-10-06, branch `followups-combined`; DESIGN v1.14 §3.4.1, v1.15 §6c.1 / §6c.7)

`followups-combined` = followup-integration + `bf-eps` (v1.14, branch-free ε pack emitter,
docs/NOTES-BFEPS.md) + `hotclock` (v1.15 §6c.7, `power.byte_f_pj`, host API 11; renumbered
from its draft "v1.14"). With the v1.14 kernels the pack time no longer depends on ε, so the
model's eps path is retired (DESIGN §6c.1 "The eps path — retired": `op_eps` features 0,
`eps_fraction` validated and ignored, `eps_s` / `eps_fraction` out 0, `op_eps` keys optional,
format 2 kept; the six `--data dominant` calibration cases dropped, the three hold-out ones
kept as a check). Then one recalibration of everything.

**Data** (RTX 4090, stock clocks, quiet gate and shared GPU lock, `ehrenfest_busy.sh` FREE;
profiling `imd_bench` at baec447; `tools/calibrate.sh --validate`, 14:15–14:52, 37 min of wall
time including the bench build and quiet waits):
- calibration sweep 14:15–14:36, 434 timed calls; cache sweep fresh (14:36–14:37);
- sustained windows **reused** from the `hotclock` run (2026-10-06 13:05, 60 s pre-heat) but
  **only the 9 windows with N ≤ 14**: they ran the pre-v1.14 pack emitter, which has no eps
  work at N ≤ 14 for typical data with these k (and the large capped windows are K4-bound),
  while the 5 N = 16 windows had the old eps path in their packs and would bias the power fit.
  Offline check on the same calibration records: 14 windows → e_tc 0.700 pJ/op, e_byte^f 305
  pJ/B, f_ref 2430 MHz; 9 windows → 0.701, 298, 2455 MHz. Recollecting the N = 16 windows
  (≈ 7 min) would have overrun the 35 min GPU budget. [review 2026-10-06] **All 14** hotclock
  windows (not only the 5 at N = 16) ran commit 0787164, before the v1.14 emitter (a491220)
  and the v1.15 eps retirement, so the next sustained run recollects **all 14** on the
  current library. Dropping the N = 16 windows also removed u_rbigk and u_tcc, the only
  big-k and contraction windows: e_byte^f now rests mostly on r_mm and u_tdf (weakly
  identified; the uncapped r_mm power is 248 W predicted vs 449 W measured, −45 %, in every
  fit including the 14- and 9-window offline fits);
- hold-out 14:37–14:52, 178 calls (same sweep as §8 plus the comment on the dominant cases).
- Raw records, logs and scripts: immerdink-eval `results/followups2/combined/`.

**Fit** (shipped as `src/host/calib/rtx4090_sm89.calib`, version baec447, host API 11):
- no `op_eps` keys (fitted 0: all-zero feature columns); `kf.chain` 23.9 → 11.1 ns, `pack.op`
  0.226 → 0.126 ps/op (the old θ carried part of the eps work in the plain terms);
- power, scaled form (§6c.7), γ = 1, byte_pj held at 0: e_tc 0.6498 → **0.708** pJ/op,
  e_byte 335 → **0**, e_byte^f **301** pJ/B, f_ref (`clock_eff_mhz`) 2455 → 2480 MHz. Sustained
  leave-one-case-out time error: median 0.8 % (capped windows −1.4 … +0.8 %: u_sq2048 −1.4 %,
  the others −0.6 … +0.8 %; the uncapped 512³ / 1024³ / r_mm windows +7.5 / −11.5 / +4.8 %, as
  before: they run above f_ref). **Not an independent figure** [review 2026-10-06]: f_ref was
  chosen by minimising this same leave-one-out median over 17 candidates (2405–2805 MHz in
  25 MHz steps), and the scan is not monotone near the minimum (2530: 0.88 %, 2505: 0.97 %,
  2480: 0.78 %, 2455: 1.07 %). The **capped clock is predicted 2–4 % high** (leave-one-out:
  8192³ 2331 vs 2282 MHz measured, 12288³ 2350 vs 2266; in-sample 2325 / 2335) and the time
  error is small only because the per-call time model's bias cancels it;
- `rank.tie_ratio` 1.078 → 1.074; bands (5 / 95 hold-out quantiles, cross-validated coverage
  82 %): call < 100 µs 0.897–1.213, 100 µs – 1 ms 0.881–1.104, ≥ 1 ms 0.932–1.046.
- `k23_pack` occupancy: the occupancy query measures the W = 3 instantiation (117 registers,
  2 CTAs / SM); W = 2 has 84 registers but the same 2 CTAs / SM (40 768 B static shared memory
  per CTA), so `kernel.pack = 2 …` holds for every N. K4 features are unchanged, so
  `k4_select` was not re-checked.

**Hold-out, same 178 calls, BURST, REPEAT** (median / p90 / max |error| of the call time).
"Before (as shipped)": followup-integration's library and calibration (its eps path on);
"before θ, eps ignored": the old θ in this library; "after": the shipped refit.

| subset | measured call | calls | before (as shipped) | before θ, eps ignored | after | target |
|---|---|---|---|---|---|---|
| all | < 100 µs | 56 | 9.6 / 31.3 / 35.2 % ✗ | 6.1 / 12.5 / 22.9 % | 6.4 / 14.5 / 28.8 % | 15 / 30 % |
| all | 100 µs – 1 ms | 58 | 6.5 / 29.4 / 43.0 % ✗ | 3.7 / 14.7 / 22.9 % | 2.9 / 11.2 / 18.0 % | 10 / 20 % |
| all | ≥ 1 ms | 64 | 3.2 / 11.1 / 24.2 % | 2.0 / 7.7 / 15.2 % | 2.1 / 7.3 / 15.7 % | 5 / 15 % |
| N = 15–17 | < 100 µs | 18 | 28.9 / 33.1 / 35.2 % | 3.8 / 6.9 / 7.2 % | 2.9 / 8.7 / 12.8 % | |
| N = 15–17 | 100 µs – 1 ms | 16 | 24.6 / 38.6 / 43.0 % | 6.2 / 16.5 / 19.1 % | 2.3 / 11.3 / 13.4 % | |
| N = 15–17 | ≥ 1 ms | 22 | 10.4 / 16.5 / 24.2 % | 2.6 / 7.3 / 15.2 % | 2.6 / 6.6 / 15.7 % | |

Per phase on calls ≥ 1 ms (median / p90, as shipped → after): KL A 8.3 / 32.4 → 2.1 / 12.2 %,
KL B 6.2 / 29.5 → 3.4 / 14.9 %, K2 5.0 / 17.2 → 4.1 / 11.3 %, K3 8.1 / 19.0 → 14.9 / 23.5 %,
K4 5.0 / 13.0 → 4.8 / 13.1 %, K5 4.5 / 19.2 → 3.4 / 18.6 %, KF 8.4 / 16.3 → 4.6 / 11.4 % (4 calls).

- The model as shipped before this branch misses the < 1 ms targets on the v1.14 kernels: its
  eps terms price N ≥ 15 packs up to 2× too slow. **Almost all of the improvement comes from
  retiring the eps terms, not from the refit** [review 2026-10-06]: the old θ with eps ignored
  (the fair baseline, `val-before.log`) already meets every target. Against that baseline the
  refit is
  - **worse on < 100 µs calls**: p90 12.5 → 14.5 %, max 22.9 → 28.8 % (N = 15–17: p90 6.9 →
    8.7 %, max 7.2 → 12.8 %);
  - neutral on ≥ 1 ms calls (median / p90 2.0 / 7.7 → 2.1 / 7.3 %);
  - better on 100 µs – 1 ms calls (3.7 / 14.7 → 2.9 / 11.2 %; N = 15–17 6.2 / 16.5 → 2.3 /
    11.3 %).
  Per phase the refit is worse for **KF at N = 8** (measured / predicted on the hold-out 1.11
  → 1.195; 1.22 in-sample, against 0.99 at N = 14 and 0.96 at N = 20: the v1.14 emitter's
  per-element |a'| → W-word cost does not scale with the residue count, and `kf.op` / `kf.row`
  fitted to 0 while `kf.chain` went 23.9 → 11.1) and for **K3** (B pack at g < N; measured /
  predicted 1.07–1.20 on the hold-out, 1.13–1.20 in-sample, old θ 1.04–1.15; `pack.op` 0.226
  → 0.126 and `band.pack` lo 0.98; K2 and K3 share the PACK class). The call totals are within
  target. W is not a model feature, and W = 3 is barely validated out of sample (the hold-out
  stops at N = 17; 17 of its pack-phase rows are W = 3): add N = 18–20 hold-out cases.
- **Dominant data no longer matter** (measured): `w_d800` vs `w_s800` (800³) 0.0918 vs 0.0922
  ms at N = 15 and 0.1052 vs 0.1034 ms at N = 16; `w_dr1` vs `w_r1` (1200 × 400 × 700) 0.0798
  vs 0.0781 and 0.0819 vs 0.0797 ms (+2–3 %, within tiny-call noise). The model predicts the
  same time for both; `v_dom` (1536² × 6144) is within +0.8 / −0.4 %.
- Examples (TFLOPS-eq measured / predicted; time measured / predicted): 8192³ N = 14 36.58 /
  36.95 (30.05 / 29.76 ms); 5000³ N = 16 27.97 / 27.75 (8.94 / 9.01 ms); 1536² × 6144 dominant
  N = 15 22.25 / 22.09 (1.303 / 1.313 ms); 800³ N = 16 9.90 / 9.59 (103.4 / 106.7 µs). Largest
  remaining errors: 1024³ N = 14 +16 / +18 % (predicted slow; 1024³ N = 8 −6 %) and the tiny CC call `v_cc12` N = 8 −29 % (45.8
  vs 32.6 µs, the < 100 µs maximum).
- SUSTAINED examples (60 s pre-heat windows): 8192³ N = 14 35.01 / 34.95 TFLOPS-eq (31.40 /
  31.46 ms, 2282 / 2325 MHz); 12288³ N = 14 36.85 / 36.86 (100.69 / 100.68 ms).

## 11. Model fix: KF element chain, PACK L2-hit re-reads, f_ref rule, full refit (2026-10-06, branch `model-fix`; DESIGN v1.17 §6c, host API 12)

The open items of §10 (KF N-dependence, K3 predicted fast, all 14 sustained windows on the
current library, N = 18–20 hold-out cases, an overfit f_ref) after the open-items analysis
(evaluation workspace `findings/openitems-model-analysis.md`).

**Design** (DESIGN v1.17 §6c.1 / §6c.2; decided here, design question below):
- **`kf.elem`** (KF slot 4): the busiest SM's per-element chain, `kf.chain` with u = E (one
  launch-shape helper for both). Each moduli part loads, quantizes and reduces its tile once,
  whatever its residue count. Not clock-bound.
- **`pack.dram`** (PACK slot 3): the bytes that are not L2-hit re-reads. A K2 / K3 re-read
  hits L2 when the call's group working set S = Σ_y (8 rows_y k + g pad_y k̂) ≤ L_hit
  (`warm.pack_l2_bytes`, set by `perf_fit.py` to 0.8 × nominal L2 = 60.4 MB; the 4090 data
  bound the step to [57, 77) MB). A hit byte costs `pack.byte`, every other byte
  `pack.byte + pack.dram`. Not clock-bound.
- **Additive, format 2 kept**: every existing key keeps its meaning, the three new keys are
  optional (absent = 0, predicts bit-identically) and API ≤ 11 rejects a file with them, so no
  file is read with features it was not fitted with. The `op_eps` keys are retired (accepted,
  discarded). Host API 12: `imd_perf_calib_t.pack_l2_hit_bytes`.
- **Not adopted**: the analysis's KF read / write byte split (residue writes ~1/3 of a read
  per byte). It needs a seventh KF slot or retiring `kf.op` / `kf.row`, i.e. a format bump. On
  the combined refit's records it moved no hold-out statistic by more than 1 point over the
  two features above.
- **f_ref**: the vertex of a least-squares parabola through the leave-one-out scan within
  ±100 MHz of its argmin (was the argmin), and the **nested** leave-one-out figure (f_ref
  chosen inside each fold) is printed and reported.
- **Latent bug fixed**: a `--l2 flush` record's phases are REPEAT state (the phase reps are
  not flushed); `perf_fit.py` now raises them to COLD and `perf_validate.py` compares them
  in REPEAT. No record used `--l2 flush`.
- **Offline check first** (CPU only, the combined refit's records): the implementation
  reproduces the analysis's `kf1p1` variant exactly (hold-out 4.2 / 12.5 / 20.1 · 2.8 / 10.1
  / 15.2 · 2.1 / 7.2 / 15.7 %).

**Data** (RTX 4090, stock clocks, quiet gate and shared GPU lock, `ehrenfest_busy.sh`
FREE; profiling `imd_bench` at 7e5ef5d; `tools/calibrate.sh --validate`, BENCH_RETRIES = 1):
- calibration sweep 17:46–18:07 (442 timed calls; new: `c_l2s1472` N = 8 / 14 / 20 and
  `c_l2s1792` N = 14 at g = 4, `c_k3hit` / `c_k3miss` at g = 2);
- **all 14 sustained windows recollected** on the current library, 60 s pre-heat
  (18:07–18:25);
- cache sweep (18:25–18:27);
- hold-out 18:27–18:42, 189 calls (new: `v_n18a` … `v_n18d`, N = 18–20).
- GPU time: 56 min of sweep wall time (including the per-case quiet gate and warm-up), over
  the 45 min budget: the calibration sweep took 21 min as in §10 although retries were off.
- Raw records, logs, scripts and comparisons: immerdink-eval `results/openitems/model-fix/`.

**Fit** (shipped as `src/host/calib/rtx4090_sm89.calib`, version 7e5ef5d, host API 12):
- KF: launch 5.54 µs, byte 0.813 ps/B, **elem 534 ns per tile step**, op / row / chain 0.
- PACK: launch 3.38 µs, byte 0 (an L2-hit re-read is hidden), **dram 0.844 ps/B**
  (1.18 TB/s), chain 0, wave 3.87 µs.
- The PACK fit error by L_hit (diagnostic): 0: 16.9 / 30.7 %, 0.6 × L2 (45 MB): 6.8 / 19.4 %,
  0.7–1.0 × L2: 5.3 / 11.0 % (median / p90). The new cases put the step above 45 MB; nothing
  in the sweep lies between 52 and 77 MB, so 0.8 is not refined.
- In-sample phases: kf 6.1 / 24.9 %, pack 5.3 / 11.0 % (median / p90).
- Power (scaled form, γ = 1, byte_pj 0, 14 windows): e_tc 0.721 pJ/op, e_byte^f 289 pJ/B.
  The scan (2405–2805 MHz) has its argmin at 2505 MHz (1.11 %), with 2480: 1.22 % and 2455:
  1.30 %. **f_ref = 2491 MHz** (parabola vertex). **Nested leave-one-case-out median |t err|
  1.19 %** (the folds chose 2405–2500 MHz), the honest figure; the scan's minimum (1.11 %) is
  not.
- Sustained LOO at 2491 MHz: windows predicted capped −1.3 … +1.6 % (u_rbigk +1.6, u_sq2048
  −1.3); the windows predicted below the cap u_s512 −2.6 / +9.7 % (N = 14 / 16), u_s1024
  N = 8 −6.5 %, u_rmm +6.4 %; u_s1024 N = 16 (predicted capped) +5.6 %. The **capped clock is
  still predicted 0.8–3.4 % high** (12288³ 2335 vs 2258 MHz +3.4 %, 8192³ 2328 vs 2274, DF
  2392 vs 2324; u_s1024 N = 16 is 4.3 % low, 2416 vs 2525), and the time error stays small
  because the per-call time model's bias cancels it, as in §10.
- [review 2026-10-06] **Power is predicted low on every small high-clock window, not only on
  r_mm.** All 14 windows ran at the cap (447–450 W), u_s512 and u_s1024 included (at
  2525–2775 MHz: they are not "uncapped", their clock is high because their power per clock
  is low). Predicted: u_s512 345 / 371 W (−23 / −17 %), u_s1024 N = 8 423 W (−6 %), u_rmm
  258 W (−43 %). A clock-dependent power term for small calls is missing, as well as the r_mm
  one.
- `rank.tie_ratio` 1.074 → 1.060; bands (5 / 95, cross-validated coverage 83 %): call < 100 µs
  0.938–1.191, 100 µs – 1 ms 0.891–1.055, ≥ 1 ms 0.932–1.024, kf 0.790–1.325, pack 0.927–1.171.

**Hold-out, the same 189 new calls for each calibration, all evaluated with this library**,
BURST, REPEAT (median / p90 / max |error| of the call time, %). (2) = the reviewer's fair
baseline: the pre-refit θ (18baccc) with ε ignored. (1) = the shipped combined refit (§10).
Targets 15 / 30, 10 / 20, 5 / 15 %.

| subset | measured call | calls | (2) old θ, ε ignored | (1) shipped refit | new |
|---|---|---|---|---|---|
| all | < 100 µs | 61 | 5.8 / 13.4 / 21.7 | 5.5 / 13.7 / 27.7 | **3.8 / 12.9 / 23.2** |
| all | 100 µs – 1 ms | 60 | 3.5 / 14.1 / 21.3 | 2.7 / 10.5 / 16.4 | **2.6 / 9.6 / 13.9** |
| all | ≥ 1 ms | 68 | 2.1 / 7.7 / 15.8 | 2.2 / 7.4 / 16.2 | **2.2 / 6.6 / 16.8** |
| N ≤ 14 | < 100 µs | 38 | 6.0 / 18.8 / 21.7 | 7.5 / 19.4 / 27.7 | 5.2 / 14.9 / 23.2 |
| N ≤ 14 | 100 µs – 1 ms | 42 | 2.8 / 13.4 / 21.3 | 3.0 / 10.3 / 16.4 | 2.5 / 9.9 / 13.9 |
| N ≤ 14 | ≥ 1 ms | 42 | 1.9 / 7.8 / 11.3 | 1.5 / 7.9 / 10.1 | 1.9 / 6.8 / 8.9 |
| N = 15–17 | < 100 µs | 18 | 4.8 / 7.5 / 9.9 | 2.1 / 10.0 / 13.7 | 2.6 / 9.0 / 10.5 |
| N = 15–17 | 100 µs – 1 ms | 16 | 6.5 / 16.0 / 17.9 | 2.6 / 10.8 / 12.3 | 2.8 / 8.2 / 9.5 |
| N = 15–17 | ≥ 1 ms | 22 | 2.6 / 7.3 / 15.8 | 2.6 / 6.6 / 16.2 | 2.3 / 6.3 / 16.8 |
| N = 18–20 | < 100 µs | 5 | 9.4 / 11.4 / 12.5 | 3.8 / 7.4 / 8.8 | 2.7 / 11.9 / 17.2 |
| N = 18–20 | 100 µs – 1 ms | 2 | 0.5 / 0.7 / 0.7 | 1.4 / 1.7 / 1.8 | 1.4 / 1.7 / 1.8 |
| N = 18–20 | ≥ 1 ms | 4 | 1.6 / 3.1 / 3.2 | 2.1 / 2.4 / 2.5 | 1.3 / 2.4 / 2.5 |

[review 2026-10-06] **Two caveats on this hold-out:**
- It is not fully independent of the design: the open-items analysis chose its features by
  comparing variants on the same hold-out shapes (the 178 calls other than N = 18–20). Only the
  11 N = 18–20 calls are shapes nobody had looked at, and on them the < 100 µs p90 / max got
  worse than both baselines (11.9 / 17.2 % vs 11.4 / 12.5 % for (2) and 7.4 / 8.8 % for (1)).
- The L2-hit side of the PACK rule is checked in-sample only: no hold-out pack call has S
  between about 5 and 85 MB (v_s512, 4.2 MB of inputs, is the only one below 85 MB; the next
  is v_n18a 2304³ at 85 MB of inputs). The K3 gain (13.3 / 22.4 → 4.9 / 14.7 %) comes
  entirely from cases that miss L2.

Against baseline (2), per size class:
- < 100 µs: median and p90 better; max +1.5 points (`z_s200` N = 14, −23.2 vs −19.9 %: K4 on a
  tiny shape, 6.3 µs predicted vs 9.2 measured under every calibration).
- 100 µs – 1 ms: better on every statistic.
- ≥ 1 ms: p90 better, median +0.1, max +1.0 point (`v_n15d` N = 16, 768² × 100000: −16.8 vs
  −15.8 %, a K4 / K5 call at −16 % under all three calibrations; the new features do not
  touch it).
- **Noise**: the same hold-out calls measured in the §10 run and in this run differ by
  median / p90 / max 1.3 / 6.5 / 12.0 % below 100 µs, 0.3 / 1.1 / 4.9 % at 100 µs – 1 ms and
  0.2 / 0.5 / 1.6 % at ≥ 1 ms. The < 100 µs max is within that noise. The ≥ 1 ms max is one
  call and within the run-to-run maximum, but above the p90; it is borderline.
- The N = 18–20 subset is 11 calls. Its < 100 µs max is `v_n18d` N = 20 (352³, 31.7 µs):
  −17.2 % new vs −7.0 % (2). KF there is 15.8 µs measured and 10.9 µs predicted: the E = 4 KF
  path at N = 20 grows with N (11.3 µs at N = 18), which `kf.elem` does not model and
  `kf.chain` (fitted 0) no longer carries. W = 3 KF on tiny shapes is the remaining gap.

Which features help, the same new records (offline refits, `--drop`; hold-out median / p90 /
max %):

| variant | < 100 µs | 100 µs – 1 ms | ≥ 1 ms |
|---|---|---|---|
| both (shipped) | 3.8 / 12.9 / 23.2 | 2.6 / 9.6 / 13.9 | 2.2 / 6.6 / 16.8 |
| without `kf.elem` | 5.8 / 15.1 / 29.0 | 2.7 / 10.4 / 16.5 | 2.2 / 7.1 / 16.8 |
| without `pack.dram` | 3.8 / 12.9 / 23.2 | 2.6 / 9.6 / 13.9 | 2.5 / 7.1 / 16.8 |
| neither (a plain refit) | 5.6 / 15.0 / 29.0 | 2.7 / 10.4 / 16.5 | 2.5 / 8.0 / 16.8 |

Each feature helps where it should (`kf.elem` the small calls, `pack.dram` the ≥ 1 ms
calls), and a plain refit on the new data reproduces the §10 regression below 100 µs. So the
gain over (2) is the features, not the new data.

**Per phase, hold-out, all calls** (median / p90 |error|, %; (2) → (1) → new): KF 9.6 / 27.8
→ 14.5 / 33.6 → 9.8 / 25.1; K2 3.4 / 8.9 → 5.4 / 12.1 → 4.4 / 7.9; **K3 9.6 / 18.9 → 13.3 /
22.4 → 4.9 / 14.7**; KL A 4.3 / 14.0 → 3.0 / 12.3 → 3.4 / 12.7; KL B 3.7 / 11.9 → 4.4 / 14.0
→ 3.6 / 13.8; K4 4.8 / 21.2 → 4.6 / 21.4 → 4.3 / 20.5; K5 4.1 / 16.2 → 4.0 / 16.5 → 4.1 /
16.5. Calibration set (in-sample for the new fit only): K2 2.8 / 53.1 → 7.2 / 43.3 → 5.0 / 9.8,
K3 15.0 / 32.8 → 17.6 / 28.6 → 5.6 / 12.6, KF 14.8 / 38.8 → 11.0 / 31.1 → 7.5 / 26.4.
[review 2026-10-06] The calibration set's **calls** at ≥ 1 ms got slightly worse in-sample
than under (1): 3.0 / 9.5 / 16.1 → 3.5 / 9.1 / 17.8 % (median / p90 / max; 161 calls; (2) 3.2
/ 10.0 / 16.5 %), while 100 µs – 1 ms improved (3.9 / 11.6 / 32.8 → 2.1 / 8.8 / 27.1 %). The
fit minimises per-phase error, not call error, so a better phase fit does not have to
improve the calls.

**Measured / predicted by N** (median per N; (2) / (1) / new):

| N | KF hold-out | KF calibration set | K3 hold-out | K3 calibration set |
|---|---|---|---|---|
| 8 | 1.107 / 1.188 / 1.126 (27) | 1.089 / 1.199 / 1.082 (81) | 1.042 / 1.066 / 1.019 (8) | 1.091 / 1.136 / 1.056 (29) |
| 14 | 0.920 / 1.011 / 1.053 (23) | 0.893 / 0.981 / 0.982 (51) | 1.086 / 1.139 / 1.040 (8) | 1.084 / 1.136 / 1.036 (29) |
| 15 | 0.958 / 1.131 / 1.075 (10) | 0.887 / 1.039 / 1.004 (9) | 1.129 / 1.173 / 1.065 (1) | |
| 16 | 0.937 / 1.122 / 1.046 (12) | 0.870 / 0.994 / 1.009 (40) | 1.143 / 1.193 / 1.094 (6) | 1.133 / 1.184 / 1.072 (5) |
| 17 | 0.927 / 1.111 / 1.055 (7) | 0.877 / 1.021 / 1.064 (10) | 1.124 / 1.174 / 1.062 (1) | 1.211 / 1.261 / 1.126 (1) |
| 18 | 0.831 / 0.951 / 1.036 (2) | | 1.086 / 1.133 / 1.041 (1) | |
| 20 | 0.941 / 1.070 / 1.212 (2) | 0.903 / 0.952 / 0.999 (31) | 1.109 / 1.162 / 1.063 (1) | 1.004 / 1.086 / 0.986 (11) |

(phase counts in parentheses). The KF trend from N = 8 to 20 is gone on the calibration set
(1.08 → 1.00 instead of 1.20 → 0.95). On the hold-out KF stays 1.13× slow-predicted at N = 8
(the 1024³-like REPEAT shapes, §3 of the analysis: the cache term's REPEAT saving is too
large there) and 1.21 at N = 20 (2 phases, the tiny E = 4 case above). K3 goes from 1.07–1.19
to 1.02–1.09; the remaining +4–9 % is in the miss cases.

**Remaining** (not done here):
- the KF roofline (latency and bandwidth terms overlap; max instead of sum) and a per-phase
  cap of the cache term's REPEAT saving (analysis §3);
- W = 3 KF on tiny E = 4 shapes (`v_n18d` N = 20);
- K4 on tiny shapes (`z_s200`, `v_s320` N = 14, −17 to −23 %);
- the capped-clock bias (0.8–3.4 % high) and the power gaps (r_mm −43 %, the small
  high-clock windows −6 … −23 %), which need memory-bound and small windows at several clocks
  (analysis §4) before a power term is chosen;
- refine L_hit with a case between 57 and 77 MB, a short-k case with large m·n (4096² × 512,
  g = 4) and a pre-packed operand at g < N; S is a proxy without K4's planes (DESIGN §6c.1
  [review]), and counting them is a feature change;
- `pack.byte` fits to 0 and the cost moves to `pack.launch` (0.91 → 3.38 µs) and `pack.wave`
  (5.58 → 3.87 µs): launch, wave and byte are not separately identifiable; price hits at a
  prior (e.g. `kl.l2byte`) or a fitted rate of their own;
- L_hit is not identifiable within the calibration set (the same PACK fit for 0.7–1.0 × L2);
- `kf.chain` fits to 0, so KF no longer grows with N on tiny E = 4 shapes (`v_n18d` −17 %);
- the f_ref scan stops at 2405 MHz and the u_rmm fold chose exactly that: extend it down.
- "Bit-identical to API 11" (absent keys vs 0) is tested within this library only, not
  against an API-11 build (low risk: the changed terms all have parameter 0 there).

**Design question for the reviewer**: keep calibration format 2 with additive optional
keys (DESIGN §6c.2 "Why format 2 is kept", §6c.5 amended), or bump to format 3 as the
analysis suggested? The additive form needs no misreading rule; a format 3 would only be
needed for the not-adopted KF read / write split.

## 12. Minor model items: cache state, PACK set with planes, busy power and boost clock (2026-10-06, branch `perf-minor`; DESIGN v1.18 §6c.8, host API 13)

The "Remaining" items of §11 after two offline analyses (evaluation workspace
`findings/perfminor-time.md`, `findings/perfminor-power.md`). Raw records, logs, scripts and
every comparison: immerdink-eval `results/perfminor/impl/`.

**What was adopted, and what not (DESIGN §6c.8 has the reasons and the formulas):**

| analysis item | finding | here |
|---|---|---|
| KF overlap / roofline (1024³ N = 14 +13.9 %) | refuted: every roofline / soft-max form fits worse than the sum; the error is the cache term's (a flushed COLD is *clean*, a REPEAT call writes back the previous call's dirty lines; the REPEAT saving has a per-call part) | **adopted:** `warm.call_ns` (c), `warm.clean_byte` (ω), state `IMD_PERF_CACHE_CLEAN`; COLD keeps its meaning (evicted by other, dirty work) and stays the default |
| KF read / write split | ≤ 0.8 points per class over the cache term alone, KF phase and open-items records slightly worse | **not adopted:** `kf.row` is non-zero in format-1 files (v6 vector 1.52 ns), so retiring it needs a format bump |
| PACK set S (proxy without K4's planes) | wrong hit / miss on 3 of 4 new non-square cases | **adopted:** `warm.pack_l2_planes = 1`, S += g b m n̂ (2 k_chunks − 1), L_hit = 1.1 × L2 = 83 MB |
| E = 4 KF at high N | a per-modulus term fits ≈ 0; parity pattern at the 1.024 µs timer step | **not adopted** (no term); 352³ N = 14–20 moved into the hold-out (`v_n18d`, 30 phase reps) |
| ≥ 1 ms calibration calls (streaming K2 7–9 % slow) | `pack.byte` 0, cost in `pack.wave`; a fixed hit price helps calls, hurts the PACK phase p90 | **not adopted** (no prior); the four S cases in the calibration sweep instead (still open, below) |
| power: clock evaluated at f_ref, missing busy power | the uncapped calls run at 2625–2790 MHz; DIRECT draws 349 W | **adopted:** `power.act_w`, `power.act_gap_us`, `power.dram_pj`, `power.clock_boost_mhz`, γ = 2.5 as a prior (`perf_fit --power-form busy`, default) |
| bench: sustained DIRECT variant | `--methods sustained --mode direct` wrote no " mode=direct" | **fixed** in `imd_bench.cu` |

**Design question decided here** (DESIGN §6c.8, "Time model: the cache state"): the default
COLD is "inputs evicted, L2 dirty with other work" (a call inside an ehrenfest sequence), not the
bench's "evicted and clean"; the clean credit applies only to `CLEAN` (flushed bench records,
method pair's COLD). **Reviewer question Q1 of the power analysis, decided:** `clock_mhz` out
reports the clock used (NVML scale, up to the boost clock), host API 13; with the boost key 0 it
is the v1.17 value. Both are open for the independent reviewer.

**Offline check first** (CPU only, the model-fix records, `results/perfminor/impl/v/`): with the
shipped calibration the v1.18 library reproduces the perfminor-time baseline exactly (hold-out
statistics identical); a refit with the v1.18 cache term reproduces the analysis's `v8p_clean`
(hold-out 4.2 / 12.7 / 24.2 · 2.5 / 6.4 / 12.7 · 2.2 / 6.7 / 16.7 %), the planes rule its `v11`
(no hold-out or calibration decision changes); the busy power form on the 25 windows of the
analyses gives e_tc 0.571 pJ/op, e_alu 74 pJ/op, P_act 158 W, e_dram 90 pJ/B, f_ref 2466 MHz,
nested LOO 1.62 % (analysis S6: 0.564, 74.5, 161, 83, 2462, 1.57 %).

**Data** (RTX 4090, stock clocks; every GPU step after `ehrenfest_busy.sh` returned FREE in a
separate step; `tools/calibrate.sh`, BENCH_RETRIES = 1; library ec44099):
- the two sustained windows no earlier run had on the current kernels, `s_k5b` (8192² × 64,
  the second K5-heavy window) and `s_dir` (DIRECT, with the fixed variant), 21:07–21:10;
- one full recalibration `tools/calibrate.sh --validate --sustained FILE` 21:10–22:38: the
  calibration sweep (449 timed calls; new: `c_w4096k512`, `c_skin1280`, `c_l2s1600`, `c_df48`),
  the cache sweep (18 pairs; new: 1024³ N = 8 / 10 / 12, 1536³, 768³), the fit, the hold-out
  (199 calls; new: `v_s1920`, `v_w3584`, `v_df64`, the 352³ N = 14–20 series; 3 hold-out pairs).
- **The sustained windows were reused, not re-measured** (the documented `--sustained FILE`
  path; budget): 26 windows = the 14 model-fix windows (7e5ef5d), 3 of the power analysis + 7 of
  the power-util analysis (684f459; its DIRECT window left out: wrong variant) and the 2 new
  ones. No kernel or bench timing code changed between 7e5ef5d and ec44099 (`git diff` of `src`
  outside `src/host` is empty; the bench diff is the two v1.18 lines: the sustained variant string and the predicted state of `--l2 flush` records). Measuring all 26 again would
  have added ≈ 34 min. `bench/sweep_sustained.txt` lists the 12 new windows (as `s_*`; the
  reused records carry their analyses' names `d_*` / `p_*`).
- **GPU time: 18.7 min** (from each case's quiet-gate pass to the next case: warm-up + case,
  an upper bound that includes waits for the shared GPU lock): 2.4 min for the two windows,
  16.3 min for the recalibration (≈ 10 calibration, ≈ 1 cache, ≈ 5 hold-out). Sweep wall time
  90 min, the rest gate waits while other GPU users ran (up to 6 min per case). Budget 60 min.

**Fit** (shipped as `src/host/calib/rtx4090_sm89.calib`, version ec44099, host API 13):
- Cache: L_w = 0.90 × L2 (68 MB), Δ = 1.29 ps/B, **c = 1.23 µs**, **ω = 0.169 ps/B** (relative
  SSE 0.0136 over 18 pairs; analysis 0.90 / 1.34 / 1.41 µs / 0.193).
- PACK (S with planes, L_hit 83 MB): launch 3.39 µs, byte 0, dram 0.841 ps/B, chain 0.85 ns,
  wave 3.92 µs. The PACK fit error by L_hit (diagnostic, now with the four S cases): 0.8 × L2
  6.0 relative SSE, 1.0 × 1.64, **1.05–1.15 × 0.94**, 1.2 × 1.33: the data bracket L_hit to
  (75.5, 90.6) MB with planes, and 83 MB sits in the flat minimum. In-sample PACK 5.6 / 11.1 %.
  `pack.byte` still fits 0 although the hits now include non-square cases of known bytes.
- KF: launch 6.00 µs, byte 0.707 ps/B, elem 667 ns; in-sample 5.3 / 24.6 %.
- Power (busy form, γ 2.5, 26 windows): e_tc 0.577 pJ/op, e_alu 92 pJ/op, e_byte^f 0, P_act
  154 W, e_dram 95.9 pJ/B, gap 1 µs, boost 2788 MHz (the median of the 6 windows that never hit
  the cap), **f_ref 2484 MHz** (parabola vertex; scan argmin 2480), **nested leave-one-out
  median |t err| 1.62 %** (folds' f_ref 2468–2498 MHz). [review] Refit with the fixed solve
  (below): f_ref **2483 MHz**, e_tc 0.576, e_alu 91.9, P_act 154.1 W, e_dram 95.9 (the power
  NNLS uses the measured times, so only f_ref moves it), nested **1.40 %** (folds' f_ref
  2474–2495 MHz, boost recomputed per fold).
- `rank.tie_ratio` 1.059; bands at the 2.5 / 97.5 quantiles (5 / 95 reached 79.9 %; cross-
  validated coverage 86 %): call < 100 µs 0.898–1.174, 100 µs – 1 ms 0.894–1.079, ≥ 1 ms
  0.924–1.094.

**Hold-out, the same 199 calls for both calibrations, the v1.18 library**, BURST, REPEAT
(median / p90 / max |error| of the call, %; targets 15 / 30, 10 / 20, 5 / 15 %; run-to-run noise
of §11: 1.3 / 6.5 / 12.0, 0.3 / 1.1 / 4.9, 0.2 / 0.5 / 1.6):

| subset | measured call | calls | shipped (v1.17, model-fix) | **v1.18** |
|---|---|---|---|---|
| all | < 100 µs | 65 | 3.9 / 10.8 / 15.8 | **3.7 / 12.2 / 19.0** |
| all | 100 µs – 1 ms | 65 | 2.7 / 10.3 / 15.7 | **2.7 / 6.9 / 13.4** |
| all | ≥ 1 ms | 69 | 2.4 / 6.6 / 16.0 | **2.6 / 6.4 / 16.4** |
| N ≤ 14 | < 100 µs | 39 | 4.3 / 11.0 / 15.8 | 3.3 / 12.1 / 19.0 |
| N ≤ 14 | 100 µs – 1 ms | 47 | 2.9 / 10.5 / 15.7 | 2.7 / 8.0 / 13.4 |
| N ≤ 14 | ≥ 1 ms | 43 | 2.0 / 6.7 / 9.1 | 2.2 / 6.5 / 9.2 |
| N = 15–17 | < 100 µs | 20 | 2.8 / 8.5 / 12.3 | 4.4 / 11.5 / 13.2 |
| N = 15–17 | 100 µs – 1 ms | 16 | 2.6 / 9.3 / 11.8 | 2.8 / 4.3 / 4.5 |
| N = 15–17 | ≥ 1 ms | 22 | 2.9 / 6.3 / 16.0 | 2.8 / 6.2 / 16.4 |
| N = 18–20 | < 100 µs | 6 | 3.0 / 9.8 / 14.1 | 5.5 / 13.1 / 17.3 |
| N = 18–20 | 100 µs – 1 ms | 2 | 1.2 / 1.4 / 1.5 | 1.3 / 1.5 / 1.6 |
| N = 18–20 | ≥ 1 ms | 4 | 1.6 / 2.7 / 2.8 | 1.7 / 2.7 / 2.7 |

Per phase, hold-out, all calls (median / p90 %, shipped → v1.18): **KF 8.9 / 24.4 → 6.6 / 21.1**,
K2 4.9 / 9.2 → 5.3 / 8.3, **K3 5.2 / 16.3 → 4.2 / 14.0**, KL A 3.7 / 13.1 → 4.1 / 13.4, KL B
3.7 / 14.1 → 3.5 / 14.1, K4 5.1 / 20.9 → 4.9 / 21.6, K5 4.1 / 17.3 → 4.2 / 17.3.
Calibration set (in-sample for v1.18 only): < 100 µs 7.6 / 17.3 / 38.6 → 8.8 / 19.4 / 36.4,
100 µs – 1 ms 2.1 / 9.7 / 26.8 → 2.0 / 9.5 / 23.3, ≥ 1 ms 3.9 / 9.2 / 17.6 → 4.0 / 9.3 / 17.6; K3
7.0 / 15.8 → 5.9 / 12.5, K2 5.4 / 11.6 → 5.2 / 9.5.

Against the shipped calibration, per size class: 100 µs – 1 ms better on every statistic;
< 100 µs median better, p90 +1.4 and max +3.2 points (within the noise 6.5 / 12.0); ≥ 1 ms p90
better, median +0.2 (= the noise 0.2) and max +0.4 (noise 1.6). **[review] That comparison
mixes two record sets** (the v1.17 calibration was fitted on the model-fix records). Against
the like-for-like refit of the v1.17 model on the same new records (`n_v117`, ablation table
below) v1.18 **is** worse beyond the noise in two places: ≥ 1 ms median 1.6 → 2.6 (5× the 0.2
noise; the planes rule, below) and < 100 µs p90 / max 9.8 / 15.2 → 12.2 / 19.0 (the per-call
saving c); the calibration set's < 100 µs (in-sample) goes 7.3 / 17.6 / 33.6 → 8.8 / 19.4 /
36.4. Every adopted part is shipped as a trade-off, not as a free gain: the planes rule fixes
the PACK phases (K3 10.6 / 19.2 → 4.2 / 14.0 %, PACK class SSE 6.0 → 0.94) and the held-out S
cases at the price of the ≥ 1 ms median, and the cache term fixes the mid class (p90 10.4 →
6.9) at the price of the small one. The caveats below say where it is worse.

**The named cases** (call error %, shipped → v1.18):
- **1024³ N = 14** (`v_sq1024`): **+14.9 → +6.3** (vec=resident +15.7 → +7.0); N = 8 −2.2 → +0.3.
  w_s960 N = 15–17 +8.0 … +11.8 → +3.8 … +4.5, v_bsm N = 14 +9.2 → +1.4. **[review] In-sample
  for the cache term, not a hold-out gain:** the cache sweep has `x_s1024` pairs at N = 8 / 10 /
  12 / 14 / 16, the same shape, N and REPEAT call as `v_sq1024`, and the term was designed
  around this case; part of the 100 µs – 1 ms p90 gain (10.3 → 6.9) comes from such
  1024³-like shapes too (v_sq1024, w_s960, v_bsm). The genuinely held-out cache evidence is
  the three `v_x*` pairs below.
- **`v_n18d`** (352³, 30 phase reps): N = 14 / 16 / 17 / 18 / 19 / 20: −5.5 / −5.2 / −4.8 / −14.1 /
  −5.4 / −4.0 → −9.1 / −8.6 / −8.2 / −17.3 / −8.9 / −7.6. Worse by ≈ 3.5 points at every N (the
  per-call saving c on a 25 µs REPEAT call, below). Re-measured quiet, the validation run's
  15.8 µs KF at N = 20 is not there (27.4 µs call); N = 18 stays the outlier (KF at E = 4).
- **S rule on shapes not used to choose L_hit:** `v_w3584` (3584² × 512 g = 4: a hit by the v1.17
  proxy, a miss with planes) N = 8 / 14: −9.9 / −7.5 → −4.9 / +0.7; `v_df64` (DF 64⁴, Q = 768) −3.8
  / −0.3 → −2.7 / +0.5; `v_s1920` (hit either way) −4.1 / −0.5 → −3.5 / +0.1.
- **Cache pairs held out** (CLEAN − REPEAT, measured / v1.18 / shipped µs): 1280³ N = 14 −11.4 /
  −11.5 / 0; 2048 × 768² N = 14 −6.1 / −5.9 / 0; 3072 × 1024² N = 8 −20.8 / −11.5 / 0. Over all 21
  pair records the absolute CLEAN (flushed COLD) error goes 7.7 → 3.8 % (median), REPEAT
  2.5 → 2.1 % (`pairs_abs.md`).
- Open-items targeted records (5 calls ≥ 1 ms, 25 at 100 µs – 1 ms, 17 < 100 µs): 5.1 / 8.8 / 11.2
  → 2.5 / 6.5 / 7.2, 8.5 / 13.0 / 14.5 → 3.5 / 5.7 / 11.0, 1.0 / 3.4 / 3.8 → 0.7 / 3.3 / 3.5.
  perfminor-time records: < 100 µs (the noisy 352³ series) 16.0 / 19.6 / 22.3 → 19.0 / 22.6 /
  25.2, 100 µs – 1 ms 5.6 / 21.1 / 26.4 → 3.7 / 18.0 / 23.3, ≥ 1 ms (4096² × 512) 6.9 → 0.7.

**Ablations on the same new records** (offline refits at f_ref 2484, `results/perfminor/impl/vn/`;
hold-out median / p90 / max %):

| variant | < 100 µs | 100 µs – 1 ms | ≥ 1 ms | K3 phase | KF phase |
|---|---|---|---|---|---|
| shipped (v1.17 fit, old records) | 3.9 / 10.8 / 15.8 | 2.7 / 10.3 / 15.7 | 2.4 / 6.6 / 16.0 | 5.2 / 16.3 | 8.9 / 24.4 |
| v1.17 model refit on the new records | 3.9 / 9.8 / 15.2 | 2.7 / 10.4 / 15.5 | 1.6 / 6.4 / 16.4 | 10.6 / 19.2 | 6.7 / 20.5 |
| + planes only | 3.9 / 9.8 / 15.1 | 2.7 / 9.9 / 15.5 | 2.6 / 6.4 / 16.4 | 4.2 / 14.0 | 6.7 / 20.5 |
| + cache term only | 3.7 / 12.2 / 19.0 | 2.9 / 7.4 / 13.4 | 1.6 / 6.4 / 16.4 | 10.6 / 19.2 | 6.6 / 21.1 |
| **v1.18 (both; shipped)** | 3.7 / 12.2 / 19.0 | 2.7 / 6.9 / 13.4 | 2.6 / 6.4 / 16.4 | 4.2 / 14.0 | 6.6 / 21.1 |
| v1.18 without c (Δ + ω) | 3.8 / 10.7 / 17.1 | 2.7 / 8.2 / 13.3 | 2.5 / 6.4 / 16.4 | 4.2 / 14.0 | 7.2 / 22.9 |
| v1.18 without ω (Δ + c) | 4.4 / 10.2 / 15.7 | 2.6 / 9.8 / 15.0 | 2.6 / 6.7 / 16.4 | 4.2 / 14.0 | 7.1 / 23.6 |

- **The planes rule moves the ≥ 1 ms median 1.6 → 2.6 against a v1.17 refit on the same records**
  (still 2.4 → 2.6 against the shipped calibration): `v_mmg` / `v_skg` (KL path, g < N,
  GB-scale K2 misses) go from −1 … +2 % to +2 … +5 %. The v1.17 rule calls the new hit cases
  (`c_skin1280`, `c_l2s1600`) misses, which pulls `pack.dram` to 0.71 ps/B and fits the
  streaming misses by accident, at the price of the PACK phases (K3 10.6 / 19.2 %, PACK class
  relative SSE 6.0 vs 0.94). The planes rule fits the phases and the held-out S cases; the
  streaming K2 at 0.84 ps/B is the open item of §11 ("launch, wave and byte are not separately
  identifiable"), now isolated.
- **The per-call saving c is what costs the small calls.** It makes every small REPEAT call
  with resident inputs ≈ 1 µs faster: hold-out < 100 µs p90 / max +1.5 / +1.9 over Δ + ω alone,
  `v_n18d` −3.5 points, v_s512 N = 14 −17.1 → −19.0 %, and the small uncapped sustained windows
  (BURST time) 1.90 → 2.65 % median (`sus_cmp.md`). It also carries most of the mid-class gain
  (1024³ N = 14 +10.8 → +6.3 %, 100 µs – 1 ms p90 8.2 → 6.9 %) and the pair fit (CLEAN 4.2 →
  3.8 %). Both variants pass the noise rule; c is shipped because the analysis's form passes it
  with the larger gain. **Reviewer question:** ship without c (`perf_fit --cache-terms
  delta,omega`, a refit of the same records, no GPU) if the small-call / N = 15–20 shift
  matters more than the mid-size gain.

**SUSTAINED** (26 windows; nested leave-one-case-out with f_ref and, [review] since
perf-minor-final, the boost clock chosen in each fold; per window in
`results/perfminor/final/power_nested_final.md`, the perf-minor figures in
`results/perfminor/impl/power_nested.md`; "large capped" = touched the cap, ≥ 1 ms). The table
is the **review-fixed** model (power and clock solve with the cache-state time, below);
in brackets the perf-minor figures before the fix:

| fit, windows | t \|err\| median | f \|err\| median | large capped f bias (median) | P \|err\| max |
|---|---|---|---|---|
| v1.15 form (scaled, γ 1), refit on the 26 | 2.70 % (2.34) | 6.6 % (6.6) | +3.4 % (+3.6) | 80.6 % (DIRECT) |
| **v1.18 busy form (shipped)** on the 26 | **1.40 %** (1.62) | **0.7 %** (1.3) | **+1.5 %** (+1.5) | **9.8 %** (9.9) |
| … its 14 calibration (u_*) windows | 1.00 % (0.95; v1.15 form 2.05) | 0.7 % (1.3) | +1.4 % (+1.5) | 4.7 % (6.0) |
| … its 12 "new" windows (see below) | 2.65 % (2.65; v1.15 form 3.00) | 0.7 % (1.0) | +1.5 % (+1.5) | 9.8 % (9.9) |
| v1.18 form fitted on the 14 calibration windows only, boost **given** 2788 MHz (from the 26) | 1.37 % (1.37; the table said 1.40, a median of rounded rows) | 0.4 % | +0.2 % | 2.2 % |
| … the same without a boost (none of the 14 is uncapped: perf_fit warns) | 1.37 % | | | |
| v1.15 form fitted on the 14 only | 0.95 % (1.20) | 2.6 % | +1.5 % | 43.6 % |

Per window (nested, v1.18 → v1.15 form; measured): u_sq12k t +0.8 / −1.7 %, clock 2317 /
2480 (2258) MHz; u_sq8192 N = 14 +0.2 / −3.2 %, 2319 / 2416 (2274); p_k5 −1.5 / +5.6 %, 2423 / 2247
(2306); s_k5b +0.4 / +2.7 %, 2547 / 2321 (2432); u_rmm +6.3 / +6.3 %, power 429 / 309 (450) W; s_dir
−2.4 / −2.4 %, 312 / 67 (346) W; d_s256 +22.1 / +22.1 % (BURST time), 323 / 217 (300) W;
p_s512n8 −5.1 / −5.1 %, 397 / 405 (406) W (perf-minor: 366 W); u_s1024 N = 8 clock 2634 (2625;
perf-minor 2752). As calibrated (not left out), the v1.17 calibration's SUSTAINED predictions
of the 26 windows have median time / clock error 1.41 / 5.4 % and power max 82 %; v1.18 as
shipped by perf-minor 1.21 / 1.1 % and 9.2 %, review-fixed 1.20 / 0.6 % and 6.4 %.
- Fitted on the 14 calibration windows alone the busy form is worse in time than the v1.15
  form (1.37 vs 0.95 %) while clock and power are right; the v1.15 form's f_ref cancels the
  time model's error at the capped clock against a clock bias. On the 26 windows the busy form
  is better on every statistic.
- **[review] The 12 "new" windows are not independent of the form choice**: 11 of them (`d_*`,
  `p_*`) are the power analysis's windows on which the S6 form and γ = 2.5 were chosen; only
  `s_k5b` is unseen (and `s_dir` re-measures `d_dir`). The nested figure covers the
  parameters, f_ref and (now) the boost clock, not the choice of form.
- The second K5-heavy window (s_k5b) is predicted within +0.4 % (clock +4.7 %); p_k5 −1.5 %
  (clock +5.1 %, was +12.6 % in the analysis's nested fit).

**Review (two independent reviews, 2026-10-06; branch `perf-minor-final`, CPU only, no GPU).**
Both: merge after fixes. Verified against the code and the raw data, every must-fix was
confirmed; the fixes:
- **Code: the power and the SUSTAINED clock divided by the COLD time in every cache state**
  (`perf.cpp` `call_seconds` without the cache terms) while `fit_power` uses each window's
  measured REPEAT time and the windows are predicted as REPEAT. Confirmed: the model's COLD /
  REPEAT gap on the windows is 23 / 17 % (u_s512), 30 / 9 % (u_s1024 N = 8 / 16), 14 / 19 %
  (d_s256 / d_s384), 30 % (p_s512n8), 18 % (p_s768), 0 elsewhere. Fixed (DESIGN §6c.8): one
  phase-time function with the request's cache terms serves the time and the solve; t_B stays
  the COLD time at f_ref; a COLD request is bit-identical. Test: (P − idle)·t is the same in
  every state at a fixed clock, a faster state draws more power, and the SUSTAINED REPEAT /
  WARM / CLEAN solve brackets the cap (fails on the perf-minor library). Effect (nested LOO):
  1.62 → 1.40 %, clock 1.3 → 0.7 %; u_s1024 N = 8 clock 2752 → 2634 (measured 2625), p_s512n8
  power 366 → 397 W (406).
- **perf_fit: the nested LOO leaked the boost clock** (`boost_clock` over all windows, reused
  in every fold). Confirmed; `fold_power_kw` recomputes it from each fold's training windows
  (a given `--clock-boost-mhz` stays fixed). Alone (perf-minor library) it leaves the nested
  median at 1.62 %; folds now use 2785–2790 MHz. A busy fit without an uncapped window warns.
  The "14 calibration windows only" row was labelled as a 14-window fit but used the boost of
  the 26 (`--clock-boost-mhz 2788`): relabelled, and its 1.40 % was a median of rounded rows
  (log: 1.37 %).
- **Docs:** 1024³ N = 14 is in-sample for the cache term (above); the acceptance sentence
  compared against a calibration on other records (reworded against `n_v117`); the hold-out is
  the selection set for c and ω (caveat); DESIGN §6c.8 named the draft's shapes (now the built
  `v_s1920`, `v_w3584`, the 3072 × 1024² pair) and promised an identifiable `pack.byte` (not
  borne out); the v7…v11 L_hit fallback is no longer exact for the shipped file (comment,
  DESIGN, and the test, which had become conditional and silently skipped, now checks the exact
  case on a v1.17-style file and that the shipped file's v11 view is exactly the documented
  degraded calibration); §6c.2's "no file is read with features it was not fitted with" is
  false for a v7…v12-sized caller struct: a stated exception (refusing such structs considered
  and not done, DESIGN §6c.8 "Host API 13"). Measured with the shipped file (N = 14, COLD):
  4096³ SUSTAINED 4614 µs at 450 W (NULL) vs 4397 µs at 342 W (v12 view, never capped),
  4096² × 512 / 3584² × 512 g = 4 BURST −7.1 / −7.5 %. The 12 "new" sustained windows are not
  independent of the form choice (above). The dirty-write-back reading of ω is marked a
  hypothesis (DESIGN §6c.8); the calibration header's "p10 p90" for the bands is now "lo hi
  quantiles".
- **Not changed:** c is kept (the reviewers differ; the power reviewer wants the timing
  question tested first), BURST power stays uncapped (documented), D_dram stays a step
  (documented; "Remaining"). Figures for the "Remaining" items with the shipped file: COLD is
  2.8 / 6.7 / 6.3 % slower than CLEAN on 256³ / 512³ / 1024³ at N = 14 and +11.5 µs on ≥ 2048³;
  the BURST power is 490 / 502 / 585 / 545 W at 960³ / 992³ / 1024³ / 1056³ N = 14 (the D_dram
  step at warm.l2_bytes), SUSTAINED clock 2698 / 2674 / 2477 / 2563 MHz; 1536³ BURST 631 W.
- Unchanged by the fix: every BURST prediction and so the hold-out tables above (BURST,
  REPEAT; identical to perf-minor), and every COLD prediction (the default); the time
  parameters are the perf-minor ones bit for bit (the offline refit reproduces them to
  ~1e-10 relative; only the power keys were taken from it). Records and logs:
  `results/perfminor/final/` (`r_fixed.*`, `r_oldC_foldboost.*`, `r_14_*`, `r_scaled.*`,
  `power_nested_final.md`, `figures.txt`, `windows_asfit.txt`).

**Caveats:**
- The small class: p90 / max +1.4 / +3.2 points, N = 15–17 below 100 µs 2.8 / 8.5 / 12.3 →
  4.4 / 11.5 / 13.2 % and N = 18–20 3.0 / 9.8 / 14.1 → 5.5 / 13.1 / 17.3 % (6 calls). A v1.17 refit
  on the same records already moves N = 15–17 to 3.9 / 8.4 / 12.8: about half is the records,
  half the per-call saving c (above).
- The hold-out is not fully independent of the design: the analyses chose the forms on the
  model-fix hold-out; the new hold-out shapes (S rule, pairs, the 352³ series) are the
  independent part. **[review]** It is also the selection set for c and ω: the `vn/*`
  variants (`n_noc`, `n_noomega`, `n_all`) were ranked by these hold-out statistics, so the
  reported hold-out figures of the shipped variant are optimistic by that choice. Of the S-rule
  shapes only `v_w3584` discriminates the rule (a hit by the v1.17 proxy, a miss with planes),
  and all three (S 68 / 95 / 142 MB) lie outside the L_hit bracket (75.5, 90.6) MB: the
  threshold itself is untested on held-out data.
- The shipped `notes` line records the window reuse; 24 of the 26 windows are 0.5–3.5 h older
  than the calibration sweep (1–3 h; same kernels, 68–77 °C at the window ends).
- A v7–v12-sized calibration struct reads the seven fields as 0: for the shipped file a degraded
  view (pass NULL; DESIGN §6c.8 "Host API 13").

**Remaining:**
- The streaming K2 of g < N KL-path calls (`v_mmg`, `v_skg`, `c_rskin2g`: +2 … +5 %, `pack.dram`
  0.84 ps/B with `pack.byte` 0): needs a separable feature (e.g. the K2 launch's own DRAM
  bandwidth at > 1 GB) or a K2-only rate.
- A pre-packed operand at g < N in the S rule (`imd_bench --methods packed` has no `--g`).
- K4 and KF on tiny shapes (`z_s200`, `v_s320`, v_s512 N = 14 −19 %, `v_n18d` N = 18).
- The BURST time of tiny windows (d_s256 +22 %) and of the r_mm-like windows (+6 … +8 %).
- Mid-size capped calls' clock (p_s1536 −5.7 %, u_s1024 N = 16 −4.7 %).
- The reviewer questions: COLD's meaning, `clock_mhz` on the NVML scale, c shipped or not
  (answered in "Review" below; c stays until the timing question is tested).
- [review] **COLD (the default) is unmeasured for res > 0**: COLD = CLEAN + ω·min(F − I, L_w)
  rests on the additive form only (for res = 0 it equals the measured REPEAT). A ≈ 1 min
  pair variant whose flush *writes* a 2 × L2 buffer would validate it.
- [review] **CLEAN / WARM vs REPEAT may be launch timing, not write-back**: on every res = 0
  pair WARM − REPEAT ≥ CLEAN − REPEAT although WARM follows a dirty-leaving call; the CLEAN /
  WARM timed calls start queued behind a GPU op, REPEAT after a host sync. Check: a REPEAT
  variant with a dummy kernel before the timed call. If timing, COLD is pessimistic by up to
  ω·L_w ≈ 11.5 µs (≤ 1–2 % of large calls) and the c / ω split (not checkable from the pairs)
  should be revisited. WARM is still predicted = REPEAT although the held-out pairs measure
  −4.5 / −7.2 / −11.9 %.
- [review] **The clean credit saturates too early**: res = 0 pairs whose clean saving exceeds
  ω·L_w are under-predicted (x_r1 N = 14 and x_s1536 N = 8 by 5.9 µs in-sample, v_x3072 by
  9.3 µs held out); `min(F − I, L_w)` is probably the wrong form for large write-backs.
- [review] **c is confounded with `call.call` on REPEAT records** (call.call 1738 → 1124 ns):
  net, small REPEAT calls ≈ 1 µs faster (v_n18d −3.5 points, small-window BURST median 1.90 →
  2.65 %). The time reviewer would ship Δ + ω (`--cache-terms delta,omega`), the power
  reviewer keep c until the timing question is tested: kept, for the user.
- [review] **D_dram is a step at `warm.l2_bytes`**, the cache fit's L_w: power and cache fit
  are coupled (L_w = 0 charges every call), the predicted BURST power and SUSTAINED clock jump
  between neighbouring sizes (figures in "Review"). Use the phases' own DRAM-byte features
  (K4 dram, k5.dram, pack.dram) or a separate threshold.
- [review] **BURST `power_w` is uncapped** (the power at the boost clock; documented in DESIGN
  §6c.8, not a reading).
- [review] **The power coefficients are not stable**: fitted on 14 windows P_act 210 W, e_alu
  0, e_dram 55; on 26 154 W, 92 pJ/op, 96 pJ/B (byte_f went 289 → 0 since v1.17). P_act and
  e_alu are collinear (exponents differ by 1 over f / f_ref ≈ 0.9–1.12; 92 pJ per ALU op is not
  physical): do not interpret the individual coefficients. 5 NNLS parameters + 2 priors + boost
  + f_ref on ≈ 20 distinct cases; 20 of 26 windows at the cap, so the power level comes from 6
  uncapped windows. Re-check after a fresh sustained run (24 of 26 windows were reused).
- [review] **The L_hit threshold is untested on held-out data** (all three S-rule hold-out
  shapes lie outside (75.5, 90.6) MB): a held-out shape with S ≈ 78–88 MB would test it.
- [review] A v7…v12-sized caller struct reads the shipped file as a never-fitted model (a
  stated exception, DESIGN §6c.2 / §6c.8 "Host API 13"); an adapter like `fold_byte_f` is
  possible if a caller ever needs it (none known: ehrenfest passes NULL).
