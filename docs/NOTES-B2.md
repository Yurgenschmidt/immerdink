# B2 notes: the performance model (DESIGN v1.8 §6c)

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
