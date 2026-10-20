# Issue 16: is a Hopper wgmma K4 worth building? The rule, fixed before the measurement

Status: rule fixed 2026-10-15 (reviewed by the independent reviewer, who sets such thresholds; the owner may overrule). Code:
`bench/campaign/decisions.py` (kind `rate_projection`), `bench/campaign/required_rate.py`, `bench/ceiling/int8_ceiling.cu`.

## The question, for a reader new to the project

immerdink turns one FP64 matrix product into N = 14 exact INT8 matrix products (one per modulus) plus conversion kernels before and after
("pack" and "CRT", together *pack/CRT*). **K4** is the kernel that does the INT8 products. It uses the instruction `mma.sync`. On an H100
(Hopper GPU) it reaches about 950 to 1000 TOPS (trillions of INT8 operations per second), roughly half of what the hardware can do. Hopper has
a newer, asynchronous instruction, **wgmma**, which stock NVIDIA kernels (CUTLASS) use to get closer to the peak. Rewriting K4 with wgmma is
expensive and risky, and cluster GPU time is scarce. So the first cluster run (R1) measures, and a rule written *now* turns the numbers into
GO, NO-GO or INCONCLUSIVE. The reference to match is GEMMul8 fast, the best other emulator (target: ratio at least 1.0 on average and no case below 0.9).

## What the data say before R1 (issue-11 run, H100)

`required_rate.py` computes how fast K4 must run for parity if pack/CRT keeps today's time P, or takes 0.67 P or 0.5 P:
at 4096³ today's K4 is 952 TOPS and parity needs 1913 / 1364 / 1189 TOPS (1.43× today at 0.67 P); at 8192³ today 995 and parity needs
1497 / 1305 / 1225 (1.31× at 0.67 P). Some cases cannot be reached by K4 speed alone (r_mm: pack/CRT alone is slower than GEMMul8's whole call).
Whether pack/CRT can reach 0.67 P is open and belongs to issue 21: GEMMul8's own trace (R1) is the existence evidence. (The "lower bound" columns in the
old phase records come from the performance model's RTX 4090 constants, so they say nothing about H100 bandwidth; do not read them as an H100 limit.)
With K4 1.3× faster and pack/CRT at 0.67 P the projected 4096³ time is still about 1.07× GEMMul8's: the gain test alone is not enough there.

## The rule

Per GPU, for the shapes sq4096 and sq8192 with N = 14. Notation: `ops` = INT8 operations executed = 2·N·b·m·n̂·k̂; `K4`, `P` = the K4 phase time and the sum of
the other phases, from the phase marks of the immerdink record; `R_K4 = ops/K4`; `T_imd`, `T*` = the whole-call times of immerdink and GEMMul8 fast from the
*interleaved* (alternating) comparison `imd_bench --methods xduel`; `s = T_imd/(P+K4)` (must lie in [0.95, 1.08], else the phase marks and the call
disagree and the case is INCONCLUSIVE).

- `R_stock` = the best rate among the stock CUTLASS Hopper kernels (several tile and cluster shapes, `bench/ceiling`) that are bit-exact against a CPU reference, run on K4's launch
  shape: per modulus m × n̂ × k̂, batch N·b in one launch, int8 output preferred (K4 stores 8-bit residues; int32 output pays four times the store traffic). The int8 exactness check is weak (with random data almost every
  output saturates at ±127), so an int8 rate counts only for a variant that also ran bit-exact with int32 output on the same shape; otherwise the int32 rate is used.
- `R_ref = max(best exact stock rate of any output type, R_cuBLASLt)` for the optimistic case O (cuBLASLt is a yardstick, never a fallback). `R_mma` = the synthetic `mma.sync` peak loop.
- Noise haircut `u = max(0.05, spreads)`; if any spread involved (GEMMul8, immerdink phase record, xduel-imd, the chosen stock record, cuBLASLt) is above 5 %, `u = max(0.10, spreads)`.
  A missing spread, or `u >= 1`, makes the case INCONCLUSIVE (a haircut of 100 % would otherwise turn noise into a pass).
- Projection `Q(R, f) = s·(f·P + ops/R) / T*`; parity within 5 % means `Q ≤ 1.05`.

Tests: **A** `R_stock/R_K4 ≥ 1.30`. **B** `Q(R_stock·(1−u), 0.67) ≤ 1.05`. **C** `0.9·R_mma` is below the rate at which `Q(R, 0.67) = 1.05` (so tuning `mma.sync` alone cannot reach it).
**O** (optimistic) `Q(R_ref·(1+u), 0.5) ≤ 1.05`.

| Verdict | Condition |
|---|---|
| GO | A, B and C hold at both shapes and every input is valid |
| NO-GO | O fails at either shape: even the best demonstrated rate, with pack/CRT halved, cannot reach parity |
| INCONCLUSIVE | everything else (O holds but GO does not: a tuning shortfall or a thin margin, repeat the ceiling block in R2; an invalid or missing input; C fails) |

Invalid inputs: a stock configuration that fails to build or run, or is not bit-exact, is dropped; if all are dropped GO is impossible for that shape and NO-GO can
still follow from cuBLASLt alone. A missing shape gives INCONCLUSIVE unless the other shape already gives NO-GO.

**H200** confirms and never vetoes by itself: its decision (role `confirm`) is PASS when `R_stock/R_K4 ≥ 1.2` at both shapes, FAIL below, INCONCLUSIVE without data. H100 GO plus an H200 FAIL
is read as INCONCLUSIVE (an anomaly to understand). The two verdicts come from two job folders, so this reading is **manual** (the analyst's, stated in the report); no code combines them.

**Frozen rule.** Every parameter of the rule (`n`, `g8_n`, `s_min`, `s_max`, `u_floor`, `u_noisy`, `noisy_spread_pct`, the thresholds and `cases`, which must be sq4096 and sq8192) is in the decisions file, whose
sha256 the manifest pins; the evaluator has no defaults for them (lint rejects a file without them). The evaluator code itself is the pinned commit's `bench/campaign/decisions.py`;
the analysis report quotes the commit.
With the issue-11 H100 numbers the thresholds mean: B needs about 1335 TOPS at 4096³ (1.40×) and 1288 at 8192³ (1.29×); NO-GO means the best rate is below about 1061 TOPS at 4096³ (1.11×) or 1097 at 8192³.

## What R1 records for this rule

Per shape on H100 and H200: the immerdink N = 14 record with phases (5 or more phase repetitions); the xduel records with `--keep-samples` for GEMMul8 fast and immerdink at their DGEMM-level N;
the `imd_ceiling` records for every stock configuration (launch shape, output type, `exact`, repetitions, spread, clocks after the block), cuBLASLt on the same shape, the `mma.sync` peak; clocks and
power per experiment (the runner does that); the GEMMul8 and cuBLAS-emulation nsys traces (`profile_report.py`: INT8 GEMM time, conversion time, other, INT8 launches per call, kernel names; these give the
pack/CRT time GEMMul8 achieves, an existence proof for the 0.67 factor). The synthetic wgmma peak loop is a later extra (reporting only).

## Not covered

The decision says whether to build the wgmma K4 for the INT8 product; it does not say how the K4 epilogue (the mod-p reduction) should look (issue 17). Stock kernels write plain int8/int32 output.
Pack/CRT speed-ups are issue 21.

## Outcome (R1, R1b, R1c on 2026-10-19)

The rule has now seen real Hopper stock-kernel rates. Data: `immerdink-eval/findings/r1-20261019.md`, run folders 20261019-132130 (R1), -181535 (R1b), -223127 (R1c).

* **Pre-registered decision: INCONCLUSIVE on H100 and H200, not GO.** In R1b the whole-call anchor at sq8192 was outside [0.95, 1.08] (1.149 H100, 1.111 H200: inside the GEMMul8 duel the immerdink call is 14% slower than alone, the interleaved
  GEMMul8 kernels change the clocks), so the formal verdict is INCONCLUSIVE for that reason. With the anchor ignored the rule still does not say GO: at sq4096 on H100 the gain A is 1.13 (needs 1.3) and the projected time ratio B is 1.32 (needs 1.05);
  the optimistic case O holds (0.91), so NO-GO is not triggered. H200: A = 1.23, B = 1.36; at sq8192 the gain is 1.10 (H100) and 1.17 (H200).
* **R1c (single GEMMs) did not change that.** The best stock CUTLASS Hopper rate is 1028 TOPS at 4096³ on H100 (NO-GO threshold 1150), 1243 at 8192³ (GO needs 1290), 1193 and 1280 on H200, against our K4's 959 to 992. Batching is not the cause (the stock kernels
  get faster with the batch at 4096³). cuBLASLt reaches 1313 to 1544 on the same shapes, 14 to 28% above the best stock configuration.
* **What this means.** A wgmma K4 built from stock CUTLASS configurations gains 1.06 to 1.33 times in the K4 phase (62% of an H100 call at 4096³), i.e. about 3 to 17% of the call, for a kernel that must also carry the modular epilogue (issue 17 showed it can be made exact).
  Reaching the 1.4 to 1.5 times that cuBLAS achieves needs kernel engineering beyond the stock configurations, whose scope has not been estimated. Issue 22 as designed (stock CUTLASS wgmma mainloop + epilogue) is not supported by the numbers; the
  larger and cheaper lever measured so far is the pack/CRT instruction count (issue 21: 38% of an H100 call at 4096³, instruction-issue bound).
* The synthetic `mma.sync` peak is 1275 TOPS on H100 and H200, so our K4 (966 to 992) is at 76% of what `mma.sync` can do; on the RTX PRO 6000 it is 843 to 894 of 887 and nothing in the vendor stack is faster.
