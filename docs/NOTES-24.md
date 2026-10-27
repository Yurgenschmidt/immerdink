# Issue 24 (R4 outcome): the sm_90 default set becomes E1 + F1 + limb CRT + klload + k5fin

R4 (2026-10-26, commit a430a5e; jobs 14468448 H100, 14468449 H200, 14468450 RTX PRO 6000 Blackwell; run folder
`generalWork/immerdink/scratch/runs/20261026-225145`; narrative in `immerdink-eval/findings/r4-20261026.md`).

## Decision (pre-registered rules applied)

| bit | H100 / H200 whole-call, N = 14 large squares | verdict |
|---|---|---|
| klload (32) | 4096 1.049 / 1.047, 8192 1.046 / 1.038, g=7 1.043 / 1.041, 2048 1.079 / 1.074, N=8 1.059 | ADOPT both cards |
| k5fin (128) | 4096 1.017, 8192 1.012, g=7 1.018, r_lowk 1.054; mirrored duel agrees, role-bias closeness 0.9994 | ADOPT both cards |
| klwarp (16) | 1.026 / 1.013 / 1.022 | KEEP (under the 3% bar) |
| klshape (64) | 0.967 / 0.975 / 0.961 (floor NO-GO 0.937) | KEEP, off (slower) |

Rule 2b (a combination is adopted only if every member is adopted on both cards, it is bit-identical, no row is below 0.98 and its
geomean is >= 0.99 of its best member): only **klload + k5fin** qualifies. Forced as seq 165 (E1 + F1 + klload + k5fin) with `crt=f64`
it is +4.7..7.5% over the previous sm_90 default (set 13): H100 4096 N=14 1.070, 8192 1.047, g=7 1.060, r_lowk 1.054, 8192 N=8 1.075, t_df 1.081
(61.6 -> 65.9, 77.4 -> 81.1, 125.2 -> 134.5 TFLOPS-eq); H200 63.4 -> 67.6, 78.9 -> 82.9, 128.1 -> 136.0. Everything bit-identical.

The class row is the bit set plus the limb CRT (bit 8, a class-default-only bit): 1 + 4 + 8 + 32 + 128 = **173**. Forcing set 165 with `crt=f64` is
the same computation. k5fin runs only with F1 and the limb CRT; with `crt=int` its bit is inert.

## What stays as it was

* sm_89 (RTX 4090) = 1, sm_120 (Blackwell) = 0: klload and klwarp tie on Blackwell (1.000..1.025), klshape loses 2..4%; k5fin needs the limb CRT, which is a NO-GO on a 1/64-rate FP64 pipe. The 4090 was not part of R4; its row is unchanged.
* Class defaults for K4 (wgmma on the profitability rule), the hooks, force masks and legality are unchanged. `IMMERDINK_SEQ_TWINS=OFF` masks the new bits (default = 1 | 8 there).
* All SASS of the shipped kernels is unchanged (only the host class table moved).

## Walking back

One table row in `src/host/seq_config.hpp` (`{9, 0, ...}`), and the three tests that pin it (`tests/test_seq_table.cpp`, `tests/test_seq_select.py`).

## Evidence not collected

The default set 173 itself (all five bits plus the limb CRT) was measured through its forced twin (seq 165 + crt f64) against seq 5 + crt f64 (= the old default 13); it has not been run as the library default on Hopper. R5 / the next Hopper run should include a `size.sweep` read-back (`seq_eff` 173 on sm_90) and the usual hash gates.
