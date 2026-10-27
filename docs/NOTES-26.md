# Issue 26 (R5 outcome): the sm_90 default set gains klwarp, 173 -> 189

## Decision

The sm_90 class default (H100, H200) becomes 189 = 173 + klwarp (16) = E1 (1) + F1 (4) + limb CRT (8) + klwarp (16) + klload (32) + k5fin (128).
sm_89 stays 1 and sm_120 stays 0. klshape stays off (4-6% slower).

## R5 evidence

Duel: klwarp set (seq 181 with `crt=f64`, the same computation as the class default 189) against 173, run on both cards. Outputs bit-identical
(`c_hash` equal in every row). Ratios, H100 / H200 (speedup of 181 over 173):

| shape | N | H100 | H200 |
|---|---|---|---|
| 4096^3 | 14 | 1.028 | 1.028 |
| 4096^3 | 8 | 1.045 | 1.044 |
| 8192^3 | 14 | 1.009 | 1.008 |
| 8192^3 | 8 | 1.009 | 1.011 |

Run folders (PACE scratch): `generalWork/immerdink/scratch/runs/20261027-133425` (H100), `generalWork/immerdink/scratch/runs/20261027-102740` (H200).

## History: why R4 did not adopt it

In R4 klwarp alone was 1.026 / 1.013 / 1.022, under the 3% bar, so it was left off. The R4 combination duel c53 (klload + klwarp) was however
1.079 / 1.075 against klload alone at 1.050 / 1.046 at 4096^3 N=14: the two KL bits compound. R5 confirms this on top of the 173 default, so klwarp is adopted.

## What changed

* `src/host/seq_config.hpp`: the `{9, 0, ...}` row carries `kSeqKlWarp`.
* Tests pinning it: `tests/test_seq_table.cpp` (189; klwarp in the sm_90 row only, klshape in no row; the `IMMERDINK_SEQ_TWINS=OFF` default stays 1 | 8, the mask drops klwarp too), `tests/test_seq_select.py` (DEFAULTS and the no-twins masking).
* `bench/imd_bench.cu` help text. `--seq 189` is not accepted from the CLI (bit 8 is class-default-only); `--seq 181 --crt f64` is the same computation.
* No kernel changed: SASS of every shipped kernel is identical (only the host class table moved). `bench/sweep_r5.txt` is untouched (R5 ran at 2410c74, default 173).

## Walking back

One table row in `src/host/seq_config.hpp` (drop `kSeqKlWarp` from `{9, 0, ...}`) and the pinned tests above (189 -> 173, the klwarp row check, DEFAULTS, the no-twins mask).
