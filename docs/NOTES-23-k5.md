# Issue 23 (K5 finish): where the N-independent part of K5 goes, and sequence bit 128 `k5fin`

Goal (large-square performance on H100 / H200): at N = 14 with the sm_90 default set 13 (E1 + F1 + limb CRT), K5 (CRT + finish) is 0.22 ms at 4096 (10 % of
2.13 ms) and 0.86-0.93 ms at 8192; a linear fit in N gives 0.0087 ms per modulus + 0.10 ms fixed at 4096 (0.0345 + 0.373 at 8192), and the same time on an H200
with 43 % more bandwidth: K5 is latency / issue bound, not bandwidth bound. Cut the N-independent part, bit-identical, behind a new default-OFF sequence bit that R4
duels on Hopper. Nothing here changes a C bit; nothing here was timed (no Hopper hardware, and no performance claim is made from the RTX 4090).

## 1. Analysis (static; the dynamic counts are the RTX 4090's SASS under ncu, instruction counts only)

The K5 launch of a large square is `k5_crt_n<N, 4, true, true>` (E = 4 elements per thread, limb CRT, F1). It is one kernel: there is no separate finish launch to
merge. Per CTA: 256 threads; the 14 W planes of 4 consecutive j-bar are one `LDG.32` each (a warp reads 128 contiguous bytes per plane); the CRT, the centring and
the F1 conversion run per element; the 32 results of a lane group are transposed through shared memory and stored coalesced.

**Dynamic instruction count** (`ncu -k k5_crt --metrics smsp__inst_executed.sum`, sm_89 build, N = 14, 2048 x 2048 x 128 uniform data, E = 4, crt=f64 seq=5, per
warp = per 128 outputs; every one of the 1203 instructions of the hot path executes once per warp, the other ~1740 static instructions are rare paths):

| part | instructions per warp | per output | N-dependent? |
|---|---|---|---|
| prologue: grid-stride bounds, two 64-bit-checked divmods (idx / ng, t / m) | ~130 | 1.0 | no |
| W plane loads (14 x LDG + 64-bit address steps) | ~52 | 0.4 | yes (3.7 / modulus) |
| limb CRT, the sums: `I2F` + 3 `DFMA` + byte extraction per modulus | ~294 | 2.3 | yes (21 / modulus / warp = 5.25 per output) |
| limb CRT, the tail: Horner, floor, 3 x `F2I`, carries, repack, subtract P | ~164 | 1.3 | no |
| finish: fA / sA / fB / sB loads, F1 (H - x, P - x, normalise, `FLO`, funnel, `I2F.U64`, exponent) | ~363 | 2.8 | no |
| stores: 20 `SHFL`, `DSETP` alpha / beta, `c_col` gathers, `STG` | ~200 | 1.6 | no |

N = 13, 14, 15 give 38.70, 39.42, 40.08 M warp instructions: the slope is 21 (20 / 22) warp instructions per modulus, so the fixed part is 1203 - 14 x 21 = about
909 of 1203 (75 %) of the instruction stream at N = 14. FP64-pipe instructions are 288 per warp (24 %); the `I2F` / `F2I` conversions are 84 per warp.
At 1203 instructions per warp, one issue slot per cycle per SMSP and 248 warps per SMSP at 4096, the kernel is about 300 k cycles = 0.17 ms at 1.755 GHz of its
0.22 ms: it is issue-bound, which is why more bandwidth (H200) does not help, and why the N-independent instructions are the lever (the time fit's fixed share is 45 %,
smaller than the 75 % of instructions because the slope part has the higher latency per instruction: `I2F.F64` conversions).

**Resources** (cuobjdump `-res-usage`, `k5_crt_n<14, E, true, true>`; the shipped kernels):

| arch | E = 4 | E = 2 | E = 1 | batch fastest |
|---|---|---|---|---|
| sm_89 | 48 regs | 48 | 40 | 54 |
| sm_90 | 64 regs, 9728 B smem | 59 | 40 | 80 |
| sm_120 | 106 regs | 74 | 44 | 128 |

sm_90 E = 4: 64 registers x 256 threads = 4 CTAs per SM (32 warps, 50 % occupancy), register-limited, no spills. Wave shape at 4096 (16384 CTAs, 4 per SM x 132 SMs =
528 resident): 31.03 waves, so the last wave holds 16 CTAs (a partial wave costs at most 1 / 32 of K5, 0.3 % of the call); 8192: 124.02 waves. No change to the CTA shape
is proposed: the per-wave tail is small and the grid-stride loop already evens it out.

**Performance model gap (recorded, not changed here).** The model's K5 occupancy rows (`kernel_occupancy` for `IMD_PERF_KERNEL_K5 + ...`, `k5_occupancy` in
`src/contract.cu`) are the integer-CRT instances `k5_crt_n<N, E>` (F64 = F1 = false). On sm_90 those have (N = 14): E = 4 64 regs, E = 2 48, E = 1 40 -> 4 / 5 / 6
CTAs per SM; the sm_90 default set actually runs the F64 + F1 instances: 64 / 59 / 40 / bfast 80 registers -> 4 / 4 / 6 / 3 CTAs per SM (E = 2: 59 registers x 256
threads = 4 CTAs, the model says 5). The rows are therefore optimistic for E = 2 and the batch-fastest kernel on sm_90 until the model keys its K5 row on the set. k5fin
instances on sm_90: 64 / 47 / 32 / bfast 76 registers.

### What does not apply, and why

* A merge of the finish into the CRT kernel tail: they are already one kernel.
* Programmatic dependent launch (`cudaLaunchAttributeProgrammaticStreamSerialization`) K4 -> K5: **skipped.** K5 reads all of K4's output, so it would only save the
  launch latency of one kernel (a few microseconds of a 2 ms call: about 0.1 %), it needs `cudaGridDependencySynchronize` in K5 and an explicit trigger story for every K4
  backend (CUTLASS mma.sync, the wgmma kernels, the fused small paths) on sm_90+ only, and a launch with the attribute is not accepted on sm_89 builds without a
  guard. The gain does not pay for the legality and graph-capture review.
* Per-element division and `ldexp`: there is none in K5 (the F1 finish adds the scale to the exponent field already); the only divisions are the two index divmods.
* Atomics: none.

## 2. Sequence bit 128 `k5fin` (default OFF in every class row)

`kSeqK5Fin = 128` in `src/host/seq_config.hpp`. A set with the bit needs F1 (bit 4; `seq_valid` rejects 128, 129, 130, 131 like Q1 without E1). It only acts with the limb
CRT (a forced set 0..7 / 128+ runs the integer CRT unless the K5 hook `crt=f64` forces limbs, exactly as for every other forced set): the bit selects separate
kernels `k5f_crt_n<N, E>` and `k5f_crt_bfast<N>` (F64 = F1 = true implied), so with the bit off the shipped K5 kernels are untouched. With the integer CRT, or if the
32-bit index arithmetic does not fit (`k5f_fits`: items + one grid stride < 2^32, every extent < 2^32, b m < 2^32), the shipped kernel runs and the bit does
nothing. Plumbing as for E1 / Q1 / F1: bit constant, `kSeqSupported` (dropped with `IMMERDINK_SEQ_TWINS=OFF`, which drops the 76 k5fin kernels with the Q1 / F1
twins), class rows (none carries it), the hooks (`imd_test_set_seq` / `imd_prof_set_seq` accept 132, 133, 135 and every valid set; the forced-set mask is now
`kSeqForceMask = kSeqSupported`, a mask-based check instead of "0..7"), `ExecPlan::seq` (a `uint8_t`: 128 fits), bench `--seq 132|133|135` and the duel side key `seq=<decimal
bits>`, `seq_eff` / `seq` records, `tests/test_seq_table.cpp`, `tests/test_seq_select.py`, `tests/test_crt_f64_gpu.py`, a launch counter for the k5fin kernels
(`imd_test_k5_fin_counts`).

What the kernels change (all bit-identical; same integers, same doubles, same addresses stored):

1. **32-bit index arithmetic**: one 32-bit `idx / ng`, `t / m` instead of two 64-bit-checked divmods with a 64-bit slow path.
2. **Metadata**: for E = 4 with n % 4 == 0 and aligned `fB` / `sB`, the four `fB` bytes are one 32-bit load and the four `sB` entries one 64-bit load, and a group of four
   with no native element (fA = 0 and all fB = 0, the usual case) finishes straight-line, without per-element branches and address arithmetic.
3. **alpha / beta**: `alpha == 1` and `beta == 0` are tested once per thread (by bit pattern), not in each of the four stores.
4. **E = 4 stores without shuffles when the warp's 32 items are 32 consecutive j-bar groups of one row** (all live, no row change; one `SHFL` + `VOTE` decides): the four
   stores of a lane are j = 4 jg0 + lane + 32 r, and the 20 `SHFL` and the per-store bounds tests (n % 4 == 0) disappear. A warp across a row end, or the grid tail, takes the shipped
   shuffle path.
5. **The final "subtract P once"** of the limb CRT as one PTX carry chain (`sub_limbs`, one `IADD3` per limb) instead of the 64-bit loop.
6. **F1 normalisation behind one test**: when the top limb of the magnitude is nonzero the 4 / 2 / 1-limb stages of `finish_f1_try` find nothing to shift; they run only when it
   is zero (`finish_f1_fast` in `numerics.cuh`; the shipped `finish_f1_try` is untouched).

Measured instruction count (the same ncu probe, sm_89 SASS, N = 14, 2048 x 2048 x 128, E = 4): 39,419,904 -> 31,935,946 warp instructions (-19.0 %), FP64-pipe
instructions unchanged (9,437,184). The sequence of cuts: items 1-5 gave 32,407,552 (-17.8 %); item 6 another -1.5 % on this uniform-data probe (it depends on how many
magnitudes have a zero top limb: a warp takes the long path if any lane does). The sums (the slope of the N fit), 3 `DFMA` + conversion per modulus, are untouched: the
cuts are the fixed part, about 20 % of the instruction stream, which at the 4096 time fit (fixed 0.10 ms of 0.22) predicts a K5 gain of 10-15 % if the kernel stays
issue-bound (0.02-0.03 ms of 2.13 ms = 1.0-1.5 % of the call at 4096 N = 14, a little more at N = 8); the 1.2 % / 1.7 % of the brief are the same order. This is a prediction
from instruction counts, not a measurement.

Resources of the k5fin instances, N = 14: sm_90 E = 4 64 registers (same as the shipped kernel: same occupancy, 4 CTAs / SM), E = 2 47, E = 1 32, batch fastest 76 (shipped 80); shared memory
unchanged (9728 B for E = 4). On sm_89 E = 4 needs 62 registers (shipped 48: 5 CTAs / SM become 4) and on sm_120 79 (shipped 106).

Expectation for sm_89 and sm_120: **no gain on the RTX 4090 and Blackwell, possibly a small loss, and the bit stays off there.** The limb CRT itself is NO-GO on both (FP64 at
1/64: 0.44-0.84x measured), and k5fin is a limb-CRT + F1 kernel, so it could only be adopted together with the limb CRT, which those classes reject on their own data. The
integer-CRT K5 is not covered by the bit. On sm_89 the register count of the E = 4 kernel rises from 48 to 62, which would cost one CTA per SM for kernels that are
FP64-bound anyway. Each class may adopt the bit later only with its own duel (`seq=5,crt=f64` against `seq=133,crt=f64`).

### Not done (candidates for later; none is measured)

* **Byte to double by `PRMT` + `DADD`** (high word 0x43300000, subtract 2^52) instead of `LOP3` + `SHF` + `I2F.F64.U32`: one fewer instruction per modulus per element and the
  conversion moves from the quarter-rate-ish conversion pipe to the FP64 pipe (a warp `DFMA` is 2 cycles on H100): a change of the slope, not of the fixed part; a pipe
  trade that needs the Hopper duel.
* **Centred quotient**: take t = floor(S/P + 1/2 - 2^-30) so the limb remainder is already centred and the x - P, H - x, P - x chains and the selects shrink to one sign
  test; needs a new certificate (spec/crt_f64.py) for the shifted estimate and a boundary fallback. The tail + finish are about 40 % of the instructions, so this is the largest
  remaining item.
* A cheaper repack of the three 40-bit limbs into 32-bit words (about 20 instructions per output), a Lemire 32-bit magic divmod (about 20 per thread), vectorised `c_col` loads when the plan knows `c_col[j] = j`.

## 3. Evidence

* **Shipped SASS is byte-identical with the bit off.** `contract.cu.o` built from `origin/port/hopper-blackwell` (e4425ea) and from this branch, every function of the baseline
  compared on the `cuobjdump -sass` text with encodings (the new kernels are added, no baseline kernel changed or disappeared):

  | arch | baseline kernels | byte-identical | differing | hash of the baseline set, before / after |
  |---|---|---|---|---|
  | sm_89 | 351 | 351 | 0 | e666522076e05e3d / e666522076e05e3d |
  | sm_90 (what the library ships for contract.cu) | 351 | 351 | 0 | 11f54daff9786e4e / 11f54daff9786e4e |
  | sm_90a-real cross-compile | 351 | 351 | 0 | 11f54daff9786e4e / 11f54daff9786e4e |
  | sm_120 | 351 | 351 | 0 | 8b093905b2b32def / 8b093905b2b32def |

  (`tools/sass_seq_counts.sh` TOTAL hashes of the baseline: sm_89 027aaaad5adad44a, sm_90 d577418bacbc973b.) 76 new kernels (`k5f_crt_n<N, 1 | 2 | 4>`, `k5f_crt_bfast<N>` for 19 N) per
  architecture; they are never launched unless the bit is forced (they only cost image size and load time on the cards that never use them; `IMMERDINK_SEQ_TWINS=OFF` drops them).
  The only edits to existing code are `crt_f64_elem`'s new defaulted template parameter `Fin = false` (forced-inline, no symbol) and new functions in `numerics.cuh`.
* **Bit-identity**: `tests/test_crt_f64_gpu.py` runs every committed hash case, N = 2..20 in both families, batched and alpha / beta shapes, and the K5 variants E = 1, 2, 4 and
  batch-fastest (shapes with n % 4 == 0 and rows of 32 / 33 / 128 groups, so the vector metadata, the shuffle-free store, the warp-across-rows path and the tails are all hit)
  under: integer CRT with the baseline finish (reference), limb CRT, F1, both, and the new combinations (limb + F1 + 128 = 132, 135, and the integer CRT with 132, where
  the bit must do nothing); the output hashes must be equal, the committed ones reproduced, and the launch counters must show that a k5fin kernel ran exactly for
  limb CRT + F1 + 128 and nowhere else. `tests/test_crt_f64_unit.py` also runs the limb CRT of the k5fin kernels (variant 10: the carry-chain subtraction) against the exact
  integers on the edge vectors (the mutation checks target the shipped limb CRT; the carry chain is one expression whose only failure mode is a wrong select, which
  the edge vectors z = P - 1, P - 2, H, H + 1, 0, 1 and the random vectors of that test cover).
* **Graph capture / replay**: `graph_capture_k5fin` runs the existing `test_graph` (global-mode capture, replay, bit-exact vs the eager call, both vec modes, packed operands, g > 1, a
  batched tensor, alpha = 0) with `IMD_TEST_SEQ=133;IMD_TEST_K5_CRT=1` and requires k5fin launches (28 on the 4090; the K5 launch is a plain kernel launch on the caller's stream).
* **Full `tools/test.sh`** (4090, sm_89, memcap, GPU lock): host-only pass 34 / 34 and GPU pass 80 / 80, including `contract_{e2e,eps,tensor,every_n,bigk}_k5fin`,
  `output_hashes_k5fin` (the 19 committed hashes reproduced; k5fin ran in all 19 cases: 11 x E = 1, 1 x E = 2, 6 x E = 4, 1 x batch fastest) and `output_hashes_large_k5fin`;
  after the last edits (`IMD_NO_SEQ_TWINS` guards, graph test) the k5fin / seq / crt_f64 / device_status / graph subset was rerun (31 / 31 and 2 / 2). With
  `IMMERDINK_SEQ_TWINS=OFF` `contract.cu` compiles clean under `-Werror=all-warnings` with no k5f kernel.
* **Device-status coverage**: `tests/check_kernel_sites.py` (the `device_kernel_sites` test) requires every launched kernel name to be in the device-status list; `extra_kernel_versions`
  gained two entries (12: a `k5f_crt_n` instance, 13: a `k5f_crt_bfast` instance; without the sequence twins they stand in with the shipped K5), `kExtraKernels` 12 -> 14.
  Host-side only: the SASS table above was rebuilt and re-compared after that change (identical).
* **Hang found and fixed during development**: the E = 4 warp fast path decided `run` with `live && jg == __shfl_sync(full mask, jg, 0) + lane`; `&&` skips the shuffle on dead
  lanes of a tail warp and the warp deadlocks (m = 37, n = 52 / 53, forced E = 4 with seq 132). The shuffle now runs unconditionally before the test; `tests/test_crt_f64_gpu.py`
  covers those shapes.
* The class table is pinned by `tests/test_seq_table.cpp` (no row carries bit 128; 128 alone is not a set) and `tests/test_seq_select.py`.

## 4. R4 rule (pre-registered)

Duel `seq=5,crt=f64` (A, the adopted default) against `seq=133,crt=f64` (B: the same set with k5fin) on H100 and H200, the same cases as R3 (the N = 14 large
squares 2048 / 4096 / 8192, N = 8, g = 7, r_lowk, t_df). **ADOPT (add bit 128 to the sm_90 class row) if** the whole-call ratio is >= 1.015 on at least two N = 14 large
cases on H100 and on H200, no case is more than 3 % slower, and the hashes are bit-identical (all `*-complete` GO, 19 hashes reproduced); floor 0.98 on every case. Otherwise
KEEP off. The bit is per device class: sm_89 and sm_120 adopt only with their own data and only together with the limb CRT.
