# R4 integration: the five branches in one tree

Branch `port/r4-integration` = `port/hopper-blackwell` (e4425ea) + merge commits (not squashed), in this order:

1. `port/22-wgmma-floor` (9fe50c1): the estimated-time profitability rule of the Hopper wgmma K4 (`src/host/k4_wgmma.hpp`), `tests/test_k4_wgmma_select.cpp`, `tools/fit_k4_wgmma_profit.py`, docs/NOTES-22.md.
2. `port/bench-accuracy` (0caa541): `imd_bench` `acc_*` fields, `--accuracy`, `bench/test_accuracy_fields.py`, docs/NOTES-BENCH-ACCURACY.md.
3. `port/23-kl-fixed-part` (4695452): seq bits 16 / 32 / 64, KL section timers, `tools/sass_ident.sh`, `sass_hash.py`, `kl_timers.py`, docs/NOTES-23.md.
4. `port/23-k5-finish` (e29d3fc): seq bit 128 (k5fin), the `k5f_crt_*` kernels, docs/NOTES-23-k5.md.
5. `port/22-seq-crt-usage-diag` (997809f): `bench/test_seq_crt.py` usage-call diagnostic (no-argument call: stdin closed, 300 s limit, self-explaining failure).

Merges 1, 2, 3 and 5 were conflict free. Merge 4 conflicted with 3 on the shared seq-bit plumbing; how, below. Nothing else was changed in any branch's behaviour.

## Conflicts and how they were resolved

| file | conflict | resolution |
|---|---|---|
| `src/host/seq_config.hpp` | `kSeqAll`, `kSeqSupported`, comments, the static_assert text | `kSeqAll = 7 \| kSeqKl` (bits the hooks force besides k5fin; never bit 8), `kSeqK5Fin = 128`, `kSeqSupported = kSeqAll \| kSeqK5Fin` (twins ON) or `kSeqEmit` (twins OFF), `kSeqForceMask = kSeqSupported`. `seq_valid` keeps both rules: Q1 needs E1, 128 needs F1, bit 8 and anything above 255 are refused; the KL bits are independent of everything. |
| `src/plan.hpp`, `src/test_hooks.cpp` | both sides added declarations / hooks at the same place | both kept (`test_kl_launch_counts`, `test_kl_timers`, `test_k5_fin_counts`). |
| `tests/harness.py` | both sides added ctypes bindings and wrappers | both kept. |
| `tests/test_seq_table.cpp`, `tests/test_seq_select.py` | valid-set enumeration (KL: all 0..127 combinations; k5fin: 128 + F1) | one enumeration over 0..255: valid iff no bit 8, Q1 => E1, 128 => F1 (and only bit 1 / 0 in a no-twins build); rejected lists merged; forced-CRT list gains 133 and 245. |
| `bench/imd_bench.cu` | `seq=` / `--seq` parsing and help text | `seq_bits_ok`: decimal 0..255 with bits in {1,2,4,16,32,64,128}, Q1 needs E1, 128 needs F1 (a bench-side pre-check; the library's hook is the authority). Help text names all bits. |
| `ExecPlan::seq` | `uint8_t` | checked: the largest value is 247 + bit 8 = 255, fits. |
| `kExtraKernels` | 14 (branch 4's two k5fin entries) | no change needed; `device_kernel_sites` (which checks the list against the launch sites) passes with both KL and k5fin kernels. |

Three small follow-ups the integration needed (tests only): `tests/test_seq_table.cpp` has two checks that only hold with the twins compiled (a forced set is now masked to the supported bits, branch 4) and failed in the `IMMERDINK_SEQ_TWINS=OFF` build; they are guarded with `#ifndef IMD_NO_SEQ_TWINS`. `bench/test_seq_crt.py` runs 133 and 245 beside 17 / 113 and refuses 128 / 130 / 256. Two new tests of the combinations (below).

## Final seq bit table

`seq` = the instruction-sequence set (`imd_bench --seq`, duel side `seq=`, `IMD_TEST_SEQ`, `imd_test_set_seq` / `imd_prof_set_seq`). A forced set has no limb-CRT bit and runs the integer CRT unless `crt=f64` (`--crt`, side `crt=`, `IMD_TEST_K5_CRT=1`) forces the limbs.

| value | name | kernel | default per class row (sm_89 / sm_90 / sm_120) | validity |
|---|---|---|---|---|
| 1 | E1 | fused residue emitter in every pack kernel | ON / ON / off | any |
| 2 | Q1 | quantize by DMUL + F2I.U64 (W = 2, N <= 16) | off / off / off | needs E1 |
| 4 | F1 | CRT finish through I2F.F64.U64 | off / ON / off | any |
| 8 | limb CRT | K5 sums in binary64 limbs | off / ON / off | class default only, never forced (hook `crt`) |
| 16 | klwarp | KL: row-wide scalar work once per row / tile | off / off / off | any |
| 32 | klload | KL: direct 16-byte global loads, no gather tile | off / off / off | any |
| 64 | klshape | KL: 16 values per thread, 256 threads, rows on chip (nc <= 2) | off / off / off | any |
| 128 | k5fin | K5 with fewer instructions per output (`k5f_crt_n`, `k5f_crt_bfast`) | off / off / off | needs F1; runs only with the limb CRT, else the shipped K5 runs and the bit does nothing |

Class rows are unchanged by the integration: sm_89 = 1, sm_90 = 13 (E1 + F1 + limb CRT) with the wgmma K4 backend on by the profitability rule, sm_120 and unlisted = 0. `tests/test_seq_table.cpp` pins it: no row carries Q1, a KL bit or 128; only sm_90 carries bit 8. All 13 R4 sets below and 5-without-128 are valid together; `seq=245` = 5 + 16 + 32 + 64 + 128 is the all-bits side.

## R4 duel sides and pre-registered ADOPT rules (copied, not changed)

A side is always `seq=5,crt=f64` (the adopted sm_90 set spelled out: the plan default 13 is the same bits with the limb CRT in the set). A `bit_identical` record (`bit_identical: true`, `elements_differing: 0`) is required in every pair.

| bit | B sides | ADOPT rule (put the bit in the `{9, 0, ...}` row of `kSeqClasses`) |
|---|---|---|
| klwarp 16 | `seq=21,crt=f64` | NOTES-23 "What R4 decides": >= +3% whole call on >= 2 N = 14 large cases (4096 / 8192 squares and the other large cases) on BOTH H100 and H200, no case more than 3% slower (floor 0.98: a case under it is a stop), everything bit-identical. Otherwise stays off. |
| klload 32 | `seq=37,crt=f64` | same |
| klshape 64 | `seq=69,crt=f64` | same |
| KL pairs / triple | `seq=53` (16+32), `101` (32+64), `85` (16+64), `117` (all three), each `crt=f64` | the combinations of the winners; the effects need not add (klload + klshape: E = 16, eight 16-byte loads per thread, 107-128 registers). |
| k5fin 128 | `seq=133,crt=f64` (5 + 128) | NOTES-23-k5 section 4: whole-call ratio >= 1.015 on >= 2 N = 14 large cases on H100 and on H200, no case more than 3% slower, hashes bit-identical (all `*-complete` GO, 19 hashes reproduced), floor 0.98 on every case; cases of R3: N = 14 squares 2048 / 4096 / 8192, N = 8, g = 7, r_lowk, t_df. Per device class (sm_89 / sm_120 only with their own data and only together with the limb CRT). |
| 128 with KL | `seq=149` (5+16+128), `165` (5+32+128), `197` (5+64+128), `245` (all) | not pre-registered in either notes file; these are the combined-tree sides (they exist in the tests). Suggested rule: the sum of the two separately adopted bits, judged by the same thresholds against `seq=5`. |
| timers first | `tools/kl_timers.py`, section timers build | NOTES-23: if `sigma + red1 + red2` < 10% of the CTA cycles at 4096 / 8192, klwarp is expected to tie; if `ssum` >= 15%, the next item is the FP64-pipe S (not part of this issue). |
| wgmma K4 (issue 22) | `k4b=default` vs `k4b=mma`; `k4b=wgmma` forced | NOTES-22: no ADOPT bit; the rule (`kK4WgmmaProfit[IMD_TABLE_SM90]`) serves a launch iff m, nc >= 512, kc >= 256, T >= 0.5 S and the estimated saving >= max(0.75 us, 10% of the mma.sync K4 time). R4: widen the floor sweep (N = 8 and 20, g = 2, 3, 7, small m or n-part, 1536 - 2048 squares at g = 1 and 2, 4096 / 8192 at g = 1, a K > 131008 shape, repeated runs of the five boundary shapes); any shape the rule serves with a measured ratio < 1.02 is a refit trigger (`tools/fit_k4_wgmma_profit.py ... --check`). `c_hash` / `bit_identical` of every k4b pair is a pass/fail gate. |

## Hooks and record fields the R4 manifest can rely on

Command line (`imd_bench`, profiling build from `bench/build.sh`):

* `--seq default|<bits>` and the duel side key `seq=default|<bits>`: decimal 0..255 as above. `--crt default|int|f64`, side key `crt=`. `--k4b default|mma|wgmma` / side key `k4b=`, `k4=`, `k5=`, `ka=` as before.
* `--accuracy 0|1` (the full-matrix DGEMM comparison, default on when its buffers fit), `--samples N` (the sampled double-double reference, default 256; 4096 for the large squares).
* `--keep-samples 1` adds `samples_s` (raw per-repetition times) to timed records and `pair_t_a` / `pair_t_b` to duel-b.

Records:

* method `imd`: `seq_req`, `seq` (the set the plan ran), `crt_req`, `crt`, `phase_seq`, `phase_crt`, `seq_after_reset`, `crt_after_reset`, `c_hash` (equal across every seq / crt of a case), `k4b_req`, `k4b_run`, `k4_wgmma_served`, `k4_wgmma_declined`, `tflops_eq`, `median_s`, and the accuracy fields `acc_ref`, `acc_n_sampled`, `acc_max_relerr_cw`, `acc_bits_cw`, `acc_max_relerr_elem`, `acc_bits_elem`, `acc_max_rel_to_norm`, `acc_dgemm`, `acc_dgemm_max_relerr_cw`, `acc_dgemm_max_relerr_elem`, `acc_vs_dgemm_max_cw`, `acc_vs_dgemm_mean_cw`, `acc_vs_dgemm_n`, `acc_vs_dgemm_nonfinite` (also `max_cwerr`, `max_relerr`, `max_normerr` on all methods with an accuracy).
* method `duel` (records `duel-a`, `duel-b`): `seq_eff`, `crt_eff` per side (the set / CRT read back after the side's hooks), `k4b_req`, `k4b_run`, `k4_wgmma_served` / `_declined`, `ratio_median` / `ratio_p10` / `ratio_p90`, `bit_identical`, `elements_differing`, `tflops_eq`, `median_s`. Duel records carry NO `c_hash`: the pair is compared element by element on the device (`bit_identical`), which is the stronger check. For `c_hash` equality across sets use method `imd` records (as `bench/test_seq_crt.py` does).

Test and tool hooks (libimmerdink built with test hooks, i.e. the standard test build):

* environment for the tests (`tests/harness.py`, read back or the run fails): `IMD_TEST_SEQ=<bits>`, `IMD_TEST_K5_CRT=0|1`, `IMD_TEST_K4=mma|wgmma|auto`, `IMD_TEST_REQUIRE_K5FIN=1` (`test_graph`).
* C hooks: `imd_test_set_seq` / `imd_test_get_seq` / `imd_test_plan_seq`, `imd_test_set_k5_crt` / `_get_` / `imd_test_plan_crt`, `imd_test_set_k4_backend` / `_plan_`, `imd_test_kl_launch_counts(reset, int64[12])` ([E = 4 / 8 / 16][Kb]), `imd_test_k5_fin_counts(reset, int64[4])` (k5fin launches per variant), `imd_test_kl_timers(reset, out, n)` (-1 in a build without the knob); the `imd_prof_*` twins for the bench.
* KL section timers: compile-time, default OFF: `-DIMMERDINK_KL_TIMERS=1` (or `2`: also stamps each modulus's emit and store; perturbs the loop) with tests ON. Not for timing runs; the shipped SASS is not the timers' SASS. Tool: `python3 tools/kl_timers.py --lib <timers build>/libimmerdink.so --shapes 4096 8192 --seq 5 21 37 69 133 245 --N 14 [--b-kn]` prints cycles per warp-tile and the share per section (`setup load red1 ssum red2 sigma reload quant emit store tail`), the KL launches seen, the R4 test lines (`sigma + red1 + red2` >= 10%, `ssum` >= 15%), and checks one C hash over all sets. It takes the GPU lock. At small shapes (1024) the KL kernel is not launched (the tool says 0 CTAs).
* SASS identity: `tools/sass_ident.sh <tag>` (sm_89, sm_90a, sm_120 cross-compile of `src/contract.cu`; hashes in `$SASS_OUT/<tag>_<arch>.hash`), `tools/check_k4_wgmma_sass.sh build/libimmerdink.so build.log` (the Hopper machine-code gate: IGMMA > DEPBAR, no local memory, no C7510).

## Tests and counts (RTX 4090, sm_89, this tree)

`tools/test.sh` (full; build -j4; GPU lock): **34 / 34 CPU-only tests and 130 / 130 GPU tests passed = 164 / 164** (`ctest -N` = **164** with `IMMERDINK_BENCH_BIN` set, **162** without: `bench_seq_crt` and `bench_accuracy_fields` are registered only then). This integration adds 22 of them (`r4_set_<seq>_<crt>` x 21, `forced_sets`). For the manifest's `--expect-tests`: 164 for an sm_89 configuration with the bench binary; a Hopper build (`-DIMMERDINK_K4_WGMMA=ON`, arch 90) registers the same names, check `ctest -N` once on a Hopper before pinning the number.

New tests of this integration:

* `r4_set_<seq>_<crt>` (21): the oracle suites e2e and eps and the 19 committed hashes (small + mid 13, large 6) under one forced set and K5 CRT. Sets 5 21 37 69 53 101 85 117 with `f64` and `int`; 133 149 165 197 245 with `f64` only (128 runs only with the limb CRT).
* `forced_sets`: in-process, all 13 sets x crt f64 and the 8 without 128 x crt int on the 16 KL contractions (336 runs): one hash equal to the baseline (set 0, integer CRT), the KL launch counter equals the kernel each operand's bits ask for, the k5fin counter is nonzero for 128 + limb CRT on every set with 128 (16 launches each) and zero everywhere else.

Tests to run on Hopper nodes (everything; these are the ones that matter for R4, the full list is `ctest -N`):

* seq plumbing: `host_seq_table`, `seq_select`, `device_kernel_sites`, `host_devices`.
* KL bits: `kl_bits`, `contract_{e2e,tensor,eps}_kl{17,33,65,113,112}`, `contract_bigk_kl{17,113}`, `output_hashes_kl{17,33,65,113,112}`, `output_hashes_large_kl{17,113}`.
* k5fin: `contract_{e2e,eps,tensor,every_n,bigk}_k5fin`, `graph_capture_k5fin`, `output_hashes_k5fin`, `output_hashes_large_k5fin`, `crt_f64_gpu`, `crt_f64_unit`, `spec_crt_f64`.
* combinations: `r4_set_*` (21), `forced_sets`.
* wgmma K4: `k4_wgmma_select`, `k4_wgmma_exact` (must report served launches > 0), `contract_{e2e,every_n,bigk}_k4wgmma`, `output_hashes_k4wgmma`, `output_hashes_large_k4wgmma`, `output_hashes_k4mma`; plus `tools/check_k4_wgmma_sass.sh` before any test.
* bench: `bench_seq_crt`, `bench_accuracy_fields` (with `IMMERDINK_BENCH_BIN` = the profiling-build `imd_bench`).
* the whole rest of the suite (`host_*`, `contract_*`, `output_hashes_*`, ...).

## Verification of the integrated tree (RTX 4090 and cross-compiles; no performance claim)

* Builds: standard (arch 89, `-Werror`), `-DIMMERDINK_KL_TIMERS=1` (clean; the tests pass there, `tools/kl_timers.py` runs at 4096 with `seq 5` and `245`, one C hash), `IMMERDINK_SEQ_TWINS=OFF` (clean; no `k5f_crt` symbols; `host_seq_table`, `seq_select`, `kl_bits`, `device_kernel_sites` pass), arch `89;90` with `IMMERDINK_K4_WGMMA=ON` (`tools/check_k4_wgmma_sass.sh`: PASS, 2 wgmma kernels, IGMMA 8 / DEPBAR 2, no local memory, no C7510).
* SASS identity against `origin/port/hopper-blackwell` with all new bits off (`tools/sass_ident.sh`): all 351 baseline kernel hashes are present in the integrated build on sm_89, sm_90a and sm_120 (`comm -23` empty on each); the integrated build has 467 kernels (351 + 40 KL instances + 76 k5fin kernels), baseline TOTAL hashes sm_89 027aaaad5adad44a, sm_90a d577418bacbc973b, sm_120 e6f03535c743394e as in NOTES-23.
* Bench: duel `seq=5,crt=f64` against `seq=21 37 69 53 101 85 117 133 149 165 197 245` (each `crt=f64`) at 2048^3, N = 14: every pair `bit_identical: true`, `seq_eff` / `crt_eff` read back as asked. Method `imd` with `--seq 245 --crt f64 --accuracy 1 --samples 512`: `seq` 245, `crt` f64, the `acc_*` fields present, `c_hash` equal to `--seq 5`.
