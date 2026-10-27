# Issue 25 ("fix 2"): klpersist, a persistent klload KL with a cp.async prefetch of the next tile (seq bit 256, default OFF)

Status: implemented on branch `port/25-kl-persist` (from `port/r5-integration` 2410c74), bit-identical on the RTX 4090, **default OFF in every class row**
(sm_89 = 1, sm_120 = 0, sm_90 = 173 unchanged). Intended for sm_90 (H100 / H200), where KL is latency bound; it is adopted there only if the R6 duels say so
(rule below). Nothing in this note is a time: there was no timing run, the 4090's KL is DRAM-bound and is not the target. Written for a reader new to the project.

## The idea

The sm_90 default packs the operands that are read along unit-stride k with the klload KL kernel (`kl_scale_pack<E, W, S, Kb = 2>`, docs/NOTES-23.md): one CTA per
tile (one 4096-element row at k_hat <= 4096, two chunks at 8192), every thread loads its E values of chunk 0 straight from global memory into registers, then scans
them (pass 1), sums squares (pass 2), computes sigma and emits the residues (pass 3). The launch has one CTA per tile, so each CTA starts with a cold global load
and nothing overlaps that latency except the other resident CTA.

klpersist makes the same kernel persistent and gives every CTA something to do while its loads are in flight:

* **grid = min(tiles, 2 x SMs)** (host-known `ExecPlan::sms`, as K5's launch does): two CTAs per SM (E = 8: 512 threads; E = 16 with klshape: 256 threads; both
  instances are bounded for two per SM), tiles walked with stride gridDim.x exactly as before;
* a **32 KB shared prefetch buffer** `s_pf` (kTileElems doubles), pair-major: pair j of thread i at `2 (j T + i)`, so a warp's 16-byte reads of one pair are
  contiguous (no bank conflicts) and every thread reads only what it copied itself;
* after pass 1 of tile t (before the gmax reduction, pass 2 and pass 3), the th row-info threads compute the **next tile's** row info (t + gridDim.x) into the second
  slot of double-buffered `s_valid / s_base / s_rho / s_dst`; after a barrier (the one of the gmax combine for rows of > 32 threads, one extra barrier for shorter
  rows) every thread issues **cp.async.cg 16-byte copies of its chunk 0 of the next tile** into `s_pf`, with fetch()'s exact semantics: an invalid row gives zeros,
  a k tail (`hl + E > k`) or an unaligned row address (odd pitch / row offset) falls back to scalar loads with the same zero fill, stored into `s_pf` by the thread;
  then `cp.async.commit_group`;
* at the start of the next iteration the thread does `cp.async.wait_group 0` and takes r0 from `s_pf` instead of `fetch(0)`. The first tile of each CTA still uses
  `fetch(0)` (no prefetch exists yet), and the last tile issues none;
* the existing end-of-tile barrier, guarded by `t + gridDim.x < tiles`, protects the reuse of the row-info slot and of `s_wg / s_ws / s_sigma`, as in klwarp.

Chunk 1 (k_hat 8192, nc = 2) is still fetched directly in pass 1 and re-read in passes 2 / 3, as klload does. The emitter, the reductions and sigma are untouched:
every element goes through the same arithmetic, so the output is the same bits by construction (the tests below are the gate).

**klwarp's row-info part is taken too.** The first version computed `t / n_d`, `t % n_d` and the store offsets per tile (as klload without klwarp does) and spilled
at the 64-register bound: ptxas on sm_90a gave `kl_scale_pack<8,2,1,6>` 144 bytes of stack and 148 / 192 bytes of spill stores / loads. With klwarp's parts (a) + (b)
(the launch scalars D / n_d / tiles / nc from the host `LargeArgs`, rho and the output offset per row in shared memory; the reductions and largest_u stay as in the
set's own Kb) it is 8 bytes of stack and 4 bytes of spill, the same as the existing klload instance (16 bytes of stack). Same values, same bits.

## Plumbing

| where | what |
|---|---|
| `src/host/seq_config.hpp` | `kSeqKlPersist = 256`; `kSeqAll` (hence `kSeqSupported` / `kSeqForceMask`) includes it; `seq_valid` requires klload with it (256 alone, 256 + 16, 256 + 64 ... are rejected); `seq_kl_kb` maps it to Kb bit 2 only together with klload; **`kSeqClasses` unchanged** (static_assert: no row carries 256). `kSeqKl` stays 112 (the twins guard) |
| `src/host/plan_data.hpp`, `src/api.cpp` | `ExecPlan::seq` is `uint16_t` (was `uint8_t`). sizeof(ExecPlan) = 480 before and after (host g++ check): `trunc` at 64, `seq` moves from 65 to 66 (the padding before `k_align` at 68), every other offset unchanged |
| `src/contract.cu` | `kKlPersist = 4` (Kb bit 2), `KlCpAsync` (cp.async.cg 16 / commit / wait_group 0; static members so a no-twins build has no unused-function warning), the kernel changes above under `if constexpr (kPersist)`, instances `Kb = 6, 7` (E = 8, 16; E = 4 clears the bit: its rows have > 2 chunks), launch: bit cleared where klload is cleared or `p.sms <= 0`, grid as above; the KL launch counter is `[3][8]` |
| hooks | `imd_test_set_seq` / `imd_prof_set_seq` accept the new sets through `seq_valid`; `imd_test_kl_launch_counts` returns **int64[24]** (`[E 4/8/16][Kb 0..7]`, was 12); new test hook `imd_test_set_kl_grid(blocks)` (0 = the rule, > 0 forces min(tiles, blocks): small cases walk several tiles per CTA) |
| bench | `imd_bench` `seq_bits_ok` accepts bit 256 with 32 (values up to 503; `--seq 256 / 257 / 272 / 512` refused before any GPU work); messages name all bits. `bench/test_seq_crt.py` adds set 421 and the refusals |
| `bench/sweep_r6.txt` (new) | three duel lines, A = class default (173), B = `seq=421,crt=f64` (= 173 + 256): 4096^3 N = 14, 8192^3 N = 14, 4096^3 N = 8. Not wired into `run_metrics.sh`; run with `bench/run_sweep.sh bench/sweep_r6.txt <out>.jsonl` |
| `tools/kl_timers.py` | unchanged (takes any integer set: `--seq 421` works; the timers build compiles with the new kernel at level 1 and 2) |

## SASS identity (owner's requirement: the shipped kernels do not change)

`tools/sass_ident.sh` (nvcc 13.4.92, `-O3 -DNDEBUG`, cross-compiled), baseline = `port/r5-integration` 2410c74, per-kernel (name, normalised SASS hash) pairs:

| build | sm_89 | sm_90a | sm_120 |
|---|---|---|---|
| production: baseline kernels / this branch | 467 / 487 | 467 / 487 | 467 / 487 |
| test hooks (`-DIMD_BUILD_TEST_HOOKS=1`) | 676 / 696 | 676 / 696 | 676 / 696 |
| baseline (name, hash) pairs missing or changed | **0** | **0** | **0** |

The 20 added kernels are exactly `kl_scale_pack<E, W, S, 6 | 7>` (E 8 / 16; W 2 with S 0 / 1 / 3, W 3 with S 0 / 1). Every shipped kernel, all KL instances included,
is byte-identical after normalisation on all three architectures, with and without the test hooks. Compile-checked with `-Werror=all-warnings` also with
`-DIMD_NO_SEQ_TWINS=1` and `-DIMD_KL_TIMERS=2`.

## Resources (ptxas / cuobjdump, W = 2, S = 1; registers / stack bytes / static shared bytes)

| instance | sm_89 | sm_90a | sm_120 |
|---|---|---|---|
| E8 Kb2 klload (shipped on sm_90) | 64 / 16 / 960 | 64 / 16 / 1984 | 64 / 0 / 1984 |
| E8 **Kb6** klload + klpersist | 64 / 8 / 36352 | 64 / 8 / 37376 | 63 / 0 / 37376 |
| E8 Kb3 klwarp + klload | 62 / 0 / 2240 | 60 / 0 / 3264 | 60 / 0 / 3264 |
| E8 **Kb7** + klpersist | 64 / 8 / 36608 | 64 / 8 / 37632 | 60 / 0 / 37632 |
| E16 Kb6 / Kb7 (klshape) | 123 / 0, 118 / 0 | 126 / 0, 120 / 0 | 115 / 0, 116 / 0 |

Static SASS instructions E8 Kb 2 / 3 / 6 / 7: sm_89 4840 / 4800 / 4976 / 5080, sm_90a 5008 / 4920 / 5184 / 5296, sm_120 4784 / 4808 / 4976 / 5200.
Occupancy: still 2 CTAs per SM (registers are the limiter); 2 x 37 KB of shared memory fits every card (sm_89 / sm_120 100 KB, sm_90 228 KB). On sm_90 the larger
static shared allocation reduces the L1 share of the unified L1 / shared array per SM (2 x 37 KB instead of 2 x 2 KB); klload's direct loads and chunk-1 re-reads
go through L1 / L2, so this is one of the things the duel measures, not a known neutral.

## Hazards (what can go wrong, and what checks it)

| hazard | guard | checked by |
|---|---|---|
| r0 read before the cp.async copies landed | `cp.async.wait_group 0` before the read; only the issuing thread reads its words | racecheck (mutation C below) |
| next tile's row info read before it is written | barrier between `setup_slot` and the prefetch (gmax-combine barrier for G > 32, an extra one for G <= 32) | hash tests (illegal address under mutation D), racecheck |
| row-info slot overwritten while still read | double buffer + the guarded end-of-tile barrier | racecheck |
| wrong data for the prefetched tile (wrong slot / base) | same row info function as the per-tile setup | hash tests (mutation A) |
| zero fill / tail / unaligned rows differ from fetch() | same predicates as fetch(): `hl + E <= k` and 16-byte address, else scalar with `h + e < k` | hash tests (mutation B), odd-pitch, tail, specials, zero rows |
| bit taken where klload is not | host clears it with klload; launch counter in the tests | `kl_bits`, `forced_sets` expect Kb 6 / 7 exactly where klload runs |
| no prefetch exercised (tiles <= grid) | `imd_test_set_kl_grid(1 / 3)` and three cases with > 2 x 144 tiles | `kl_bits`, `forced_sets`, `kl_persist_race` |
| graph capture | the kernel launch is an ordinary launch with a host-computed grid; no host sync, no allocation | unchanged launch path (no new test) |

## Tests and their evidence (RTX 4090, 2026-10-27, all GPU work under the GPU lock)

* `tests/test_seq_table.cpp` (CPU): seq_valid over 0..1023 (256 needs 32, nothing >= 512), kb mapping (421 -> 6), no class row carries 256. PASS.
* `tests/test_seq_select.py`: all valid sets of the 8 forceable bits (incl. every 256 + 32 combination) forced and read back; 256, 257, 272, 256 + 5, 256 + 133, 511,
  512, 1024 ... rejected; 421 with the K5 CRT hook. PASS (sm_89 default 1).
* `tests/test_kl_bits.py`: 19 cases (16 old + 3 with more tiles than 2 x SMs: 600 x 4096, 300 x 8192, 9301 x 70) x emitter sets {0, 1, 5, 7} x the 12 KL
  combinations; the 4 klpersist combinations run with the rule grid and with the grid forced to 1 and 3. **1520 runs, PASS**: one hash per case over all sets, and the
  launch counter shows Kb 6 / 7 exactly where expected.
* `tests/test_forced_sets.py`: the R4 sets plus 293, 421, 309, 437, 357, 501 (crt f64 and int; the klpersist sets also with grid 3). **684 runs, PASS**, all equal
  to the baseline (set 0, integer CRT); k5fin counted for 421 / 437 / 501.
* `tests/test_kl_persist_race.py` (ctest `kl_persist_race`, registered when compute-sanitizer is found): racecheck and synccheck over 8 cases x sets 289 / 305 /
  353 / 369 x grids 1 / 3. PASS (0 hazards, 0 errors). memcheck over the same cases with set 305 grid 3: 0 errors.
* ctests `r4_set_293_{f64,int}`, `r4_set_421_f64` (oracle e2e / eps + the 19 committed hashes under the forced set): see "Full suite" below.
* Mutation checks (by hand, source restored after each): (A) prefetch from the current tile's slot instead of the next: caught by `kl_bits --quick` (hash
  mismatches in the grid-forced runs); (B) scalar tail bound `<=` instead of `<`: caught (k1000_tail, g4_n14, odd_pitch_A); (C) `cp.async.wait_group` skipped at
  run time: **NOT caught by the hash tests on the 4090** (the copies land long before passes 2 / 3 end), caught by racecheck (32768 hazards per access) - hence the
  `kl_persist_race` ctest; (D) the short-row barrier skipped: `kl_bits` fails (illegal memory address from stale row info), racecheck fails.

### Full suite

`tools/test.sh` on this branch's final source (RTX 4090, compute-sanitizer found, test hooks on): CPU stage **34 / 34 passed**, GPU stage **131 / 131 passed**
(1638 s), rc = 0. Among them `host_seq_table`, `kl_bits`, `seq_select`, `r4_set_293_f64`, `r4_set_421_f64`, `forced_sets`, `kl_persist_race`.
The run exposed one registration bug: the r4_set loop skipped crt=int for every `seq > 127` (meaning "has k5fin" before bit 256 existed), so `r4_set_293_int` was
not registered. It now tests bit 128; `r4_set_293_int` was then run on its own: PASS.

## What R6 decides (pre-registered; written before any Hopper number)

`bench/sweep_r6.txt` on H100 and H200 (same job, interleaved duel): A = default (173), B = 421 + crt=f64.
**ADOPT** bit 256 into the `{9, 0, ...}` row (sm_90 becomes 429) if B / A >= 1.03 whole call on at least 2 of the 3 lines on **both** cards, no line below 0.98,
and every c_hash equal (`seq_eff` 173 / 421 in the records). Otherwise it stays off (KEEP). sm_89 / sm_120 stay as they are until their own data says otherwise.
Expectation (hypothesis, not measured): the gain, if any, is at 4096 and N = 8 where KL is a larger share and the per-tile load latency is exposed; at 8192 the
chunk-1 loads are not prefetched and the larger shared allocation costs L1.

## Walking back

The bit is off in every row, so nothing runs it unless forced. To remove it: drop `kSeqKlPersist` from `kSeqAll` (the hooks then refuse it and the Kb 6 / 7
instances are never launched), or revert this branch's commit; `ExecPlan::seq` can stay 16 bits (same struct size). If it is adopted later, walking it back is the
one `kSeqClasses` row, as for every other bit.

## Not done / limits

* No Hopper run: sm_90a is cross-compiled only (SASS, registers); runtime behaviour on Hopper is covered only by the same tests on the cluster.
* No timing (by instruction). The bench build was not completed in this session, so `bench/test_seq_crt.py` (with 421) was not run; the `seq_bits_ok` change is a
  one-line mask.
* Chunk 1 of two-chunk rows is not prefetched (the buffer would be 64 KB per CTA); only chunk 0 is.
* The performance model (`kl_launch`, occupancy) does not describe the persistent grid.
