# Issue 23: the fixed part of the large-row pack KL (three cuts, default off) and the KL section timers

Status: implemented and bit-identical on the RTX 4090; **no performance claim**: the 4090's KL is DRAM-bound, the target is H100 / H200 where KL is latency / issue
bound. Everything below that is a number from the 4090 is an instruction count or a resource figure (structural), never a time. Rule R4 decides on Hopper data.
Written for a reader new to the project.

## The problem

R3 phase budget, H100, N = 14: the square 4096 call is 2.13 ms = KL (pack A + B) 0.69 ms (32%) + K4 1.217 + K5 0.22; at 8192 KL is 2.48 of 13.5 ms. A linear fit in N
gives KL = 0.0085 ms per modulus + **0.228 ms fixed per pack** at 4096 (0.035 + 0.754 at 8192): the part that does not scale with N (scale, quantize, gather,
reductions) is 66% of KL at N = 14 and 77% at N = 8, 3.1x off the memory floor. The per-modulus emitter is already at the integer-pipe peak (about 7.7 lane
instructions per element and modulus), so **the emitter is not touched**. This issue cuts the fixed part, each cut as its own bit of the instruction-sequence set
(the E1 / Q1 / F1 mechanism of docs/NOTES-21.md), default OFF in every class row; R4 duels them on Hopper.

## What is redundant (static analysis, `kl_scale_pack<8, 2, 1>`, W = 2, E1, sm_89, `-lineinfo`, nvdisasm -g)

One CTA owns one 4096-element row (th = 1, tk = 4096) with 512 threads = 16 warps, 8 values per thread. Static non-NOP instructions of the kernel: 4096 (sm_89).
Counts per source line (static: the nc > 1 branches and the unrolled copies are in them, so they overstate what one thread executes at nc = 1; the executed counts are below):

| what | static instr | executed by | why redundant |
|---|---|---|---|
| the cross-warp combine of gmax and the flag: `for (w = w0; w < w0 + nw; ++w) gmax = max(gmax, s_wg[w]), fl \|= s_wf[w]` | 142 | **every thread**, 16 iterations (nw = 16) | one warp's lanes could take one entry each and shuffle-reduce (4 steps) |
| the same for S: `for (w ...) S += s_ws[w]` (u128 adds) | 167 | **every thread**, 16 iterations | same |
| `largest_u` (u128 shift, bit length, a 256-bit shift and compare) | about 360 (a serial dependent chain) | lane 0 of **every one of the 16 warps** (a warp with one live lane issues the full instruction count) | the row has one sigma: one warp is enough, the others wait at a barrier instead of issuing 16 copies |
| `D = b rows / radix`, `n_d`, `tiles`, `nc` (64-bit division, shifts, multiplies) | about 115 plus the `div_s64` subroutine | **every thread**, once per kernel | functions of the launch arguments: host side (`kl_launch` already has them) |
| `t % n_d`, `t / n_d` (64-bit) and `divmod(rho, rows)` (twice: the tile setup by th threads, the store offsets by every valid thread) | about 70 per divmod | every thread | once per row (th threads), kept in shared memory |
| the gather `load(c)`: per element `s_valid`, `s_base` loads, the `k_affine` / `along_rows` selects, 64-bit address, STS, later LDS | 440 (4 copies, about 110 executed per thread) | every thread | for unit-stride k the thread's 8 consecutive doubles are one 64-byte contiguous piece: four 16-byte loads straight into registers |

Dynamic confirmation (ncu on the 4090, 4096^3, N = 14, E1, one KL launch = one operand, 4096 CTAs of 16 warps; warp instructions executed, `smsp__inst_executed.sum`;
instruction counts are structural and independent of the card's bottleneck). The C hash is `791030bff12bab6a` in all of them:

| seq | kernel | warp instr | per warp | vs baseline |
|---|---|---|---|---|
| 1 (baseline KL) | `<8,2,1,0>` | 196.3 M | 2995 | |
| 17 klwarp | `<8,2,1,1>` | 165.1 M | 2519 | -15.9% |
| 33 klload | `<8,2,1,2>` | 180.9 M | 2760 | -7.9% |
| 49 klwarp + klload | `<8,2,1,3>` | 151.7 M | 2314 | -22.7% |
| 65 klshape | `<16,2,1,0>` | 161.9 M | 2470 | -17.5% |
| 113 all three | `<16,2,1,3>` | 131.2 M | 2002 | -33.2% |

The same at square 8192 (nc = 2, 8192 CTAs, 5301 warp instructions per warp in the baseline, C hash `d711cf266c9e1c81` for seq 1, 49 and 113): 694.8 M baseline; klwarp 634.2 M (-8.7%);
klload 634.9 M (-8.6%); both 576.7 M (-17.0%); klshape 612.0 M (-11.9%); all three 512.4 M (-26.3%). (The fixed part is half as large a share there: nc = 2 doubles the S and
emit work per row while the row-wide scalar work stays; klload with nc = 2 includes its two extra re-reads of chunk 1.)

So the fixed part really is about a third of the executed instructions of a pack at N = 14, and klwarp alone removes about 480 warp instructions per warp.
The section timers below split the same kernel in cycles.

## The three bits (`src/host/seq_config.hpp`)

| bit | name | what the launch / kernel does | applies when |
|---|---|---|---|
| 16 | `klwarp` | (a) the 64-bit launch scalars (`D`, `n_d`, `tiles`, `nc`) come from the host in `LargeArgs` (new fields appended: no earlier field moves); (b) the tile's row info (`t / n_d`, `rho`, `divmod`, the base and the output offset of every row) is computed once per row by th threads and kept in shared memory; (c) the cross-warp combines of gmax / flag / S are lane-parallel: lane l of a warp reads row-warp l's entry, shuffle reduction over nw lanes; (d) `largest_u` runs **once per row** (lane 0 of the row's first warp), sigma goes through shared memory (one more `__syncthreads` when a row spans several warps; none when a row is inside a warp); (e) no final barrier after the last tile | every KL launch (any E) |
| 32 | `klload` | the gather tile and its barriers are gone: each thread loads its E consecutive values of a chunk straight from global memory into registers, 16-byte loads (`double2`) when the thread's address is 16-byte aligned and all E values are inside the row, else per element (zero outside the operand as the gather did). Chunks after the first are re-read from global memory (L1 / L2) in pass 2 and pass 3 (the gather path kept the last chunk in shared memory) | the operand is read along k-bar (not the side-by-side row read), `k_affine` with stride 1, at most two chunks (nc <= 2), and the operand base pointer is 16-byte aligned (host check; the kernel checks every thread's address again, so an odd row offset or pitch falls back per thread); otherwise the gather runs |
| 64 | `klshape` | 16 values per thread, 256 threads per CTA (`kl_scale_pack<16,...>`, launch bounds 256 x 2) instead of 8 / 512 for the rows held on chip | nc <= 2 (rows of more than two chunks keep their 4-values-per-thread kernel) |

The kernel has a template value `Kb` (bit 0 klwarp, bit 1 klload) next to the pack sequence `Seq`; klshape is the launch's E. `Kb = 0` is the original kernel: see
"SASS identity" below. The cuts are independent of the emitter sets and of each other: every combination of the bits 1, 2, 4, 16, 32, 64 is a set except Q1 without E1
(48 sets); bit 8 (the limb CRT) stays a class-default-only bit. They are kernel twins like Q1 / F1: **with `IMMERDINK_SEQ_TWINS=OFF` they are not compiled and the hooks
refuse them**. Bit 128 is taken by another change (K5): expect a trivial merge conflict in `seq_config.hpp`.

### Plumbing (as E1 / Q1 / F1)

* bit constants, `kSeqAll` (= 7 | kSeqKl), `kSeqSupported`, `seq_kl_kb`, `seq_kl_shape`, `seq_valid` (the mask-based check: `bits & ~kSeqSupported == 0`, Q1 needs E1)
  and the class rows `kSeqClasses` (**all KL bits off in every row**: sm_89 stays 1, sm_90 stays 13, sm_120 and unlisted 0; `seq_default_for` masks to
  `kSeqDefaultMask`, so a row may carry a KL bit later, per device class, with that class's own data) in `src/host/seq_config.hpp`;
* hooks: `imd_test_set_seq` / `imd_prof_set_seq` accept every supported combination (it was 0..7 by hand; now the mask), reject Q1 without E1, bit 8, the unsupported
  bits of a no-twins build and everything out of range, and leave the forced set unchanged on rejection; `imd_test_plan_seq` / `imd_prof_plan_seq` (the `seq_eff` of the
  records) read back the full set; `ExecPlan::seq` is a `uint8_t` (bit 7 is free for the K5 change);
* bench: `imd_bench --seq <bits>` and the duel side key `seq=<bits>` take a decimal number (bits 1 2 4 16 32 64; `--seq 8`, `24`, `x`, `-1` are refused before any GPU
  work, `bench/test_seq_crt.py` checks that and runs 17 and 113); the records carry `seq_req`, `seq`, `phase_seq`, `seq_eff` as before;
* tests: `tests/test_seq_table.cpp` (CPU: all 128 values, rows, kb / shape mapping), `tests/test_seq_select.py` (all 48 sets forced and read back, the rejection list),
  `tests/test_kl_bits.py` (below), ctest `contract_{e2e,tensor,eps}_kl{17,33,65,113,112}`, `contract_bigk_kl{17,113}`, `output_hashes_kl{...}`,
  `output_hashes_large_kl{17,113}` (the existing oracle suites and committed hashes under `IMD_TEST_SEQ`, whose read-back fails the run if the set did not take effect).

## SASS identity with the bits off (and on sm_89, sm_90a and sm_120)

The owner's requirement: with bits 16 / 32 / 64 off, the shipped KL kernels (and every other kernel) are byte-identical to `origin/port/hopper-blackwell` on sm_89,
**sm_120** and sm_90a. `tools/sass_ident.sh <tag> [nvcc flags]` compiles `src/contract.cu` alone for the three architectures (cross-compiles; sm_90a and sm_120
cannot run here) and `tools/sass_hash.py` hashes, per kernel, the instruction stream of `cuobjdump -sass` with addresses, encodings and the kernel's name removed (so a
template parameter that renames a symbol is not a difference), then `--cmp` / `comm` compares the multisets. Baseline = a checkout of `origin/port/hopper-blackwell`
(commit e4425ea), this branch compiled with the same flags (`-O3 -DNDEBUG -std=c++17 --expt-relaxed-constexpr`, nvcc 13.4.92):

| build | sm_89 TOTAL (kernels) | sm_90a | sm_120 |
|---|---|---|---|
| baseline `origin/port/hopper-blackwell` | 027aaaad5adad44a (351) | d577418bacbc973b (351) | e6f03535c743394e (351) |
| this branch, bits off (no hooks = production) | 207c2589b4f8bea7 (391) | 76ad9ecc3b4d6568 (391) | fa3a3646f33565ff (391) |
| this branch, `-DIMD_BUILD_TEST_HOOKS=1` (a test build) | 20ade282324208b9 (581) | 3720cccc1888b64e (581) | bcacb6a9810c66ed (581) |
| this branch, `-DIMD_KL_TIMERS=1` | d10947be9cc135dd (391) | a9a5c7753b28b3c9 (391) | be200bc3775cb5ba (391) |

TOTAL is the hash of the sorted kernel hashes, so it differs where kernels were added (+40 = the new KL instances; +190 in a test build = the test-only kernels). The
statement that matters is per kernel: **all 351 baseline kernel hashes are present in the bits-off build on all three architectures** (`comm -23 base new` is empty), in
particular the 10 shipped KL kernels `kl_scale_pack<E,W,S>` (E = 4 / 8, W = 2 / 3, S = 0 / 1 / 3) have the same hash as the off instances `<E,W,S,0>` (10 / 10 on each
architecture); the same holds with the test hooks compiled in. With the timers ON exactly the 10 KL kernels change and the other 341 are identical. `-lineinfo` does not
change a hash. To redo it: `tools/sass_ident.sh base` in a checkout of the baseline, `tools/sass_ident.sh new0` here, then `comm -23` on the sorted first columns of
`build-sass23/base_<arch>.hash` and `new0_<arch>.hash`.

The off path is unchanged because every cut is an `if constexpr` on a template value, the new `LargeArgs` fields are at the end of the struct (earlier fields keep their
constant-bank offsets), `klshape` is a different launch of a different instance, and the timers are `#ifdef` code. A cut that needs a runtime branch would not satisfy this
and none has one.

## Resources (ptxas, `cuobjdump --dump-resource-usage`; W = 2, E1 = the sm_90 default pack, S = 1; W = 3 and S = 0 / 3 in `build-sass23/*.res`)

Registers per thread / local stack bytes / static shared bytes. Occupancy = CTAs per SM from registers (65536 per SM, allocated in units of 8 per thread), threads per CTA
and shared memory; resident threads per SM out of 2048 (sm_90) or 1536 (sm_89, sm_120).

| variant | threads | sm_89 | sm_90a | sm_120 | CTAs/SM (all three) | resident threads/SM |
|---|---|---|---|---|---|---|
| E8 Kb0 (baseline) | 512 | 64 / 8 / 36288 | 64 / 24 / 37312 | 64 / 0 / 37312 | 2 | 1024 |
| E8 Kb1 klwarp | 512 | 60 / 0 / 37568 | 60 / 0 / 38592 | 60 / 0 / 38592 | 2 | 1024 |
| E8 Kb2 klload | 512 | 64 / 16 / 960 | 64 / 16 / 1984 | 64 / 0 / 1984 | 2 | 1024 |
| E8 Kb3 both | 512 | 62 / 0 / 2240 | 60 / 0 / 3264 | 60 / 0 / 3264 | 2 | 1024 |
| E16 Kb0 klshape | 256 | 128 / 0 / 36096 | 126 / 0 / 37120 | 124 / 0 / 37120 | 2 | **512** |
| E16 Kb1 | 256 | 107 / 0 / 37376 | 107 / 0 / 38400 | 106 / 0 / 38400 | 2 | 512 |
| E16 Kb2 | 256 | 126 / 0 / 768 | 128 / 0 / 1792 | 119 / 0 / 1792 | 2 | 512 |
| E16 Kb3 | 256 | 116 / 0 / 2048 | 110 / 0 / 3072 | 106 / 0 / 3072 | 2 | 512 |
| E4 Kb0 / Kb1 (rows of > 2 chunks, 1024 threads) | 1024 | 64 / 8 / 36672, 59 / 0 / 37952 | 64 / 8 / 37696, 58 / 0 / 38976 | 64 / 0 / 37696, 62 / 0 / 38976 | 1 | 1024 |

Static SASS instructions (including NOPs; W = 2, S = 1): sm_89 E8 5832 / 5776 / 4840 / 4800 (Kb 0..3), E16 9184 / 9176 / 7648 / 7680; sm_90a E8 6128 / 6096 / 5008 / 4920,
E16 9472 / 9496 / 7832 / 7824; sm_120 E8 6408 / 6384 / 4784 / 4808, E16 9656 / 9720 / 7712 / 7768.

Reading: klwarp and klload do not cost registers (they lower them) and keep two CTAs per SM; klload frees 34 KB of shared memory per CTA (not the limiter here: registers
are). **klshape is the only variant that costs something**: 107-128 registers and half the resident warps (512 instead of 1024 threads per SM, 16 instead of 32 warps), in
exchange for 17.5% fewer executed instructions. It was also tried with 3 resident CTAs (launch bound 256 x 3, 80 registers): 40-96 bytes of spills (Kb 1 alone has none),
and with 4 (64 registers): 152-224 bytes of stack and up to 668 bytes of spill loads: clearly worse, not shipped; the shipped instance is (256, 2). The baseline kernels
already spill 4-76 bytes at 64 registers (existing behaviour, sm_90a most).

## Tests, and the test of the test

* `tests/test_kl_bits.py` (ctest `kl_bits`): 16 contractions x {baseline, E1, E1 + F1, E1 + Q1 + F1} x all 8 combinations of the KL bits = 512 runs, forced KL for both
  operands (`imd_test_set_fused(2)`, or 3 for 64-row tiles): one chunk (k = 4096), a tail (k = 1000), two chunks (8192), three chunks (8300: the 4-values-per-thread kernel),
  tiny rows (k = 70, 17: several rows per tile, G < 32), g = 4 and a batched g = 3, an odd row pitch (row bases not 16-byte aligned), the base pointer one and three doubles
  off a 16-byte boundary, a transposed operand (the gather stays), 64-row tiles, zero rows, Inf, NaN, subnormals and an all-non-finite row. One hash per case over all sets;
  the KL launch counter (`imd_test_kl_launch_counts`, recorded by `launch_kl_kb` in test builds) must show per operand the kernel the set asks for (E = 4 / 8 / 16 and
  `Kb`, klload only for a unit-stride k read along k-bar, nc <= 2, an aligned base). Without the counter a bit that silently ran the baseline would pass everything.
* Mutation check of this test (done once, by hand): a shuffle loop one level short in the klwarp S combine, and a tail bound one element too large in the klload per-element
  path. Both were caught by the quick run (hash mismatches, in the sets with bit 16 and with bit 32 respectively). The mutated source was restored.
* The existing suites run under the forced KL sets (IMD_TEST_SEQ): see the list under Plumbing; results in "Results" below.

## Results (4090, 2026-10-26, this branch)

`tools/test.sh` (full; build -j4, GPU lock): **34 / 34 CPU tests and 98 / 98 GPU tests passed**, including `host_seq_table`, `seq_select`, `kl_bits` (512 runs), `contract_{e2e,tensor,eps}_kl{17,33,65,113,112}`,
`contract_bigk_kl{17,113}`, `output_hashes_kl{...}` and `output_hashes_large_kl{17,113}` (the 19 committed hashes are reproduced under the forced KL sets, as under the class default), and all the
older suites (the sm_89 class default is E1, KL bits off). `bench_seq_crt` needs a profiling-build `imd_bench` (`IMMERDINK_BENCH_BIN`) and was not run here.

## The KL section timers (task A)

Compile-time knob, default OFF: `-DIMMERDINK_KL_TIMERS=1` (CMake cache string 0 / 1 / 2; defines `IMD_KL_TIMERS`) with `IMMERDINK_BUILD_TESTS=ON`. Lane 0 of every warp
stamps `clock64()` at the section boundaries of `kl_scale_pack` and adds the elapsed cycles to a shared-memory counter; thread 0 adds a CTA's counters to the device
array `g_kl_timers` at its end (16 slots: 11 sections, warp-tiles, CTAs, summed CTA lifetime). Level 2 also stamps around each modulus's emit and store (two more stamps per
modulus: perturbs the loop; use level 1 for the split, level 2 only to see emit against store). `imd_test_kl_timers(reset, out, n)` reads and optionally zeroes them (-1 in a build without the knob, a
synchronizing call), `Library.kl_timers()` in `tests/harness.py`. **`tools/kl_timers.py --lib <timers build>/libimmerdink.so [--shapes 4096 8192] [--seq ...] [--N 14] [--b-kn]`**
prints the dynamic split per section (cycles per warp-tile and share) for each shape and set, holding the GPU lock, and checks that the C hash is the same over the sets.
Sections: `setup` (tile row info and its barrier), `load` (pass 1: gather or direct loads and the flag / gamma scan, barriers included), `red1` (gmax / flag reduction),
`ssum` (pass 2: the u128 square sum, re-reads included), `red2` (the S reduction), `sigma` (`largest_u`, broadcast, row offsets, sigma / flag stores), `reload` (chunks re-read
in pass 3), `quant` (the emitter's quantize), `emit` (the modulus loop; at level 2 the emit only), `store` (level 2), `tail` (the closing barrier). Barrier waits sit in the section
that ends at the barrier; the numbers are sums over warps, so they split a warp's time, not the kernel's wall time. Rule R4: `sigma + red1 + red2` >= 10% of the CTA cycles
-> do B1 (klwarp, this issue); `ssum` >= 15% -> item 2 (FP64-pipe S; not part of this issue).

Demonstration run on the 4090 (square 4096, N = 14, E1, level 1; **not decision data**: DRAM-bound, `load` includes the memory wait): baseline `seq 1`: load 48.6%, emit
14.3%, ssum 7.8%, red1 6.1%, setup 5.4%, red2 5.3%, quant 4.8%, sigma 4.3%, tail 2.7% (sigma + red1 + red2 = 15.8%); `seq 49` (klwarp + klload): load 20.7%, emit 22.7%,
red1 18.8% (it now absorbs the wait for the loads that the direct path issues without a barrier), setup 11.7%. Cycle totals per warp-tile are 52.9 k and 51.5 k.
The hash is identical. The timers build's kernels differ from the shipped ones (extra registers, the clock reads): the split is a diagnostic, not a profile of the shipped
SASS. On the cluster run it on the H100 and H200 at 4096 and 8192 with `seq=5` (the sm_90 pack set; the plan default 13 also carries the limb CRT, irrelevant to KL).

## What R4 decides (pre-registered; written before any Hopper number)

* **Duels** (`imd_bench --methods duel`, N = 14, the large squares 4096 / 8192 and the other large cases of the campaign, H100 and H200): A side `seq=5,crt=f64` (the adopted
  sm_90 set spelled out), B sides `seq=21,crt=f64` (klwarp), `seq=37,crt=f64` (klload), `seq=69,crt=f64` (klshape); then the combinations of the winners (`seq=53`, `101`,
  `85`, `117` with `crt=f64`). Every record carries `seq_eff`; all C hashes in a case must be equal (they are by construction here, on the 4090, and the 19 committed
  hashes are in the test suite).
* **ADOPT a bit** (put it in the `{9, 0, ...}` row of `kSeqClasses`) if it gives >= +3% whole call on >= 2 N = 14 large cases on **both** H100 and H200, no case is more than
  3% slower, and everything is bit-identical; the floor is 0.98 (a case under 0.98 is a stop). Otherwise it stays off. The bits are per device class: the 4090 (sm_89) and
  Blackwell (sm_120) rows stay as they are until their own data says otherwise.
* **Timers first**: if the Hopper timers show `sigma + red1 + red2` < 10% of the CTA cycles at 4096 / 8192, klwarp has little to win there and is expected to tie; if
  `ssum` >= 15%, the next item is the FP64-pipe S (not here).
* Interactions: klload with klshape (E = 16: eight 16-byte loads per thread, 107-128 registers) and klwarp with either are separate instances; the combined duels exist
  because the effects need not add.

## Expectations per class (hypotheses, nothing is measured except instruction counts and registers)

| bit | sm_90 (H100, H200: latency / issue bound) | sm_89 (RTX 4090: KL DRAM-bound, 85% of peak in ncu) | sm_120 (RTX PRO 6000 Blackwell) |
|---|---|---|---|
| klwarp | **helps**: -16% executed instructions per pack at N = 14 (more at small N: the part is N-independent); the extra barrier of the one-warp sigma is the risk | neutral (instructions are not the limit); possibly a hair slower or faster from the barrier / shuffle trade | probably neutral to slightly positive (if it is also memory-bound, neutral) |
| klload | **probably helps** the operand that is read along k (A of C = A B in the usual layout; B only if stored [n, k]): -8% instructions, no gather barriers, no shared-memory round trip; the risk is the 64-byte lane stride of the direct loads (4 L1 wavefronts per 16 bytes of useful data at the LSU, the same bytes from DRAM) and the extra L1 / L2 re-reads of chunk 1 at 8192 (nc = 2) | neutral to **hurt**: the DRAM traffic is the same, the L1 access pattern is worse | like the 4090, unmeasured |
| klshape | **uncertain**: -17.5% instructions but half the resident warps (512 of 2048 threads), 128 registers; latency-bound code loses what it gains if the warps are what hid the latency | likely **hurts**: DRAM-bound code needs many loads in flight, and this halves the warps (33% occupancy) | likely hurts for the same reason |

Hence all three stay off outside sm_90 until a class has its own data, and on sm_90 until R4 adopts them.

## Not done / limits

* No Hopper or Blackwell hardware here: the sm_90a and sm_120 builds are cross-compiled (SASS identity, registers, spills), never run. The runtime path of the bits on a
  Hopper is checked only by the same tests on the cluster (`kl_bits`, `seq_select`, `contract_*_kl*`, `output_hashes_*_kl*`).
* The Hopper dynamic split does not exist yet (that is what the timers are for); the section shares above are a 4090 illustration.
* klload with nc = 2 re-reads chunk 1 from global memory twice more (L1 / L2); keeping it in registers would cost 16 registers and spill at the 64-register bound. Only the
  8192 duel can tell whether that matters; `nc <= 2` was kept in the rule so that the case is measured rather than excluded.
* klload does not apply to an operand read along rows (k not unit stride) or to a `v_k` vector K class (several fused K modes): they take the gather as before.
  An operand whose base pointer is not 16-byte aligned takes the gather (host check); an operand with an odd row offset or pitch takes the per-element direct path
  for those rows (still no gather).
* The performance model (`imd_prof_kernel_occupancy`, `kl_launch`) describes the baseline instances; the new instances are not modelled.
