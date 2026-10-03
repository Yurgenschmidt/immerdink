# M4 notes: performance

DESIGN.md is unchanged. Every change below is kernel-internal (tiling, thread mapping,
vectorization, launch configuration, an exact modular reduction realized differently), so
the semantics, the §4a workspace layout, the K0–K5 sequence and buffer reuse of
`spec/Pipeline.tla` and the public API are untouched. All 22 ctest tests pass bit for bit
after every step (they compare with `spec/oracle.py`; the invariance tests check that
results do not depend on tiling / launch choices). Design proposals that would need a
DESIGN change are at the end; none was implemented.

## What M4 built

| Piece | Where | Notes |
|---|---|---|
| Per-phase timing | `src/profile.{hpp,cpp}`, `src/profile_hooks.h`, `IMD_PROF` marks in `contract.cu` / `gemm_k4.cu` | CMake option `IMMERDINK_PROFILE` (default OFF; `bench/` turns it on). OFF: `IMD_PROF` expands to nothing, no symbol is exported. ON: a mark returns at once unless the bench called `imd_prof_enable(1)`; then it records a CUDA event with a static tag. `imd_prof_read` returns the time of each phase (event to event), `imd_prof_plan_dims` the post-swap GEMM view. |
| Bench phase report | `bench/imd_bench.cu` (`profile_imd`), `bench/summarize.py` | After the timed loop (marks off), `--phase-reps` (5) more calls with marks on; the record gets `phases`: per phase the median ms, its MEMORY_MODEL §4 lower bound (same formulas and datasheet peaks as `summarize.py`) and the % of that bound. `summarize.py` prints a measured-phase table and, with `--before`, a before/after table. |
| Test hooks only in test builds | `CMakeLists.txt` | `src/test_hooks.cpp` (`imd_test_set_limits`, `imd_test_lowering`) is compiled only with `IMMERDINK_BUILD_TESTS=ON` (PLAN M4 item 6, NOTES-M2M3 #7). The bench library (tests OFF) has neither hook; `harness.py` already checks for them. |
| Pack residues by byte weights | `src/device/modp.hpp` (`mod_p_bytes`), `src/device/residue.hpp`, `PackEmitter` | `mu mod p` was a 64-bit `%` per residue (≈100 instructions). Now `mu = Σ b_i 2^(8i)` and two `dp4a` with the weights `2^(8i) mod p` (derived at compile time from the generated `pow2_mod` table, no new constants), then one `umulhi64` reduction, exact for every sum < 2^32. The rare `eps > 0` factor and the sign follow. Same congruence class, so the same int8. |
| Pack tile shape | `contract.cu` (`k23_pack`, `pack_tile_lg_rows`) | 4096-element tiles; when reading along rows the height is the power of two ≥ D (rows along the pivot), clamped to [8, 64], and the tile is wider in k̄ instead (NOTES-M2M3 #6: a stride-1 batch pivot of extent 16 used 16 of 64 rows). Shared memory is padded by one word per 16 so the 16-wide conversion reads are conflict-free. |
| K4 swizzle | `gemm_k4.cu` | `GemmIdentityThreadblockSwizzle<8>`: a wave of 128 CTAs covers about 16 × 8 tiles instead of 128 × 1, so its A and B panels stay in L2 at large k. The tile offset's `k()` is still `blockIdx.z` (the batch index `VisitorModP` reads). |
| K1 over pack tiles | `contract.cu` (`k1_tiled`) | When the pack reads along rows (the operand's smallest stride is a row / batch mode), K1 takes the pack's row tile over all of k̄, so its reads are coalesced too (NOTES-M2M3 #5). Same two passes, same integers; per-row partial results are combined in shared memory (max, or, u128 sum: exact and order-independent). |
| K1 grid | `contract.cu` (`k1_blocks`, `k1_scale<T>`) | Grid capped so the rows in flight (about 40 MB) stay in L2 for the second pass; 1024-thread blocks when that leaves fewer than 2 blocks per SM (long or few rows). |
| K5 mapping | `contract.cu` (`k5_quad`, `k5_crt`, `k5_crt_bfast`), `index_map.cpp` (`k5_batch_fastest`) | 4 consecutive j̄ per thread (one u32 load per plane). When C's fastest label is a batch label (NOTES-M2M3 #4), a block computes a λ × j̄ tile with j̄ fastest (coalesced W reads), stages D in shared memory and writes C λ-fastest. |
| K5 CRT step | `kernels/numerics.cuh` (`crt_mac_sub`, `crt_step`), `modp.hpp` (`mod_p_small32`) | `x += M_l y; if x ≥ P: x −= P` as one PTX carry chain (lo and hi `mad` chains, then `x − P` whose borrow selects); `y = (q W) mod p` by a 32-bit magic `⌈2^32/p⌉` (derived from `epi_magic`). All N plane loads are issued before the chain. |

Exactness of the new reductions (CPU, `test_mod_exhaustive`, the kernels' own host/device
code): `mod_p_small32` for every n < 2^16 and every modulus; for `mod_p_bytes` every byte
weight against repeated doubling, the final reduction over its whole input range
[0, 7·255²], and 2^20 pseudo-random plus edge `mu < 2^54` per modulus against `mu % p`.
The PTX carry chain is checked end to end by the contract suites (every N in 2…20, so every
limb count 1…5). Mutation checks during development (reverted): scaling the λ-fastest K5
stores and an off-by-one σ in `k1_tiled` are both caught by `contract_random` /
`contract_highrank` / `contract_robust`, so the new paths are exercised.

## Log (CUDA-event times via `tools/run_bench.sh`, N = 14, ms)

The first column is this branch before any change (dev `14dbcdf` plus the instrumentation),
measured the same way. "Whole call" is the timed median (marks off); phases are the
medians of the profiled calls.

| case | step | K1 | pack | K4 | K5 | whole call |
|---|---|---|---|---|---|---|
| sq8192 | baseline | 1.31 | 9.98 | 24.9 | 3.15 | 39.9 |
| | 1: dp4a residues, K4 swizzle | 1.31 | 3.41 | 25.7 | 3.25 | 32.7 |
| | 2: pack tiles, K1 tiles, K5 quad / λ-fastest | 1.31 | 3.39 | 25.8 | 2.73 | 32.0 |
| | 3: PTX CRT step | 1.31 | 3.38 | 25.3 | 2.28 | 31.5 |
| | 4: K1 1024-thread rows | 1.31 | 3.40 | 25.8 | 2.31 | 31.4 |
| | 5: K5 load hoisting | 1.31 | 3.37 | 25.2 | 1.94 | 31.8 |
| sq16384 | baseline | 9.20 | 41.8 | 300 | 13.2 | 363 |
| | 1 | 9.20 | 13.8 | 201 | 13.2 | 242 |
| | 2 | 6.27 | 13.7 | 204 | 10.9 | 238 |
| | 4 | 6.40 | 13.8 | 210 | 9.25 | 233 |
| t_ikl | baseline | 5.01 | 7.69 | 1.15 | 1.78 | 15.6 |
| | 1 | 4.99 | 1.99 | 1.12 | 1.76 | 9.88 |
| | 2 | 0.466 | 0.896 | 1.21 | 0.709 | 3.30 |
| | 3 | 0.466 | 0.892 | 1.21 | 0.540 | 3.12 |
| | 5 | 0.449 | 0.857 | 1.12 | 0.480 | 2.94 |
| t_cc | baseline | 0.370 | 2.61 | 3.42 | 0.818 | 7.26 |
| | 2 | 0.359 | 0.834 | 3.24 | 0.641 | 5.13 |
| r_bigk | baseline | 2.32 | 9.42 | 5.97 | 0.189 | 18.0 |
| | 2 | 1.96 | 3.38 | 6.03 | 0.186 | 11.7 |
| | 4 | 2.02 | 3.38 | 6.03 | 0.156 | 11.8 |
| r_skinny | baseline | 0.677 | 5.30 | 2.08 | 0.402 | 8.52 |
| | 2 | 0.674 | 1.75 | 1.73 | 0.353 | 4.59 |
| r_mm | 2 | 1.24 | 1.36 | 0.909 | 0.015 | 3.97 |
| | 4 | 0.658 | 1.36 | 0.908 | 0.012 | 3.45 |
| r_lowk | 2 | 0.198 | 0.430 | 4.75 | 2.68 | 8.14 |
| | 5 | 0.199 | 0.429 | 4.75 | 1.92 | 7.37 |
| t_nn | 5 (full sweep `fc91409`) | 0.919 | 0.835 | 3.23 | 0.467 | 5.53 |
| | 6: K1 tiles of 8 rows, 1024 threads | 0.511 | 0.835 | 3.23 | 0.47 | 5.11 |

Steps: 1 = `d537234`, 2 = `c0840d8` + `7c4948f`, 3 = `2ef197c`, 4 = `73bf69a`,
5 = `5618296`, 6 = `b29b464`. The final sweep (all methods, all cases, commit `b29b464`)
is in docs/BENCHMARKS.md ("M4 sweep") with the before/after table and the measured
per-phase table.

K4 moves by a few % between runs with no K4 change (the card runs these at its 450 W
power cap, throttle reason 0x4), so K4 differences below ≈ 3 % are noise.

Notes on the baseline profile:
- **sq16384 (55 % of bound) was K4 and the pack.** K4 took 300 ms against 186 at the bound
  (the identity swizzle streams 128 distinct 2 MB A panels per wave: ≈ 260 MB of DRAM
  traffic per wave against ≈ 0.22 ms of math, so the GEMM was memory-bound; at 8192 the
  panels still fit L2). The pack was compute-bound on the 64-bit modulo (42 ms for
  ≈ 2·2.7 GB of traffic). Both fixed: K4 204 ms (91 % of bound), pack 13.7 (86 %).
- **t_ikl (19 % of bound) was every non-GEMM phase.** A is `[i][k][l]`, B `[k][j][l]`, C
  `[i][j][l]`: the stride-1 label is the batch label l (extent 16). K1 read rows along
  k̄ at stride 16 (B: 16384), the pack's 64-row tiles held 16 valid rows, and K5 wrote C
  at stride 16 with j̄ fastest.

## K4 tile and stage A/B (sq8192, sq16384, sq2048, t_batch, r_skinny, r_mm, r_lowk)

Built as separate bench binaries (`-DIMD_K4_TM/TN/WM/WN/STAGES`, `-DIMD_K4_SWIZZLE`; these
macros exist for this only), K4 phase medians in ms:

| config | sq8192 | sq16384 | sq2048 | t_batch | r_skinny | r_mm | r_lowk |
|---|---|---|---|---|---|---|---|
| 128×256×64 s3, swizzle 8 (kept) | 24.9 | 202 | 0.457 | 1.12 | 1.71 | 0.909 | 4.75 |
| 128×256×64 s3, swizzle 1 | 25.0 | 301 | 0.494 | 1.21 | 2.07 | 0.909 | 4.50 |
| 256×128×64 s3, swizzle 8 | 25.7 | 204 | 0.445 | 1.07 | 1.69 | 0.919 | 4.52 |
| 128×128×64 s4, swizzle 8 | 27.5 | 222 | 0.517 | 1.31 | 1.85 | 0.884 | 4.99 |
| 128×128×64 s5, swizzle 8 | 27.6 | 221 | 0.556 | 1.29 | 1.84 | 0.886 | 5.00 |

256×128 is within run-to-run noise of 128×256 (the 4090 runs these at its 450 W power cap,
`clocks_event_reasons` 0x4, so the K4 time moves by a few % between runs); 128×128 is slower
everywhere. A wave-quantization tile menu would not buy anything on these shapes, so it is
not instantiated (each config is two more CUTLASS kernels to compile). The swizzle matters
only at large k, and there it is a 1.5× win.

## Remaining gaps (where the time still goes)

- **K4** is 85–97 % of its bound at k ≥ 4096 (the bound uses the datasheet 660 TOPS; the
  best cuBLASLt int8 rate on this box is ≈ 617 TOPS, so 93 % of datasheet is "at
  cuBLASLt"). At k = 1024 (t_batch, t_ikl, r_lowk) it is 60–65 %: the CUTLASS 2.x
  epilogue (mod p, u8 store of N planes) runs after the main loop and is not overlapped.
  Still faster than N × cuBLASLt int8 (int32 out) on those shapes.
- **K5** is at ≈ 75 % of its bound.
- **K1** is 65–84 % at k ≤ 8192 (55 % for B of `t_nn` / A of `t_permout`, read across
  rows) and 55–67 % for long rows (r_bigk 256 KB rows, sq16384): the second pass
  re-reads part of each row from DRAM. See proposal 2.
- **K0** (the H2D copy of the scatter vectors) costs 0.03–0.37 ms, far above
  `bytes / PCIe`: the copy is a separate small DMA, and when the GPU is not busy the PCIe
  link sits in a low-power state (the measured rate is ≈ 2 GB/s). It is 6–9 % of r_mm and
  sq1024. See proposal 1.
- Small shapes (sq1024 0.25 ms vs GEMMul8 0.175) are dominated by fixed costs: 8–10
  launches, K0, and K4 at small k.

## Design proposals (not implemented; each changes DESIGN text and/or a spec)

1. **K0 as a device kernel instead of an H2D copy.** The scatter vectors are a pure
   function of the fused modes (≤ 15 modes × 3 tensors of (extent, stride)), which fit in
   a kernel argument. A K0 kernel would write exactly the same `vec` region contents at
   the same point of the sequence, removing the PCIe copy (0.03–0.37 ms per call) and the
   plan's pinned buffer.
   *Spec impact:* `Pipeline.tla` is unchanged (K0 still writes `vec` once, before K1).
   DESIGN §4's K0 line changes, and §6's plan-lifetime rule ("a plan must outlive every
   in-flight call because K0 reads its pinned buffer") could be relaxed; `ALLOC_FAILED`
   from `cudaMallocHost` would disappear. The vec region could even be dropped, but that
   is a §4a layout change; keep it.
   *Payoff:* 5–9 % on small or long-k shapes (r_mm, sq1024, sq2048), < 1 % at 8192³.
2. **K1 split into a γ pass and an S pass across the whole operand.** A first launch
   writes γ (int16) into `sA`/`sB` and the flags into `fA`/`fB`; a second reads γ back and
   overwrites `sA`/`sB` with σ. Each pass is a single coalesced streaming read with full
   parallelism (no per-row second pass that must hit L2), so K1 would sit at ≈ 2 reads at
   full bandwidth for any row length — today it is 1–2 reads depending on L2 hits.
   *Spec impact:* a kernel-sequence and buffer-reuse change (K1 becomes two kernels and
   `sA`/`sB` hold an intermediate value): `Pipeline.tla` needs the K1a → K1b edge and the
   "σ final before K2" invariant. *Payoff:* r_bigk ≈ 0.5 ms (4 %), sq16384 ≈ 2 ms (1 %).
   Not worth a design change now.
3. **CUTLASS operators built once per plan** (NOTES-M1 gap 10). Removes the per-launch
   host work of `GemmUniversalAdapter::initialize` (and fixes the thread-safety note in
   §6). Host-side only, so it helps only when the host cannot run ahead of the GPU (tiny
   shapes); it changes the plan's contents, not the API. Left for later.

## PLAN M4 item 6 (hygiene)

- Test hooks only with `IMMERDINK_BUILD_TESTS=ON`: done.
- `oracle.lower` building scatter vectors lazily, and TLA+ models for the n̂ grid split and
  the κ ≥ T scope: not done (spec / oracle work, outside this kernel pass).
