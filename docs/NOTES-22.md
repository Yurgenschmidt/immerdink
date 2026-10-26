# Issue 22: the Hopper K4 ladder, rung 1 (timing)

Status: harness written and cross-compiled 2026-10-25, revised the same day after an independent review (APPROVE-WITH-CHANGES, 12 points, all
applied below); **never run on a Hopper card** (no Hopper on the dev box). Later the same day: the synthetic wgmma peak (`p_` rows, section "The synthetic peak") was
added; it is equally unrun. Code: `src/k4/hopper/rung1.cu`, `tools/build_rung1.sh`. Rung 0
(`rung0.cu`, exactness at tiny size) is unchanged except that `tools/build_rung0.sh` now also builds with `-DNDEBUG`.

## Why this exists

The stock CUTLASS wgmma ceilings of issue 16 (`bench/ceiling`) probably under-measure what a wgmma K4 can do. Section 3 of
`~/repos/immerdink-eval/findings/offline-theory-20261025.md` fits `t = O + n (aK + e)` to the stock R1b/R1c records: the mainloop-only rate of the
cluster variants is 1530-1570 TOPS-eq (about cuBLAS), and the per-tile boundary cost `e` is 5-9 us, 20-28% of a K = 4096 tile. The stock gap is the
epilogue (CUTLASS's `EpilogueScheduleAuto` resolved to the direct-store epilogue, 256 scalar STG per thread, default scheduler arguments), not the
mainloop. Our own variants (`f_*`: `K4ModEpilogue` + `HopperEpilogue`, 16-byte stores, no shared memory) have not been timed at all. Prediction for them,
batch 14, store mode: **4096^3: 1260-1456 TOPS** (f_256x128x128 2x1 and f_128x128x128 2x1 best, about 1400), i.e. 1.3-1.5x of today's mma.sync K4 (966).
Also from the theory: at 8192^3 batch 14 the stock kernels collapse (868 against 1152 at 4096); leading cause `max_swizzle_size = 1` (the default), rival
power throttling. Rung 1 tests both.

## What the harness does

For each configuration of the table `kEntries` (14 rows: `m_` = the product's mma.sync K4, `f_` = our fast epilogue, `g_` = the generic byte-store
epilogue kept for comparison) and each shape m = n = k in {4096, 8192}, batch L = 14 (the N moduli: 14 independent GEMMs in one launch, batch z uses
modulus `moduli[slot0 + z]`, b = 1), operands `A[z][m][k]`, `B[z][n][k]` int8 from a counter-based hash, output the u8 residue planes of W, store mode:
warm-up (3 launches plus about 0.3 s of load so the clocks ramp), then at least 10 repetitions in two modes, `single` (one launch between two
cudaEvents) and `loop` (R back-to-back launches, divided by R; R >= 8 and enough for 20 ms). Reference batches L = 4 and 1 follow.

**Same-session baseline (`m_128x256x64_mma_sync`).** The product's own K4 (`k4_ops_cfg0()`: the A2 kernel, mma.sync, 128 x 256 tile, 8 warps,
persistent with `sms x occupancy` resident CTAs, as the library launches it) is a row of the same table, on the same operands, shape, batch, checks and
timing code, in the same process. The `f_`/`m_` ratio therefore needs no cross-run 966 figure (that one is from another job, about 5% run-to-run noise,
and a 1.3x ratio against it inherits that noise). One launch over all 14 batches and the whole K per shape, not the library's chunked group launches.

**Order** (a timeout or a hang keeps the valuable lines; `starting <row> <shape> L slot0 sw raster clusters` is printed to stderr BEFORE each row, so a hang
names its row): (the 20 synthetic-peak rows (`p_`, below) are NOT part of this work list: they are a separate second process, see "Two passes"); (1) the `m_` baseline and every non-"last" `f_` row, default scheduler, first shape then the next, L = 14; (2) the scheduler sweep for the
six `sweep` rows at 8192^3: max_swizzle_size {1, 2, 4, 8} x raster {AlongM, AlongN} (CUTLASS `args.scheduler`; `VT<>` needed no change, the knobs are runtime
arguments and rung 0 is untouched), skipping points that resolve to the same (raster, swizzle) as the default or as an earlier point (the heuristic picks
AlongN for 128x256 and 128x128 tiles but AlongM for 256x128, so the duplicate differs per row); (3) with `--sustain S` a sustained block for the best `f_` row
at the last shape; (4) the `g_` rows; (5) L = 4 and L = 1 for everything; (6) the rows flagged `last`: `f_128x256x128_4x1_coop`, a cluster shape never launched
on any card (`--skip-last` drops it). `--plan` prints this work list without a GPU.

The configurations, the default shapes and batches and the sweep grid are one block of data tables (`kEntries`, `kDefaults`, `kSweepSwizzle`,
`kSweepRaster`, `kRefSlot0`). Each row has the compute capability it is for; the harness runs the rows of the device's major and exits 3 if there are none,
so a Blackwell (sm_100 / sm_120) port adds rows and a Variant-like kernel type, nothing else. Every JSON record carries `gpu` (device name) and `cc`.

**Cluster rows and the persistent grid.** CUTLASS sizes the persistent grid from `hw_info.max_active_clusters`; left 0, its heuristic launches fewer CTAs
than SMs for cluster shape 4 (116 on 132 SMs, per the review). The harness sets it from `cudaOccupancyMaxActiveClusters` with the kernel's real dynamic shared
memory (`--clusters queried`, default); `--clusters heuristic` leaves it 0 (how rung 0 and the stock ceilings ran); `--clusters both` runs every cluster row both
ways (at the same shapes and batches). Records carry `max_active_clusters_arg`, `max_active_clusters` (the value used), `max_active_clusters_occupancy`,
`max_active_clusters_cutlass` (CUTLASS's own query, which ignores dynamic shared memory) and the launched `grid_x/y/z`, `ctas_launched`.

Output (stdout; first a meta record `method: "rung1_meta"` with CUDA driver / runtime versions, NVML driver version, hostname, `SLURM_JOB_ID`, nvcc version (the
build script passes it), commit, L2 size and the argv; then one JSON line per row, shape, batch, scheduler and mode; conventions of `bench/ceiling`): `method`
"rung1", `variant`, `kind`, `m n k batch slot0`, `swizzle` / `raster` (requested; heuristic = CUTLASS default) and `swizzle_resolved` / `raster_resolved` (what the
scheduler actually uses), `mode` (single / loop / sustain), `launches_per_rep`, `reps`, `tops` (int8 TOPS = 2 m n k L / median), `tops_best`, `tops_worst`,
`median_s/min_s/max_s` and `_ms`, `spread_pct = 100 (max - min) / median`, the exactness fields below, `regs`, `local_bytes`, `smem_dynamic_bytes`, `gpu`, `cc`,
`sm_count`, `clock_khz_max` (cudaDevAttrClockRate), `ndebug`, `commit`, `status`. Error / skipped records carry `"exact": null`. `nvidia-smi` is not used; NVML is
`dlopen`ed (no link dependency, `--no-nvml` to skip) and polled every 5 ms by a host thread during each timed block: `sm_mhz_min`, `sm_mhz_mean`, `power_w_max`,
`temp_c`, `throttle_reasons` (NVML clocks-event bits except 0x1 and 0x2: 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake)
and `idle_app_flags` (0x1 idle, 0x2 application clocks: not throttling); all `null` without NVML. `nvmlDeviceGetPowerUsage` is an average over about a second, so a
short block under-reports `power_w_max`; for a power-limited rate use `--sustain 4` (the peak pass takes its own `--sustain`, default 8) (record `mode: "sustain"`: `tops` over the block and over its first and last
second, plus the clock and power over the whole block).

**Exactness** (`exact` = all three): `ref_mismatches == 0`: about 390 sampled output bytes (random positions plus corners and tile edges) against a host reference
from the definition (int64 sum over k, `((s mod p) + p) mod p`, p = `moduli[slot0 + z]`); `hash_eq_first`: a position-dependent hash of all of W equals the hash of the
first row run on the same (shape, batch, slot0), named in `hash_ref_variant`; `exact_after_timing`: the hash and the sampled reference recomputed after the timed
loops (about a hundred launches later) still equal the first hash (a race that only bites under sustained load). W is poisoned with 0xEE before the checked launch.
A failing row records the first sampled mismatch (`first_bad_z/i/j`, `first_bad_want`, `first_bad_got`). The hash comparison is relative to the first row: a fault
shared with it is seen only by the sampled reference. Whether the `m_` row is a good reference is checked by the sampled definition too (it passed the same
host reference on the 4090 in a smoke test, see below).

Exit status of one program run: 0 clean; 1 some row not exact; 2 usage / setup (unknown `--keys` are an error); 3 no CUDA device or no table row for this device (the 4090 dev box: harmless,
message on stderr); 4 sticky device error after its record; 5 every row exact but some record has status error / skipped (the count is printed); 124 / 137 from the
build script's `timeout -k 30` (a hang; the lines printed so far are in the file). In a `--variants p` run, rc 1 (a `p_` row not exact) and rc 5 (a `p_` row skipped / error,
or a peak-sustain item that found no exact row of its data kind) can come from the `p_` rows alone; see "Two passes" for why that no longer touches the primary results.

**Two passes (`tools/build_rung1.sh OUT run`).** The `p_` kernels have never run on any card, and a sticky fault or a hang in the first one would end a process (rc 4) or eat the whole
timeout before any `f_` row. So `run` starts TWO processes, one after the other: (1) the **primary pass**: the default selection (`all` = `m_`, `f_`, `g_`; the `p_` rows are not in `all`
or `quick`), the sweep, the reference batches and the GEMM `--sustain`, arguments `RUNG1_ARGS`, limit `RUNG1_TIMEOUT` (1800 s); (2) the **peak pass**: `imd_rung1 $RUNG1_PEAK_ARGS --variants p`
(`RUNG1_PEAK_ARGS` default `--sustain 8`), limit `RUNG1_PEAK_TIMEOUT` (600 s), `RUNG1_PEAK=0` skips it. Both passes run whatever the other's exit code. Files in `$OUT`:

| file | content |
|---|---|
| `imd_rung1`, `build.log`, `sass_counts.txt`, `nvcc_version.txt` | binary, nvcc -Xptxas -v log, IGMMA / DEPBAR / CALL table, nvcc version |
| `rung1.jsonl`, `rung1.err` | primary pass: stdout (meta record, then the GEMM / `m_` records), stderr (progress) |
| `rung1_peak.jsonl`, `rung1_peak.err` | peak pass: stdout (its own meta record, then the `p_` records), stderr |
| `rung1_status.json` | `{"primary_rc": N, "peak_rc": N or null}` (`null` = peak pass skipped with `RUNG1_PEAK=0`) |

The script's own exit code is the **primary pass's** (0, 1, 3, 4, 5, 124 / 137, or 6 for a refused build); the peak pass's code is only in `rung1_status.json`, and a non-zero one
prints a loud message on stderr ("the synthetic-peak pass exited N") but does not change the exit code, so a peak failure cannot fail the primary results. Both passes exit 3 on a card
without table rows (the 4090).

## Commands

Build and run (the cluster takes the GPU lock itself). Run rung 0 first, but note that **rung 0 does not certify rung 1**: they are different binaries and different
kernel instances (rung 0 on tiny problems, rung 1 at 4096 / 8192 with 14 batches); both are now built `-O2 -DNDEBUG`, so at least the machine code of a given variant
is the same in both. A rung-1 `exact: false` is a finding in its own right, whatever rung 0 said.

```
export IMMERDINK_CUTLASS_DIR=<cutlass checkout>          # default third_party/cutlass
tools/build_rung0.sh $OUT run                             # exit 0 first (now -DNDEBUG too)
BENCH_COMMIT=$(git rev-parse --short HEAD) RUNG1_TIMEOUT=1800 RUNG1_ARGS="--sustain 8 --clusters both" tools/build_rung1.sh $OUT run
#   $OUT/imd_rung1, build.log, sass_counts.txt, nvcc_version.txt, rung1.jsonl + rung1.err (primary pass),
#   rung1_peak.jsonl + rung1_peak.err (synthetic peak pass), rung1_status.json {primary_rc, peak_rc}
```

Quick (about a minute): `RUNG1_ARGS="--quick" tools/build_rung1.sh $OUT run`. Other knobs: `--variants` (groups all / f / g / m / p / quick / sweep as exact words; `all` = m_/f_/g_, `p` = the synthetic peak rows, else a name or
substring: `--variants g` is the generic rows only), `--shapes 4096,8192` (or MxNxK), `--batches 14,4,1`, `--sweep 8192|4096,8192|none` (uses the selected rows flagged
sweep, or `--sweep-variants`), `--reps`, `--max-seconds`, `--samples`, `--skip-last`, `--list`, `--plan`.
The default (primary) plan is 122 tasks (`--plan`; `--plan --variants p --sustain 8` is the peak pass: 20 rows + 3 sustain items); with `--clusters both` and `--sustain` more (unmeasured guess: tens of minutes; raise `RUNG1_TIMEOUT` for `both`). The build script checks CUTLASS
and nvcc exist, builds with `-Xptxas -v` into `build.log`, **refuses to run if ptxas printed C7510** (exit 6; `RUNG1_ALLOW_C7510=1` overrides: the numbers would then be of
serialised wgmma), prints spill bytes, and counts `WARPGROUP.DEPBAR` against `IGMMA` per kernel from `cuobjdump -sass` (`SERIAL?` if DEPBAR >= IGMMA). `-O2 -DNDEBUG`
are mandatory (the program warns and records `"ndebug": false` otherwise).

## Cross-compile evidence (nvcc 13.4, CUTLASS 4.8.0, sm_90a, -O2 -DNDEBUG; this machine, after the revision)

No ptxas C7510. No spills in any of the 13 CUTLASS kernels (including ping-pong 128x128, which spilled 8 bytes without NDEBUG in issue 17's table), 168 registers
per thread (consumers raise to 232 at run time), no stack frame; the product's mma.sync K4 (128x256x64, store / add): 254 / 246 registers, no spills. SASS per kernel,
`WARPGROUP.DEPBAR` against `IGMMA`: 2 DEPBAR in all 13, against IGMMA 4 (K64), 8 (128x256 and 128x128 coop) or 16 (256x128 coop, 128x128 ping), i.e. the wgmma
are issued back to back and drained twice, not serialised. One `CALL` per kernel remains (an out-of-line helper, not a serialisation; ptxas reports no C7510). Dynamic shared
memory per CTA and stages: 128x256x128 and 256x128x128: 196 864 B, 4 stages; 128x128x128: 229 760 B, 7 stages; 128x256x64: 221 568 B, 9 stages; mma.sync K4: 73 728 B.
Rung 0 with and without `-DNDEBUG`: 16 kernels either way, C7510 16 -> 0, spill bytes 32 -> 0, 168 registers.
Smoke test of the harness code on the RTX 4090 (a build for sm_89 with the table forced to the `m_` row and a 512 / 256 problem, under the shared GPU lock, a
functional check only; the rates it printed mean nothing): meta record, sampled reference, hash, post-timing check, `slot0`, JSON, NVML polling and exit codes all work;
the row was exact. Nothing of the Hopper rows ran.

## Reading the outcome (decision rule from the theory; thresholds are the theory's, fixed before the run)

Primary number: `tops`, mode `loop` (launch amortised), 4096^3, L = 14, `exact` true, best `f_` row, and its ratio to the `m_` row of the same run.

| Result (f_*, 4096^3, L = 14) | Meaning |
|---|---|
| above 1456 TOPS | Above the theory's range (1456 is its upper edge, near the mainloop-only rate): do not celebrate. Check `exact` (all three parts), `sm_mhz_mean` against `clock_khz_max`, that `m_` is in its usual range (a sick baseline inflates the ratio), the SASS gate, and that the sampled bytes are not accidentally trivial; then repeat the row. A real result this high means the stock ceiling was the epilogue and the mainloop is better than fitted. |
| 1260-1456 | Theory confirmed: the stock gap was the epilogue. The issue-16 rule's `R_stock` was pessimistic; re-run the rate projection with these rates (as a new, separately labelled measurement), and issue 22 proceeds to ladder rungs 2-6 with the best row. |
| 1100-1260 | Partial: better than stock, short of the fit. Look at the `g_` / `f_` difference (epilogue cost), `sm_mhz_min` (clock), `ctas_launched` against `sm_count` (undersubscribed cluster grids) and the sweep before judging the mainloop. |
| 1100 or below | Theory falsified: the epilogue is not the main loss. Next suspects in the theory's order: mainloop slope (cluster sweep, `f_128x256x128_4x1`, K64 stages), then the SASS. A custom (non-CUTLASS) kernel is justified only if the mainloop slope stays 8% or more below cuBLAS (a > 6 ns/K at 128x256) with NDEBUG, cluster, swizzle and our epilogue all in. |
| any `exact` false | Do not use that row's rate. Stop and look: which part failed (`ref_mismatches`, `hash_eq_first`, `exact_after_timing`), where (`first_bad_*`), and against which reference row. Rung 0 passing does not clear it (different binary and size). |

8192^3, L = 14 (stock collapsed to 868 against 1152 at 4096):

| Result | Meaning |
|---|---|
| tops rises with `swizzle` 4-8 (either raster), `sm_mhz_min` stays near max | Locality: the default `max_swizzle_size = 1` was the cause; set the swizzle in the product launch. |
| tops flat over the sweep, `sm_mhz_min` well below `clock_khz_max` / 1000 or `throttle_reasons` non-zero (0x4 / 0x8 / 0x20 / 0x40 / 0x80); `--sustain` rate below the short-block rate | Power or thermal throttling: not a kernel problem; report per card, do not tune. (`idle_app_flags` 0x1 / 0x2 are not throttling.) |
| tops flat over the sweep, clock flat, still lower than 4096 | Neither: something else (L2 footprint of A and B per wave; wave quantisation is small at L = 14 per the theory). Report; the `loop` against `single` difference and L = 4, 1 rows help. |

Cluster sweep (theory item 3): `f_128x256x128` at 1x1, 1x2, 2x1, 4x1. `4x1` is new and untested on any card; a cluster launch error is recorded as `status` "error: ..." and
the run goes on; a hang is caught by the timeout, which is why it runs last and prints `starting ...` first. Compare `queried` against `heuristic` rows for the same cluster shape
(`ctas_launched`). `f_` against `g_` rows: epilogue cost. Ping-pong 128x128 and K64: theory item 5, low value.

Reference batches L = 4 and L = 1: they use `slot0 = 1` (recorded), because `moduli[0] = 256` turns the epilogue into a mask and an L = 1 run at slot 0 would measure only that
(epilogue-trivial) case. L = 1 at 4096 is still partly L2-resident (A + B = 33 MB against a 50 MB L2) and has fewer tiles than SMs for the large tiles (4096^2 / (128x256) = 512
tiles on 132 SMs: wave-quantised): these rows are context, not part of the decision.

## The synthetic peak (`p_` rows)

**Question.** The datasheet says 1979 INT8 TOPS dense (H100 SXM); cuBLASLt reaches 1313-1544 and the best stock wgmma kernels about 1150-1570. How much of the gap
is the card (power, clock, instruction issue) and how much is software? A kernel that does nothing but wgmma, from shared memory, with no global traffic, no
producer and no epilogue gives an **upper bound without memory system, TMA or epilogue** for what wgmma can do on this card; the real rows are read against it (the same role
`mma_peak` plays for `mma.sync`: 1275 TOPS on H100/H200 in `bench/ceiling`). It is a bound, not an achievable rate, and the cold ~1 s numbers are the weakest form of it: the quotable ones are the sustained blocks.

**Design** (`k_wgmma_peak<N, NWG, ST>` in `rung1.cu`; the code comments repeat the reasons).
* Persistent, one CTA per SM (`grid = SM count`), `NWG` consumer warpgroups per CTA (1, 2, 3 for N = 256, 2 and 4 for N = 128; the accumulator is 64 x N int32 =
  N/2 registers per thread, so 3 x 128 threads x 154 registers still fits the 64K register file at N = 256). No producer warp: the operand tiles are copied into
  shared memory once at the start (plain stores through CuTe's swizzled layout, then `fence.proxy.async` and `__syncthreads`; the wgmma reads shared memory
  through the async proxy) and never change.
* Descriptor-based SS wgmma `m64nNk32.s32.s8.s8`, K-major, 128-byte swizzle, built exactly as the stock sm90 TMA mainloop builds them: `tile_to_shape` of
  `GMMA::Layout_K_SW128_Atom<int8_t>` (a 128-wide int8 K tile is one atom wide, the same layout `Variant` static-asserts), `make_tiled_mma(SM90_64xNx32_S32S8S8_SS_TN)`,
  `make_fragment_A/B` (the descriptor tensors), `cute::gemm`, `warpgroup_arrive / commit_batch / wait<1>`, `warpgroup_fence_operand`. No hand-written descriptor and no
  inline wgmma asm in this repo's code. CuTe's `make_gmma_desc` static-asserts the canonical layout at compile time, which it did.
* Each warpgroup has its own A tile (64 x 128, different data per warpgroup); all warpgroups of a CTA share one B tile (N x 128), like the cooperative kernel's
  128 x 256 tile. One loop iteration ("block") = 4 K tiles x 4 k32 instructions = **16 wgmma** in one commit group; then `wait_group 1`: the previous group has
  finished, this one is running while the next one is issued, so the tensor pipes never drain except once at the end. 16 instructions per wait is within the
  8-16 asked for, with one group always in flight.
* The loop count (`iterations_blocks`) is chosen at run time so one launch lasts about 0.25 s (from a 0.3 s warm-up of short launches); the same count is used for
  all timed and checked launches of the row. Accumulators start at zero, `ScaleOut::One` throughout, and are written to global memory once at the end. A `bar.sync` over all
  128 NWG threads precedes the first `clock64()` read, so the warpgroups of a CTA start their loops together.
* Rows (`--list` / `--plan` print them, with their shared memory): `p_n256_wg{1,2,3}`, `p_n256_wg1x2cta` (two 1-warpgroup CTAs per SM, 76 KiB of dynamic shared memory each so that
  2 fit and 3 cannot: occupancy exactly 2, a run-time check), `p_n128_wg{2,4}`; each with data `_zero`, `_small`, `_full`, and `_full4` for `n256_wg2`, `n128_wg4`. One-CTA-per-SM rows
  request 116 KiB of dynamic shared memory (more than half of the 227 KiB) so that no second CTA can land on the same SM; a row whose occupancy is not exactly its CTAs per SM is skipped.
  **Placement** is verified, not assumed: each launch's CTA -> `%smid` map is logged on the device (timed, sustained and final launches; not the short warm-up launches) and `placement_ok`
  is true only if in every logged launch each of the card's SMs holds exactly the row's CTAs (`placement_launches_checked` / `_total`; `sms_used`).
  `p_n256_wg2_*` is the structure of the stock cooperative 128 x 256 kernel (two consumer warpgroups); `wg3` and `n128_wg4` ask whether more resident warpgroups
  issue closer to the limit; `wg1` and `wg1x2cta` ask whether one warpgroup suffices.
* Data: `zero` (A = B = 0: the least toggling; the exactness check is trivial and marked `exact_check_trivial`), `small` (values -3..3: realistic accumulator magnitudes),
  `full` (counter-based hash, -127..127), `full4` (same hash, four different tiles rotating through the stages, so the operands change every 16 wgmma instead of every 4 k-slices;
  4 x the shared memory). Which data draws more power than a real GEMM is **not known**: the operands are static in shared memory (less operand toggling than a GEMM streaming tiles),
  but the `full` accumulators wrap around int32 and toggle all their bits every instruction, which real accumulators with modest magnitudes do not; only `small` has realistic accumulator
  magnitudes. So no data kind is claimed to bound a real kernel's power from either side; the three are reported separately and the sustained rate of each is quoted.
* Not in the loop: no `mbarrier`, no TMA, no global load or store, no `bar.sync`, no epilogue.

**Measurement.** Per row: 0.3 s of short launches (clock ramp), calibration, then `--peak-seconds` (default 1.25 s) of single launches of about 0.25 s, an event pair around
each (at least 5), median / min / max, `spread_pct`; the NVML sampler (the harness's `Sampler`, 5 ms) runs over the timed block. That record is a **cold burst** (`ceiling_kind`
"cold_burst_upper_bound"): do not quote it as a ceiling. With `--sustain S` the peak pass adds, after all `p_` rows, three more blocks of S seconds, on the best exact row **of each data kind**
(`small`, `full`, `full4`) by its timed rate (`only_sustain`: no second timed block): tops over the block and over its first and last second, `throttle_reasons` / `power_w_max` over it,
`ceiling_kind` "sustained". A ceiling quote is always written with the block's `seconds`, `temp_c` (last sample), `sm_mhz_last` / `sm_mhz_mean` and `power_w_last`. If a kind has no exact row, a status record
(`variant` `peak_sustain_<data>`, `status` "skipped: ...") is emitted and counted (rc 5). Before every launch the output buffers are poisoned (outside the event pair).

**Fields** (JSON line per row and mode `peak` / `sustain`; the common ones as for the other rows): `tops` = (2 x 64 x N x 32 ops per wgmma) x (wgmma per warpgroup =
16 x `iterations_blocks`) x warpgroups x CTAs / launch time, int8 TOPS over all SMs and warpgroups (the convention of `bench/ceiling`: 2 ops per MAC, median);
`ops_clk_sm_clock64` = ops per SM clock per SM: for each SM, ALL the ops of its warpgroups divided by (latest `clock64` end - earliest `clock64` start of those warpgroups) (absolute
counter values, one window per SM, no sampling), mean over the SMs (`_min` the slowest SM; `frac_of_datasheet_ops_clk` against 8192 = 4096 MAC). It is **only valid when
`cycles_max_over_min` (the slowest over the fastest warpgroup loop, in clocks) is about 1**: if warpgroups finish unevenly the common window is longer than the busiest warpgroup's loop and the figure
understates the instantaneous rate, and a per-warpgroup sum would have overstated it by up to +33-50%, which is why it is not used; `cycles_min/mean/max` are in every record. `sm_mhz_clock64` = cycles / launch time;
from NVML (null without it): `sm_mhz_min/mean/last`, `tops_per_ghz_nvml` (TOPS per GHz, the clock-normalised ceiling), `ops_clk_sm_nvml`, `ceiling_tops_at_max_clock` and
`ceiling_tops_at_nvml_clock` (SMs x 8192 x clock: what the instruction rate would give at that clock), `datasheet_tops` = 1979 (with `datasheet_ref_clock_ghz` 1.83, `datasheet_ref_sms` 132),
`power_w_max` / `power_w_last`, `temp_c`, `throttle_reasons`; `regs`, `local_bytes`, `max_blocks_per_sm`, `smem_dynamic_bytes`, `iterations_blocks`, `wgmma_per_wg`, `grid`, `sms_used`, `placement_ok`.

**Exactness.** After R = `iterations_blocks` blocks every accumulator must equal R x S mod 2^32, where S[row][col] is the host-computed sum, over the four tiles of a
block (stage j mod ST), of the int32 dot product A_tile . B_tile^T (so with ST = 1 it is 4 x the one product). The wgmma `.s32` form without `.satfinite` wraps,
integer addition is associative and the host multiplies in uint64, so the wrapped result is deterministic and exact; for `full` data R x S overflows int32 many times and the
check still has to match bit for bit. EVERY element of EVERY CTA and warpgroup is compared (up to 26 million int32), against the accumulator position of the PTX D fragment
(row = 16 warp + lane/4 + 8 ((v/2) mod 2), col = 8 (v/4) + 2 (lane mod 4) + v mod 2, the order `layout_ok` in `k4_hopper.cuh` checks against CuTe at compile time); the loop counter of every warpgroup must
equal R. It is checked on the last launch of the timed block, of the sustained block, and on a fresh launch after both (`ref_mismatches`, `ref_mismatches_after`,
`iters_mismatches`); `exact` is all of them, and `exact: false` means do not use the row's rate. A failing row records the first bad (cta, wg, thread, register, want, got).
The `zero` rows can only see that results were written and the loop ran R times: they are **trivially exact**, so a `zero` figure is only meaningful next to the exact `full` row of the same configuration.

**SASS evidence** (cross-compile here, nvcc 13.4, CUTLASS 4.8.0, sm_90a, `-O2 -DNDEBUG`, `tools/build_rung1.sh`; it prints this table from `cuobjdump -sass` and warns on a `CALL` in a
peak kernel; re-run after the review changes, same result): no ptxas C7510, no spills and no stack frame in any of the seven peak kernels (the first version had an 8-byte spill in `n256_wg3` with four stages at the 168-register cap
of 384 threads; that instantiation, whose row added little, was dropped), registers 154 (`n256` 1 stage, any warpgroup count), 172 (`n256_wg2` 4 stages), 90 / 108 (`n128`). SASS per kernel:

```
ok  IGMMA 16  WARPGROUP.DEPBAR 2  CALL 0  peak wgmma-n256 wg1 stages1     (also wg2, wg3; n128 wg2, wg4; stages4 for n256 wg2 and n128 wg4)
```

i.e. one loop of exactly 16 `IGMMA.64x256x32.S8.S8` on the same accumulator registers (R24..R151) with one `WARPGROUP.DEPBAR.LE gsb0, 0x1` and the back branch, and
one `DEPBAR.LE gsb0, 0x0` after the loop; before the first IGMMA of a block there are about twenty uniform-datapath instructions (descriptor arithmetic, `R2UR`, `UMOV`), which run while the
previous group executes; no CALL, no `BAR`, no memory instruction in the loop.

**Reading the result.** `ops_clk_sm_clock64` is the instruction rate per SM clock, 8192 being the architecture's dense INT8 rate (4096 MAC); a power cap lowers the clock (`sm_mhz_*`), not the ops per clock.
TOPS = SMs x ops/clk/SM x clock. First check each row: `exact` true, `placement_ok` true, `cycles_max_over_min` about 1 (else the per-clock figure is not valid; use `tops` and the clock). Then, per configuration:
* **>= ~0.95 x 8192** (`frac_of_datasheet_ops_clk` >= 0.95) on `small` (or `zero`) **and the same configuration's `full` row exact** (the zero rows are trivially exact and prove nothing alone):
  wgmma from shared memory issues at the datasheet rate. Whether the TOPS gets to 1979 then depends only on the clock.
* **0.85 - 0.95: partial.** Report the rate; check `cycles_max` / `cycles_min` (uneven warpgroups make the window longer, so a ratio above 1.05 may explain the shortfall), `placement_ok`, and whether `wg2`/`wg3`/`n128_wg4`
  differ; do not call either "reached" or "not reached".
* **< ~0.85 in every configuration**, clock at maximum, no throttle bit: the instruction does not issue at the datasheet rate in this form (or `clock64` does not tick at the SM clock; compare `sm_mhz_clock64`
  with NVML's). The real kernels' gap to 1979 is then not theirs to close; read them against the best sustained `p_` TOPS.
* **Power-limited:** a data kind has the same ops/clk as `small` but its sustained clock (`sm_mhz_mean`, `sm_mhz_last`, `tops_last_1s`) is lower, `power_w_max` / `power_w_last` is near the card's limit and `throttle_reasons` has 0x4.
  The sustained TOPS of that data kind is then the clock-limited bound for that data; quote it with its `seconds`, `temp_c` and clock. Which kind is closest to a real GEMM's power is not known (see Data).
* **Warpgroups:** `wg1` well below `wg2`: one warpgroup cannot keep the pipes full (the real kernel's two consumers are needed); `wg3` or `n128_wg4` above `wg2`: more resident warpgroups would help a real kernel.
  **`n128` below `n256`** has two candidate explanations that this data cannot separate: issue slots (an n128 instruction does half the work of an n256 one per issue) and shared-memory traffic per instruction:
  n256 reads A 2 KiB + B 8 KiB = 10 KiB per 128 clocks at the datasheet rate = 80 B/clk, n128 reads 2 + 4 = 6 KiB per 64 clocks = 96 B/clk, against about 128 B/clk/SM of shared-memory bandwidth, so n128 sits closer to
  that limit. (Two warpgroups reading the same B tile do not share the read.)
* The headline for the decision rule above: `ceiling` = the highest exact **sustained** `tops_last_1s` among the data kinds, quoted with data kind, `seconds`, `temp_c` and clocks, and `real / ceiling` for the best `f_` row
  (and for cuBLASLt's 1313-1544 from `bench/ceiling`, a different run, same card). The cold-burst `peak` records are upper bounds without memory system, TMA or epilogue and are never to be called achievable. The 1456 TOPS
  upper edge of the theory and the 1260 lower edge should be compared with this ceiling before they are interpreted: if the ceiling itself is near 1500 (power-limited), 1456 is not "above the mainloop", it is at the card's limit.

## Not verified (no Hopper here)

* The synthetic peak, all of it: it has been cross-compiled and its SASS read, nothing more. That the CuTe-built descriptors address the swizzled tiles my shared-memory fill wrote
  (the fill and the descriptors both come from the same `tile_to_shape` layout, which is what the TMA-fed stock mainloop relies on; the 1024-byte base alignment is done by hand and the
  `base_offset` field is 0), so an `exact: false` on the very first `p_` row would be this or the host's fragment-position formula, not a speed result; that `wait_group 1` (one group of 16 in flight)
  is deep enough to saturate the pipes (a deeper variant is a one-line change); that one launch's `clock64` window is the whole loop; the occupancy / placement of the one-CTA-per-SM rows; the
  runtime of a row (about 2 s of GPU time plus about a second of host checking per row is an estimate) and that the 26 MB accumulator write-out and the host comparison do not disturb the following row's clock;
  that NVML's power reading lags (average over about a second): the `sustain` record is the one to trust for power.
* Which data kind is closest to a real GEMM's power is unknown (static operands vs wrapping accumulators, see Data); `full4` changes operands more often than `full`.
* The new checks of the review (the placement log, the common `clock64` window with its `bar.sync`, the per-kind sustain items, the two-process run script) are cross-compiled; the script's run path was exercised on the 4090 only up to the exit-3 of both passes and the status file.
* Everything dynamic on the Hopper rows: that the harness runs there, the numbers, the exactness of any variant on the card, that the 2x1 and 4x1 128x256 clusters launch, that
  the queried `max_active_clusters` gives a sensible grid, that NVML works in the allocation (it is optional), that the 0.3 s load and 12 repetitions give a stable clock
  (check `spread_pct` and `sm_mhz_min`), and the SASS gate on the cluster's own nvcc (the build script prints it).
* The sampled reference reads about 390 bytes per row; the hash check is relative to the first row run.
* `b = 1`, `batch0 = 0`: the modulus mapping of the product's split launches is rung 0's business. `m_` is one launch over the whole K, not the library's chunked groups.
* Operands are random int8, not residues of a real matrix (tensor-core rates do not depend on the values; power does, a little).
* Only store mode (no add mode) and only K = m (a single chunk, no chunking at 131008).
* No L2 flush between launches (the loop mode runs them back to back, as K4 does).

## Library integration (design)

Rung 1 timed the Hopper kernels as a harness. This section is the design of the same kernel as the library's K4 on sm_90, written before the integration (step 1) and then kept as the reference for it (step 2 landed with it, see "What is implemented"). Goal: H100 / H200 run K4 on wgmma + TMA where the evidence says it wins, every other card is bit-for-bit and SASS-for-SASS what it was, and the mma.sync K4 stays the fallback of every launch the new kernel does not serve.

### What the kernel is and what it replaces

K4 is the batched INT8 GEMM of the emulation with the mod-p epilogue: for a plan (m, n-hat, k-hat, b) and the moduli of one g-group, `k4_for_each_launch` (host/k4_launches.hpp) cuts the work into launches by K chunk (K_c' = 131008: the first chunk stores W mod p, later chunks add mod p), by batch part (<= 65535) and by n-hat part (gridDim.y). A launch is the GEMM `W[l][z] = (A[z] * B[z]^T) mod p_l` with `l = slot0 + (batch0 + z) / b`. Today it is `K4Ops::launch` (src/k4/k4_cfg.cuh, one `K4Gemm` instantiation per configuration, parameters in a `K4Blob` per launch shape). The Hopper backend is a second implementation of that one launch: CUTLASS `Variant<Tile, Cluster, Sched, kAdd, kFast>` of src/k4/hopper/k4_hopper.cuh (TMA loads, two consumer warpgroups, `K4ModEpilogue` through `HopperEpilogue`'s quad-transposed 16-byte stores, no shared memory in the epilogue). Same launch structure, same launch count, same plan host bytes; a different kernel per launch.

### Backend selection (K4Ops / device class)

* **Backend** = `K4Backend {mma = 0, wgmma = 1, wgmma forced = 2}` (src/host/k4_wgmma.hpp, CUDA-free so the CPU tests include it). A plan stores the backend it tries first (`K4Plan::backend`, in the existing padding: `sizeof(K4Plan)` and the plan host bytes do not change) and keeps the mma.sync blobs it always had.
* **Device class.** One row per K4 table set of the registry (`kK4WgmmaTable[IMD_TABLE_*]`): `IMD_TABLE_SM90` (H100 and H200 share it: R2 shows the same kernel best on both) is true, `IMD_TABLE_SM89` (RTX 4090, and sm_120 which the registry maps onto it) is false. `k4_wgmma_device_ok` adds the compute capability 9.0 test, so a generation match that lands on the sm_90 row from another capability never runs the backend. Blackwell data centre parts (sm_100) are not in scope: no table row, mma.sync.
* **Availability** = class row AND the library has the backend (`IMMERDINK_K4_WGMMA`, below) AND a per-device probe at plan creation (the setup of both kernels succeeds, `cudaFuncGetAttributes.binaryVersion == 90` so the sm_90a image is what the device runs, and `cudaOccupancyMaxActiveClusters` gives 1..255 clusters). The probe is cached per device, but the per-device function attributes (dynamic shared memory) are set again on every plan creation, as the mma.sync setup does, and the probe never clears an error the caller left pending. A device without the backend never makes a CUDA call for it.
* **Override** for A/B and tests: process-wide, plan-time (`imd_test_set_k4_backend` / `imd_prof_set_k4_backend`; the library reads no environment variables, the harness maps `IMD_TEST_K4=mma|wgmma|auto` to the hook and reads it back). The override is a REQUEST read at plan creation only. `wgmma` makes the plan **backend 2, "wgmma forced"**: every LEGAL launch of that plan goes to the Hopper kernel, the profitability floor lifted for the plan's life (a property of the plan, stored in the existing byte `K4Plan::backend`; legality is never lifted). The launches never read the process-wide override, so resetting it after `imd_plan_create` (as `imd_bench` does) leaves the plan forced, and the floor sweep times wgmma and not mma.sync twice. Plan backends: 0 mma.sync, 1 wgmma with the floor (the device class default), 2 wgmma forced; the hooks and records report these values. Forcing wgmma on an unavailable device gives an mma.sync plan (the plan reports its backend: `imd_test_plan_k4_backend`). `k4_backend_resolve(override, available, class_default)` is the whole rule and is tested exhaustively.
* **Build.** CMake option `IMMERDINK_K4_WGMMA` = AUTO (default) | ON | OFF. AUTO is ON only if the architectures include sm_90 (matched by `^90`, or `all` / `all-major`; **not** `native`, whose GPU is not known at configure time) AND CUTLASS >= 3 is found AND nvcc >= 12.0; otherwise OFF, and then the library is built exactly as before (nothing of the backend is compiled, the hooks report "not built"). wgmma needs sm_90a, while the cluster builds `CMAKE_CUDA_ARCHITECTURES=90`, so the Hopper translation unit is its own OBJECT library compiled for `90a-real` and linked into libimmerdink **A build meant for a Hopper configures with `-DIMMERDINK_K4_WGMMA=ON`** (ON fails at configure time if the toolchain cannot do it, AUTO would silently come out OFF), and `tests/test_k4_wgmma.py` FAILS on a cc 9.0 device whose library has no backend (unless `IMD_ALLOW_NO_WGMMA=1`), so a silently-OFF Hopper build cannot pass the suite. (an sm_90a-only object in a shared library loads and runs on sm_89 including `CUDA_MODULE_LOADING=EAGER`; only a launch of its kernel fails, and the host gate makes that unreachable; checked on the 4090).

### Tile, cluster and scheduler choice (from R2)

Source: `rung1.jsonl` of 14421919 (H100) and 14421920 (H200), exact rows, store mode, L = 14 unless stated. The choice is a pure function of the launch (`k4_wgmma_pick`) with the scheduler arguments set explicitly; CUTLASS's defaults are not relied on, because at 8192^3 they cost the 128x256 kernels about a quarter of their rate.

| shape (m = n = k, batch 14) | mma.sync K4 (m_ row) | chosen: 128x256x128, 2x1 cluster, cooperative | swizzle / raster | notes |
| --- | --- | --- | --- | --- |
| 4096 | 922 H100 / 894 H200 | 1360 / 1357-1365 | 1, heuristic (CUTLASS default) | best on both; spread 5 / 8 % |
| 8192 | about 880-895 | 1408-1414 / 1328-1332 | 8, AlongN | spread 6-9 %; the default scheduler (swizzle 1) gave 1035-1043 on H100 (1x1) |

(TOPS-eq of the whole emulated GEMM, ops = 2 m n k L.) The rule: tiles per batch (`ceil(m/128) * ceil(nc/256)`) >= 1024 (8192^2 has 2048, 4096^2 has 512) -> swizzle 8 AlongN, else swizzle 1 heuristic. One variant, on purpose: the single best row at 8192^3 is 1x1 swizzle 8 AlongN (1427 H100, 1383 H200) but its H200 rows have a 26.5 % spread and it is only 1-2 % above the 2x1 rows on H100; the 2x1 cluster is the row that is good and stable on both cards at both sizes. The variant list is an array (`kK4WgmmaVariants`) so a later per-shape / per-card variant is a table row, not a redesign (memory: modular per hardware).

Expected benefit. At 4096^3, L = 14 the K4 is 2 * 4096^3 * 14 = 1.92e12 integer ops. Today's mma.sync K4 on H100 takes 1.994 ms (the `m_` row). At the measured 1360 TOPS the wgmma K4 takes 1.41 ms (1.41x of today's K4); at the 1830 TOPS sustained synthetic ceiling of the same card it would take 1.05 ms (1.9x). The end-to-end gain is bounded by K4's share of the call (K0 / KF, K5 and the packs are unchanged) and is a cluster measurement (R3).

### Small and odd shapes (no rung-1 data below 4096)

Rung 1 timed m = n = k in {4096, 8192} only. Nothing below is measured, so the default path serves only the region of the evidence (the profitability floor, below) and falls back to mma.sync everywhere else; a forced run exercises everything legal so that exactness is tested on the odd shapes too.

* **Floor (default only).** Serve a launch if m >= 1024, n-part >= 1024, K chunk >= 1024 AND tiles x batch >= 2 x SMs (two waves). The numbers are a placeholder just inside the measured region; the R3 shape sweep replaces them (see R3 below).
* **k not a multiple of 128 (or of the 128-wide k tile).** The operand rows are `k_hat` long (a multiple of 16, of the plan's alignment A = 16 / 64 in fact) and the launch's K chunk `kc <= k_hat`. The TMA tensor extent of the launch is `kc`; the last k tile is zero-filled out of bounds (`oobFill 0` = zeros, which add nothing to the INT32 accumulator), so any `kc` that is a multiple of 16 is exact. `kc` itself is a multiple of 16 for every chunk (K_c' = 131008 = 16 * 8188; the test limits are multiples of 16).
* **m / n not multiples of the tile (128 x 256).** Tensor extents are m and n-part, the partial tiles read zeros out of bounds, and the epilogue masks its stores by the same extents (K4ModEpilogue / HopperEpilogue are bounds-aware: rows < m, columns < nc; tested by `test_hopper_epilogue`).
* **n-hat part not a multiple of 16** or n-hat not a multiple of 16: not served (the 128-bit stores need it; the plan already guarantees n-hat % 16 == 0 for every plan).
* **Tiny shapes (m = 1, kc = 16)** are legal and exact when forced (one k tile of 16, rest zeros), never served by default.

### TMA alignment and every legality rule

`k4_wgmma_illegal` is the single place; a launch that fails it is served by mma.sync and counted as declined. CUTLASS's tensor-map encoder prints on failure and its `assert(false)` is a no-op under NDEBUG, so an illegal launch must never reach it:

* global base address of A, B 16-byte aligned (the plan's alignment contract A = 16 / 64 makes `k0` offsets, row strides and the workspace plane offsets multiples of 16; checked per launch anyway, the pointers are caller / workspace pointers), W base and plane stride multiples of 16;
* strides multiples of 16 bytes and < 2^40 (k_hat, batch strides of A and B), dims <= 2^32, m / nc / kc <= INT32_MAX, batch part <= 65535 (the existing split), tile count < 2^31;
* the operand batch planes must not overlap (`batch_a >= m * k_hat`, `batch_b >= n_hat * k_hat`): an operand whose batch stride is smaller than the plane it is read with (a packed operand viewed through shorter blocks) is a legal strided view for the mma.sync K4 but not one I can show every driver's encoder accepts, so mma.sync serves it;
* `kc <= k_hat` and `kc`, `nc`, `k_hat`, `n_hat` multiples of 16, W plane >= m * n-hat.

TMA descriptors are encoded on the host per launch (a few microseconds) and passed by value as kernel parameters, so a launch is capture-safe exactly like the mma.sync one (no host callback, no allocation); the persistent-grid cluster count is queried once at plan creation (`hw_info.max_active_clusters`).

### Add-mod, K-splits and group launches

The same `K4Launch` drives both backends, so the backend sees exactly what the mma.sync K4 sees: store-mod for `k0 == 0`, add-mod for later chunks (`kAdd` template parameter of the epilogue: `W = (W + acc) mod p`), the modulus `l = slot0 + (batch0 + z) / b` computed in the epilogue from `K4Args{W, ldw, plane_w, slot0, b, batch0}`, `call_of`'s pointers (A + z0 * plane_a + k0, B + z0 * plane_b + j0 * k_hat + k0, W + (g0 * b + z0) * plane_w + j0), batch parts and n-hat parts as separate launches. What the design needs of `K4Call` / `K4Args` is therefore only what they already hold; no new field in the plan. Per-launch fallback means a mixed run (some launches wgmma, some mma.sync) is legal: both write the same W planes bit for bit, and a later add-mod launch of the other backend reads them.

### Fallback rules (decided on the host, per launch, never silently wrong)

1. Plan backend mma -> unchanged code path, no Hopper code reached.
2. Plan backend wgmma (1 or 2): for each launch, build the `K4WgmmaLaunch` (shape, strides, the three base addresses, SMs); `k4_wgmma_pick` -> illegal (a reason) or unprofitable (backend 1 only: backend 2 lifts the floor) -> mma.sync blob of the plan; counted `declined`. Otherwise the backend's launch: CUTLASS `can_implement` is its first step, and a refusal (nothing encoded or queued) is also run on mma.sync and counted `declined`; only after `cudaLaunchKernelExC` succeeded is it counted `launched`.
3. If the Hopper launch itself errors, the call fails with the library's CUDA_ERROR status (no silent retry: a failed launch may already have queued earlier launches of the group).
4. There is no run-time (device-side) selection and no data-dependent choice.

The counters are process-wide atomics read through the hooks; tests and benchmark records show them (`k4_wgmma_served` / `_declined`, `k4b_run`), so a forced wgmma run that served nothing cannot pass for a wgmma run. **A packed B** (`batch_b = n * k_hat` with `n < n_hat`, the packed operand viewed through shorter blocks) is always declined as overlapping planes: correct, served by mma.sync, a performance matter only (relevant to ehrenfest's packed path; lifting it needs a TMA-legal view of such an operand, a later item).

### Why it is bit-identical

K4 is exact integer arithmetic: for each output, the INT32 accumulation of int8 products over at most K_c' = 131008 terms (the section 3.5 bound; every partial sum fits), and the reduction mod p_l is a function of that integer. The order in which products are summed cannot change an integer sum, so mma.sync m16n8k32 and wgmma m64n256k32 accumulate to the same INT32 for the same operands; `K4ModEpilogue` is one code (the same `mod` device function, `test_cutlass_mod_epilogue`, `mod_exhaustive_cpu`) in both backends; add-mod adds the same stored residues; zero fill adds zeros. What changes is only which instructions run. Evidence on this machine is the CPU tests and the 4090's forced-fallback runs; evidence of the Hopper path itself is Hopper-only (below).

### Test plan

On the RTX 4090 (all offline, pass on every device; none may skip because the campaign harness counts a skipped test as not passed):

* `k4_wgmma_select` (CPU, CUDA-free): class table, the registry rows through `imd_device_select`, the resolve matrix, the R2 table (4096 -> swizzle 1; 8192 -> swizzle 8 AlongN), the 1024-tile threshold, floor vs force, edge shapes (m = 1, kc = 16; 129 x 272 x 1040; 4097 x 4112 x 4112; K_c' = 131008 odd chunks), every illegal reason, a 200000-launch random property sweep (served launches are legal, tile arithmetic never overflows, force only lifts the floor).
* `k4_wgmma_exact` (`tests/test_k4_wgmma.py`): status of the backend read back (unavailable off sm_90, plan reports mma.sync when forced, counters stay 0), 15 shapes with C hashes compared between forced mma.sync and forced wgmma (tails in m, n, k; k not a multiple of 128; k shorter than a tile; m = 1; n = 5; N = 2 and 20; alpha / beta; batch; K-chunk, batch and n-hat splits alone and together via `imd_test_set_limits`; and `swizzle8_tails`, 4100 x 8200 x 256, N = 14, about 0.5 GB of W: 33 x 33 = 1089 tiles per batch, the swizzle-8 / AlongN scheduler path with tails; skipped with a printed note below 3 GiB of free device memory; on a Hopper the served-launches check makes it non-vacuous, and the dry-run table pins that its pick is swizzle 8 / AlongN), a plan created under a forced request stays backend 2 after the request is reset, and 30 host dry runs of the Hopper launch (`can_implement`, scheduler choice). **On a Hopper the same test is the exactness test of the real path**: the forced runs must serve launches (counter > 0, none declined) and equal the mma.sync C. A cc 9.0 device with a library built without the backend fails it (`IMD_ALLOW_NO_WGMMA=1` opts out).
* the existing oracle suites re-run with the backend forced (`contract_{e2e,every_n,bigk}_k4wgmma`), the 19 committed hashes with it forced (`output_hashes_k4wgmma`, `output_hashes_large_k4wgmma`) and with mma.sync forced (`output_hashes_k4mma`, the fallback on a Hopper).
* SASS: `tools/check_k4_wgmma_sass.sh libimmerdink.so [build.log]`: IGMMA present, `WARPGROUP.DEPBAR` fewer than IGMMA (the C7510 serialisation signature), no local memory in the sm_90a kernels, no C7510 in the logs; SASS of the sm_89 image identical to the baseline with the option ON and OFF.
* Cluster (R3, below): the same tests on H100 and H200, the duel `k4b=mma` vs `k4b=wgmma` of imd_bench.

A/B duel key: `imd_bench --methods duel --duel-a k4b=mma --duel-b k4b=wgmma` (side keys `k4b=default|mma|wgmma`, and `--k4b` for method imd). The side's plan keeps its backend (a plan-time choice), so the two plans run interleaved with no per-call switching; the records carry `k4b_req`, `k4b_run` (0 / 1 / 2 as above), `k4_wgmma_served` / `_declined` (per duel side, over its timed calls; for method imd over the method's calls) and `c_hash` / `bit_identical`.

### What is implemented (step 2)

* `src/host/k4_wgmma.hpp` (CUDA-free): backends, class table, resolve, legality, profitability, tile / scheduler pick.
* `src/k4/hopper/k4_wgmma_{ops.hpp,impl.cuh,v0.cu}`: the backend (one variant today), its own OBJECT library compiled for `90a-real` when the option is ON.
* `src/gemm_k4.cu`: per-launch dispatch with the mma.sync fallback, probe, counters; `src/api.cpp`: backend chosen at plan creation; `src/test_hooks.cpp`, `src/profile_hooks.h`: hooks (`imd_test_*` / `imd_prof_*`); `bench/imd_bench.cu`: `k4b`; `CMakeLists.txt`: the option; tests and `tools/check_k4_wgmma_sass.sh` as above.
* New ctest tests: 8 (`k4_wgmma_select`, `k4_wgmma_exact`, `contract_{e2e,every_n,bigk}_k4wgmma`, `output_hashes_k4wgmma`, `output_hashes_large_k4wgmma`, `output_hashes_k4mma`): `ctest -N` of the standard configuration goes 98 -> **106** (the manifest's `--expect-tests 98` must become 106).

### Checks and what they cannot see

Checks done on the RTX 4090 are listed with results in the hand-off report of this change; what they cannot see: that the Hopper kernel runs and is exact on a Hopper (nothing here executes sm_90a), that the `K4Args` modulus mapping for batch parts, add-mod and slot offsets is right on the hardware (the cutlass-level epilogue tests cover the host half and the layout), that TMA accepts every descriptor the legality rule admits, the occupancy / cluster count on the real parts, and every timing.

### Open questions (with a recommendation each)

1. **Default on, or opt-in, on H100 / H200?** Recommend default on above the floor, after R3 passes the hash gate and the exactness test on both cards; until then the manifest runs `k4b=wgmma` explicitly. The override stays for A/B.
2. **Floor values.** The 1024 / two-wave floor is a placeholder; replace with the crossover measured in the R3 shape sweep (m = n = k from 512 to 4096, batch 1 / 4 / 14, and varied m:n:k, wgmma vs mma.sync). Recommend a floor on estimated time, not on size, once the data exists.
3. **One variant or several.** Recommend one (2x1 cooperative 128 x 256 x 128) until R3 shows a shape where another is > 5 % better on both cards; then add a table row (1x1 swizzle 8 at 8192 is the candidate: +1-2 % on H100, noisy on H200).
4. **Add-mod and K-split on Hopper are not timed.** Rung 1 is store mode, one chunk. Recommend R3 times a K > 131008 shape; if add-mod is slower than mma.sync's, serve only chunk 0 by wgmma (the per-launch rule allows it: a floor row).
5. **TMA descriptors per launch.** Encoded on the host every launch (microseconds). Recommend leave it; cache per plan only if R3 shows the planner-call overhead matters for small launches (they are below the floor anyway).
6. **Persistent grid under concurrency.** The grid is sized by the cluster count the device could run alone; if ehrenfest runs other kernels on the same stream order nothing changes, on other streams the kernel is slower but correct. Recommend no change.
7. **sm_100 / sm_120.** Out of scope (tcgen05 is a different kernel). sm_120 keeps mma.sync (its table row maps to sm_89's).
8. **Toolchain.** The backend needs nvcc >= 12.0 for sm_90a and CUTLASS 3+; the cluster uses 13.x. Recommend keep AUTO as is.

### Risks

* Everything dynamic is unverified here (above); the first Hopper run of `k4_wgmma_exact` is the real test.
* Rung-1 prototype assumptions that are NOT safe to integrate as they were: (a) `b = 1`, `batch0 = 0`, one K chunk, store mode only -> the library's modulus mapping, add-mod and chunking are new code paths on the hardware; (b) the harness's `Problem` used dense, 16-byte aligned operands it allocated itself -> the library's pointers come from the caller and the workspace, so alignment, strides and overlap are checked per launch (legality, above), never assumed; (c) m, n, k multiples of 128 / 256 -> tails are handled by TMA zero fill and a bounds-aware epilogue, but the tails were not run on a Hopper; (d) CUTLASS failing to encode a descriptor only prints (`assert` is compiled out): the reason legality is decided on the host before any CUTLASS call; (e) the scheduler default at 8192^3 (-27 % on H100): the choice sets swizzle / raster explicitly.
* The per-launch counters are process-wide atomics (relaxed): negligible cost, but shared by all plans and threads.
* A library built with the backend loads and runs on sm_89 / sm_120 (checked on sm_89); a launch of the sm_90a kernel there would fail, which the host gate makes unreachable (class row AND capability 9.0 AND probe).

### R3 cluster block: proposal

1. Configure with `-DIMMERDINK_K4_WGMMA=ON` (explicit: AUTO may come out OFF), arch 90. **Hard prerequisite, before any test:** `tools/check_k4_wgmma_sass.sh` on the library and the build log passes (a library that fails it, or has no sm_90a image, stops the block), then `ctest -N` = 106.
2. Run the 8 new tests plus the full suite: all must PASS on H100 and H200; `k4_wgmma_exact` must report served launches > 0 and no declined launch (it prints PASS only then), including `swizzle8_tails` (not skipped for memory: the cluster GPUs have it).
3. `output_hashes_k4wgmma`, `output_hashes_large_k4wgmma`, `output_hashes_k4mma`: the 19 committed hashes under both forced backends. Also an odd K chunk checked by hash, not only timed: a shape with K > 131008 (the second chunk starts at `k0 = 131008` = 64 mod 128, an add-mod launch whose TMA box starts mid-tile of the 128-wide k tile) run under forced mma.sync and forced wgmma with equal `c_hash`.
4. **`c_hash` equality of every k4b duel pair is an explicit pass/fail gate of R3** (`bit_identical` true and equal `c_hash` between the sides): it is the only check of the DEFAULT (non-forced, floor-applying) path on a Hopper, which has no ctest of its own (a duel `k4b=default` vs `k4b=mma` covers it). imd_bench duels (`--duel-a k4b=mma --duel-b k4b=wgmma`, `--keep-samples 1`, N = 14, 8, 20) at 4096^3, 8192^3 batch 1 / 4 / 14 and the shape sweep for the floor (512 .. 4096, m:n:k varied, one K > 131008 shape), recording TOPS-eq first and time alongside, `c_hash` equality, `k4b_run` and served / declined PER SIDE (a side with `k4b_run` 2 and served 0 is not a wgmma side). The floor sweep is meaningful only because force is a plan property: before that, `imd_bench` reset the override right after plan creation and the timed calls applied the floor, i.e. mma.sync against mma.sync.
5. imd_bench method imd of the default (no override) path on both cards, against cuTENSOR and the emulation baselines, to see the end-to-end effect.
