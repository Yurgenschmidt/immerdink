# Issue 22: the Hopper K4 ladder, rung 1 (timing)

Status: harness written and cross-compiled 2026-10-25, revised the same day after an independent review (APPROVE-WITH-CHANGES, 12 points, all
applied below); **never run on a Hopper card** (no Hopper on the dev box). Code: `src/k4/hopper/rung1.cu`, `tools/build_rung1.sh`. Rung 0
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
names its row): (1) the `m_` baseline and every non-"last" `f_` row, default scheduler, first shape then the next, L = 14; (2) the scheduler sweep for the
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
short block under-reports `power_w_max`; for a power-limited rate use `--sustain 4` (record `mode: "sustain"`: `tops` over the block and over its first and last
second, plus the clock and power over the whole block).

**Exactness** (`exact` = all three): `ref_mismatches == 0`: about 390 sampled output bytes (random positions plus corners and tile edges) against a host reference
from the definition (int64 sum over k, `((s mod p) + p) mod p`, p = `moduli[slot0 + z]`); `hash_eq_first`: a position-dependent hash of all of W equals the hash of the
first row run on the same (shape, batch, slot0), named in `hash_ref_variant`; `exact_after_timing`: the hash and the sampled reference recomputed after the timed
loops (about a hundred launches later) still equal the first hash (a race that only bites under sustained load). W is poisoned with 0xEE before the checked launch.
A failing row records the first sampled mismatch (`first_bad_z/i/j`, `first_bad_want`, `first_bad_got`). The hash comparison is relative to the first row: a fault
shared with it is seen only by the sampled reference. Whether the `m_` row is a good reference is checked by the sampled definition too (it passed the same
host reference on the 4090 in a smoke test, see below).

Exit status: 0 clean; 1 some row not exact; 2 usage / setup (unknown `--keys` are an error); 3 no CUDA device or no table row for this device (the 4090 dev box: harmless,
message on stderr); 4 sticky device error after its record; 5 every row exact but some record has status error / skipped (the count is printed); 124 / 137 from the
build script's `timeout -k 30` (a hang; the lines printed so far are in the file).

## Commands

Build and run (the cluster takes the GPU lock itself). Run rung 0 first, but note that **rung 0 does not certify rung 1**: they are different binaries and different
kernel instances (rung 0 on tiny problems, rung 1 at 4096 / 8192 with 14 batches); both are now built `-O2 -DNDEBUG`, so at least the machine code of a given variant
is the same in both. A rung-1 `exact: false` is a finding in its own right, whatever rung 0 said.

```
export IMMERDINK_CUTLASS_DIR=<cutlass checkout>          # default third_party/cutlass
tools/build_rung0.sh $OUT run                             # exit 0 first (now -DNDEBUG too)
BENCH_COMMIT=$(git rev-parse --short HEAD) RUNG1_TIMEOUT=1800 RUNG1_ARGS="--sustain 4 --clusters both" tools/build_rung1.sh $OUT run
#   $OUT/imd_rung1, build.log, sass_counts.txt, nvcc_version.txt, rung1.jsonl (stdout), rung1.err (progress)
```

Quick (about a minute): `RUNG1_ARGS="--quick" tools/build_rung1.sh $OUT run`. Other knobs: `--variants` (groups all / f / g / m / quick / sweep as exact words, else a name or
substring: `--variants g` is the generic rows only), `--shapes 4096,8192` (or MxNxK), `--batches 14,4,1`, `--sweep 8192|4096,8192|none` (uses the selected rows flagged
sweep, or `--sweep-variants`), `--reps`, `--max-seconds`, `--samples`, `--skip-last`, `--list`, `--plan`.
The default plan is 122 tasks (`--plan`); with `--clusters both` and `--sustain` more (unmeasured guess: tens of minutes; raise `RUNG1_TIMEOUT` for `both`). The build script checks CUTLASS
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

## Not verified (no Hopper here)

* Everything dynamic on the Hopper rows: that the harness runs there, the numbers, the exactness of any variant on the card, that the 2x1 and 4x1 128x256 clusters launch, that
  the queried `max_active_clusters` gives a sensible grid, that NVML works in the allocation (it is optional), that the 0.3 s load and 12 repetitions give a stable clock
  (check `spread_pct` and `sm_mhz_min`), and the SASS gate on the cluster's own nvcc (the build script prints it).
* The sampled reference reads about 390 bytes per row; the hash check is relative to the first row run.
* `b = 1`, `batch0 = 0`: the modulus mapping of the product's split launches is rung 0's business. `m_` is one launch over the whole K, not the library's chunked groups.
* Operands are random int8, not residues of a real matrix (tensor-core rates do not depend on the values; power does, a little).
* Only store mode (no add mode) and only K = m (a single chunk, no chunking at 131008).
* No L2 flush between launches (the loop mode runs them back to back, as K4 does).
