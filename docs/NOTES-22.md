# Issue 22: the Hopper K4 ladder, rung 1 (timing)

Status: harness written and cross-compiled 2026-10-25; **never run on a Hopper card** (no Hopper on the dev box). Code: `src/k4/hopper/rung1.cu`,
`tools/build_rung1.sh`. Rung 0 (`rung0.cu`, exactness at tiny size) is untouched and must pass on the same card first.

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

For each configuration of the table `kEntries` (13 rows: `f_` = our fast epilogue, `g_` = the generic byte-store epilogue, kept for comparison) and each
shape m = n = k in {4096, 8192}, batch L = 14 (the N moduli: 14 independent GEMMs in one launch, batch z uses modulus z, b = 1), operands
`A[z][m][k]`, `B[z][n][k]` int8 from a counter-based hash, output the u8 residue planes of W, store mode: warm-up (3 launches plus about 0.3 s of load so
the clocks ramp), then at least 10 repetitions in two modes, `single` (one launch between two cudaEvents) and `loop` (R back-to-back launches, divided
by R; R >= 8 and enough for 20 ms). Reference batches L = 4 and 1 follow. Work order (a timeout keeps the valuable lines): (1) default scheduler at the
first shape, then the next, L = 14; (2) the scheduler sweep: for the six `sweep` rows at 8192^3, max_swizzle_size {1, 2, 4, 8} x raster order {AlongM,
AlongN} (CUTLASS `args.scheduler`; `VT<>` needed no change, the knobs are runtime arguments and rung 0 is untouched); (3) L = 4 and 1.

The configurations, the default shapes and batches and the sweep grid are one block of data tables (`kEntries`, `kDefaults`, `kSweepSwizzle`,
`kSweepRaster`). Each row has the compute capability it is for; the harness runs the rows of the device's major and exits 3 if there are none, so a
Blackwell (sm_100 / sm_120) port adds rows and a Variant-like kernel type, nothing else. Every JSON record carries `gpu` (device name) and `cc`.

Output (stdout, one JSON line per variant, shape, batch, scheduler, mode; conventions of `bench/ceiling`): `method` "rung1", `variant`, `m n k batch`,
`swizzle`, `raster` (heuristic = CUTLASS default), `mode`, `launches_per_rep`, `reps`, `tops` (int8 TOPS = 2 m n k L / median), `tops_best`, `tops_worst`,
`median_s/min_s/max_s` and `_ms`, `spread_pct = 100 (max - min) / median`, `exact` and its parts, `regs`, `local_bytes`, `smem_dynamic_bytes`, `gpu`,
`cc`, `sm_count`, `clock_khz_max` (cudaDevAttrClockRate), `ndebug`, `commit` (env BENCH_COMMIT), `status`. `nvidia-smi` is not used; NVML is `dlopen`ed
(no link dependency, `--no-nvml` to skip) and polled every 5 ms by a host thread during each timed block: `sm_mhz_min`, `sm_mhz_mean`, `power_w_max`,
`temp_c`, `throttle_reasons` (OR of the clocks-event bits: 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake);
`null` without NVML. Progress goes to stderr.

**Exactness** (two checks, both in the record): `ref_mismatches` counts mismatches of about 390 sampled output bytes (random positions plus corners and tile
edges) against a host reference computed from the definition (int64 sum over k, `((s mod p) + p) mod p`, p = `kTables.moduli[z]`); `hash_eq_first` says
a position-dependent hash of all of W equals the hash of the first variant run on the same (shape, batch). `exact` = both. W is poisoned with 0xEE
before the checked launch. The hash comparison covers every byte but relative to the first variant: if that one is wrong in the same way as a later one,
only the sampled reference sees it (and only at the sampled bytes). Rung 0 is the full known-answer test.

Exit status: 0 all exact; 1 some record not exact; 2 usage/setup; 3 no row for this device (the 4090 dev box: harmless, message on stderr); 4 sticky
device error after its record; 124 from the build script's `timeout` (a hang; the lines printed so far are in the file).

## Commands

Build and run (the cluster takes the GPU lock itself; run rung 0 first and only continue if it exits 0):

```
export IMMERDINK_CUTLASS_DIR=<cutlass checkout>          # default third_party/cutlass
tools/build_rung0.sh $OUT run                             # exit 0 required
BENCH_COMMIT=$(git rev-parse --short HEAD) RUNG1_TIMEOUT=1800 tools/build_rung1.sh $OUT run   # OUT/imd_rung1, OUT/rung1.jsonl, stderr = progress
```

Quick (about a minute): `RUNG1_ARGS="--quick" tools/build_rung1.sh $OUT run`. Other knobs: `--variants f_256x128,f_128x128` (names, substrings or the groups
all / f / g / quick / sweep), `--shapes 4096,8192` (or MxNxK), `--batches 14,4,1`, `--sweep 8192|4096,8192|none`, `--reps`, `--max-seconds`, `--samples`;
`imd_rung1 --list` prints the table with each kernel's shared memory (no GPU needed). Build flags: `-O2 -DNDEBUG` are mandatory (without NDEBUG CUTLASS's
assert leaves a call in the mainloop, ptxas prints C7510 and serialises the wgmma; the program warns on stderr and records `"ndebug": false`).
`IMD_NVCC_FLAGS="-Xptxas -v"` adds the register report. After the run: `cuobjdump -sass $OUT/imd_rung1` and count `WARPGROUP.DEPBAR` against `IGMMA`
per kernel (the numbers below must reproduce on the cluster's nvcc; the theory asks for this gate on the PACE binary).

## Cross-compile evidence (nvcc 13.4, CUTLASS 4.8.0, sm_90a, -O2 -DNDEBUG; this machine)

No ptxas C7510. No spills in any of the 13 kernels (including ping-pong 128x128, which spilled 8 bytes without NDEBUG in issue 17's table), 168 registers per
thread (consumers raise to 232 at run time), no stack frame. SASS per kernel: DEPBAR 2 in all 13, against IGMMA 4 (K64), 8 (128x256 and 128x128 coop) or 16
(256x128 coop, 128x128 ping), i.e. the wgmma are issued back to back and drained twice, not serialised. One `CALL` per kernel remains (an out-of-line
helper, not a wgmma serialisation; ptxas reports no C7510). Dynamic shared memory per CTA and stages: 128x256x128 and 256x128x128: 196 864 B, 4 stages;
128x128x128: 229 760 B, 7 stages; 128x256x64: 221 568 B, 9 stages.

## Reading the outcome (decision rule from the theory; thresholds are the theory's, fixed before the run)

Primary number: `tops`, mode `loop` (launch amortised), 4096^3, L = 14, `exact` true, best `f_` row; the `single` mode shows what one call pays.

| Result (f_*, 4096^3, L = 14) | Meaning |
|---|---|
| 1260-1456 TOPS | Theory confirmed: the stock gap was the epilogue. The issue-16 rule's `R_stock` was pessimistic; re-run the rate projection with these rates (as a new, separately labelled measurement), and issue 22 proceeds to ladder rungs 2-6 with the best row. |
| 1100-1260 | Partial: better than stock, short of the fit. Look at the `g_` / `f_` difference (epilogue cost), `sm_mhz_min` (clock), and the sweep before judging the mainloop. |
| <= 1100 | Theory falsified: the epilogue is not the main loss. Next suspects in the theory's order: mainloop slope (cluster sweep, `f_128x256x128_4x1`, K64 stages), then the SASS. A custom (non-CUTLASS) kernel is justified only if the mainloop slope stays 8% or more below cuBLAS (a > 6 ns/K at 128x256) with NDEBUG, cluster, swizzle and our epilogue all in. |
| any `exact` false | Do not use that row's rate. Rung 0 must also have failed or the harness is wrong: stop and look. |

8192^3, L = 14 (stock collapsed to 868 against 1152 at 4096):

| Result | Meaning |
|---|---|
| tops rises with `swizzle` 4-8 (either raster), `sm_mhz_min` stays near max | Locality: the default `max_swizzle_size = 1` was the cause; set the swizzle in the product launch. |
| tops flat over the sweep, `sm_mhz_min` well below `clock_khz_max` / 1000 or `throttle_reasons` has 0x4/0x8/0x20/0x40/0x80 | Power or thermal throttling: not a kernel problem; report per-card, do not tune. |
| tops flat over the sweep, clock flat, still lower than 4096 | Neither: something else (L2 footprint of A and B per wave, wave quantisation is small at L = 14 per the theory). Report; the `loop` against `single` difference and L = 4, 1 rows help. |

Cluster sweep (theory item 3): `f_128x256x128` at 1x1, 1x2, 2x1, 4x1 (`4x1` is new and untested on any card; a cluster launch error is recorded as `status`
"error: ..." and the run goes on; a hang is caught by the timeout, which is why these rows come last). `f_` against `g_` rows: epilogue cost.
Ping-pong 128x128 and K64: theory item 5, low value.

## Not verified (no Hopper here)

* Everything dynamic: that the harness runs, the numbers, the exactness of any variant on the card (rung 0 is the prerequisite), that the 4x1 and 2x1
  128x256 clusters launch, that NVML works in the allocation (it is optional), that the 0.3 s load and 12 repetitions give a stable clock (check
  `spread_pct` and `sm_mhz_min`).
* The sampled reference reads W at about 390 bytes per record; the hash check is relative to the first variant.
* `slot0 = 0, b = 1, batch0 = 0`: the modulus mapping of the product's split launches is rung 0's business.
* Operands are random int8, not residues of a real matrix (rates of tensor cores do not depend on the values; power does, a little).
* Only store mode (no add mode) and only K = m (a single chunk, no K4 chunking at 131008); the epilogue's per-tile cost at the product's K is what the
  m = n = k shapes give.
* No L2 flush between launches (the loop mode runs them back to back, as K4 does).
