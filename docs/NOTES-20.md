# Issue 20: timing every K4 tile shape and refitting the planner's choice

Status: offline part done (fit script, adoption rule, CPU occupancy check, R1 sweep); the data come from R1.
Written for a reader new to the project.

## What this is about

K4, the INT8 matrix-multiply kernel, exists in four tile shapes: 128×256, 128×128, 64×128 and 64×64 (rows × columns of
the output block one thread block computes). A big tile reuses loaded data best but wastes work on small or thin
problems; a small tile fills more of the GPU. The planner chooses per call. It predicts each tile's time from a small cost
table and takes a smaller tile only if it is predicted at least 5% faster than 128×256. The table was fitted on the RTX 4090
(docs/NOTES-A3.md) and the fit script was never committed. On H100 and H200 (132 SMs) the planner picks 128×256 in 14
of 17 benchmark cases (13 on the RTX PRO 6000, 188 SMs). The question: do the other cards want other constants?
A refit needs no new kernel.

## What was done

1. **`bench/k4_fit.py`: the missing fit script**, plus its self-test `bench/test_k4_fit.py` (ctest `bench_k4_fit_selftest`).
   * It mirrors the planner's integer model (`k4_estimate_fs`, `k4_select`) exactly, and a test compares its pick with the
     library's own (`imd_lowering`'s `k4_config`) on 150 random shapes at 46, 128, 132 and 188 SMs.
   * **It reproduces the shipped 4090 constants from the A3 raw data**: `bench/k4_fit.py a3-check` gives round_ns / mac_fs
     2716 / 418, 2550 / 453 456, 1424 / 505 612 and 1333 / 944 1088 802, equal to `kK4Costs`, from the 84 cases (42 views at
     N = 8 and 14; nsys kernel medians for the small and medium views, the bench's K4 phase marks for the large ones; the view
     `s_df`, which the shipped fit left out, excluded). The data are in `immerdink-eval/legacy/bench/results/a3-configs-*.jsonl`
     (outside the product repository); the test is skipped where they are missing.
   * Relative least squares per configuration, constants non-negative.
2. **The adoption rule, fixed before the data** (the issue's thresholds), in `k4_fit.py decide`:
   per card, adopt the refit only if on the held-out cases (every 4th case by name, alphabetical)
   (a) the geometric mean of the refit pick's measured K4 time over the current table's pick is at most 1/1.015,
   (b) no held-out case/N is more than 3% slower with the refit's pick, and
   (c) the mean leave-one-case-out regret of the refit rule over all cases is at most 2%.
   Otherwise the table is kept. INCONCLUSIVE with fewer than 8 cases that have all four tiles. Phase marks below 0.05 ms
   are not used (CUDA-event marks are too coarse there; A3 used nsys kernel medians for such views).
   **Run on the 4090's own A3 data the rule says KEEP** (held-out gain 0.13%, worst slowdown 0.00%, regret 1.71%): the shipped table is already
   (nearly) what the rule would produce there, and adopting nothing leaves the 4090's tile choices and output hashes untouched.
3. **How many shapes does a fit need?** `k4_fit.py identify` gives the sampling error of each constant when every (shape, N, tile)
   is timed once with 3% noise (the issue's figure). With the 17 benchmark shapes timed in **all four tiles** at N = 8 and 14 it
   is 0.6–3% for every constant on 128, 132 and 188 SMs (e.g. 64×64 on 132 SMs: round 3%, one-CTA round 3%, two-CTA round 3%,
   three-CTA round 1%). That differs from the 26% (H100) and 126% (RTX PRO 6000) the issue derived. I could not reproduce those
   figures and did not find the assumption behind them; the likely difference is that those are for what R1 *had*, the planner's
   pick only (14 of 17 cases on 128×256, which carries no information about the other tiles), whereas the new sweep forces all
   four. A constant that no shape exercises is reported as infinite (a test pins that). The sweep is nevertheless widened to the
   33 A3 views plus the benchmark cases A3 lacked, so the held-out cases cover the small and mid regime where the tile choice
   matters (`bench/sweep_k4_tiles.txt`, 35 views × N = 8, 14 × 4 tiles; estimate about 8 min on an H100).
4. **CPU check of shared memory, registers and blocks per SM** (`k4_fit.py occupancy`, tested): CTAs per SM of each tile on each
   card from the smallest of the shared-memory, register, thread and block limits. The 4090 gives 1, 2, 2, 3, equal to the shipped
   assumption (`ctas_per_sm`). **H100 / H200 give 1, 2, 3, 6** (more shared memory per SM: 228 KiB against 100 KiB): the cost model's
   concurrency for 64×128 and 64×64 differs from the 4090's, and `K4Cost` holds constants for at most three concurrent CTAs
   (`mac_fs[4]`). The plan itself uses the measured occupancy for its persistent grid (`resident = SMs × occupancy`), so only the
   model is affected. Options for the refit: keep the 4090's concurrency in the model and let the constants absorb it (the fit takes
   `--ctas 1,2,2,3`), or extend `K4Cost` (issue 18's `K4Table` already carries `ctas_per_sm` per table). Registers per thread
   (from `cuobjdump --dump-resource-usage` of the sm_89 build): 248, 252, 168, 168; an sm_90 build may differ.

## What R1 adds

The R1 manifest gets the experiment `tiles.sweep` (`bench/sweep_k4_tiles.txt`, K4 phase marks with 31 profiled calls per case) and the
decision `k4-tiles` (kind `tile_refit` in the campaign kit, which runs `decide` on the records). Offline, the analyst runs
`k4_fit.py fit` on the records for the table and `decide` for the verdict, per card (H100, H200 as a transfer test of H100's table,
RTX PRO 6000).

## Not done here

* **Plan-time timing above the 256 MiB cap** (`k4_tune` on a plan whose scratch exceeds `kK4TuneMaxBytes`). It needs a new value of
  `imd_options_t.k4_tune` (host API change) and so waits for the device-class change (issue 18, host API 14); noted for the same version bump.
* **Hopper tile timings from issue 22's ladder** (R2).
* **nsys kernel medians for the smallest views.** The phase marks are the only K4 timing in the R1 sweep; views whose K4 phase is under
  0.05 ms are not judged. If R1's data leave the small regime undecided, the A5 tooling (`bench/nsys_sweep.sh`) is the way to measure it in R2.
