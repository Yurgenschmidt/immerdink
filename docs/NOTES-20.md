# Issue 20: timing every K4 tile shape and refitting the planner's choice

Status: R1 data in (2026-10-19). H100 and H200: the rule says ADOPT; the H100 table is shipped as table set `sm90` (`IMD_TABLE_SM90`, registry row H100 132 SMs, matched to the H200 by architecture). Blackwell: KEEP. See "R1 results" at the end.
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
2. **The adoption rule, fixed before the data** (the issue's thresholds, refined after an independent review and before any data), in `k4_fit.py decide`:
   per card, adopt the refit only if on the held-out cases (every 4th **distinct GEMM view** (m, n̂, k̂, b) in sorted order, so a view that appears under two names
   falls on one side; N = 8 and 14 of a view are correlated, not independent rows)
   (a) the geometric mean of the refit pick's measured K4 time over the current table's pick is at most 1/1.015,
   (b) no held-out case/N is more than 3% slower with the refit's pick, and
   (c) the mean leave-one-view-out regret of the refit rule over all cases is at most 2% (absolute; the current table's own regret on the same cases is reported beside it).
   The refit is fitted on the training views; **if it is adopted, the table to ship is the fit on every usable observation** (`table`; the
   training-split fit that was validated is `validated_on_training_split`), which is also exactly what `k4_fit.py fit` prints (same filter, a test pins it).
   A table with a constant that is not positive (a tile that looks free: the data does not identify it, typically because the short marks were cut) or a tile with more than 3 concurrent CTAs is not a table:
   if the rule would have adopted it the verdict is INCONCLUSIVE, if the rule says keep the table is kept (and the reason says which constants were not identified). On a rehearsal on the 4090 with the final sweep, 198 of 280 marks are
   usable, the small tiles' `round_ns` comes out as 0, and the verdict is KEEP (held-out gain 0.00%, regret 0.38%).
   Otherwise the table is kept. INCONCLUSIVE with fewer than 8 cases that have all four tiles. Phase marks below 0.1 ms are not used by either command (CUDA-event jitter of a
   few microseconds is 5% there; A3 used nsys kernel medians for such views).
   **Power.** Rule (a) is a geometric mean over about 16 to 20 held-out rows, most of which have the same pick under both tables: a real 10% gain on three of 35 cases
   would still read KEEP. The rule is built to avoid false adoption, not to find small gains; a KEEP is "no large gain", not "the table is optimal". The held-out views are a
   fixed, not designed, sample.
   **Run on the 4090's own A3 data the rule says KEEP** (held-out gain 0.00%, worst slowdown 0.00%, leave-one-out regret 1.71% against the current table's 1.50%): the shipped table is already
   (nearly) what the rule would produce there, and adopting nothing leaves the 4090's tile choices and output hashes untouched.
3. **How many shapes does a fit need?** `k4_fit.py identify` gives the sampling error of each constant when every (shape, N, tile)
   is timed once with 3% noise (the issue's figure), using only the rows whose K4 phase is at least 0.1 ms (shorter marks are dropped by `decide`). With the 17 benchmark shapes timed in
   **all four tiles** at N = 8 and 14 it is **1–4% for the MAC constants but 11–14% for the per-round constant `round_ns` of the three small tiles** (128×128, 64×128, 64×64;
   on 132 and 188 SMs; 10% for 128×256 on 188 SMs). Without the cut it was 0.6–3% everywhere, because many of the 17 shapes are short: the cut is what matters. That is still far from the
   26% (H100) and 126% (RTX PRO 6000) the issue derived. I could not reproduce those figures and did not find the assumption behind them; one possible difference is that
   they were computed for what R1 *had*, the planner's pick only (14 of 17 cases on 128×256, which carries no information about the other tiles), whereas the new sweep forces all
   four: this is a guess, not a finding. A constant that no shape exercises (or every shape of which is cut) is reported as infinite (tests pin both). The sweep is widened to the
   31 A3 views (12 of A3's 43 cases are not in it, mostly small K4 views) plus four benchmark cases A3 lacked, so the held-out views cover the small and mid regime where the tile choice
   matters (`bench/sweep_k4_tiles.txt`, 35 views × N = 8, 14 × 4 tiles; estimate about 8 min on an H100). The honest consequence: `round_ns` of the small tiles may stay
   uncertain on R1 data; the verdict then rests on the tile *picks* (what the rule judges), not on the constants, and a KEEP or INCONCLUSIVE is the likely outcome for the three small tiles.
4. **CPU check of shared memory, registers and blocks per SM** (`k4_fit.py occupancy`, tested): CTAs per SM of each tile on each
   card from the smallest of the shared-memory, register, thread and block limits. The 4090 gives 1, 2, 2, 3, equal to the shipped
   assumption (`ctas_per_sm`). **H100 / H200 give 1, 2, 3, 6** (more shared memory per SM: 228 KiB against 100 KiB): the cost model's
   concurrency for 64×128 and 64×64 differs from the 4090's, and `K4Cost` holds constants for at most three concurrent CTAs
   (`mac_fs[4]`). The plan itself uses the measured occupancy for its persistent grid (`resident = SMs × occupancy`), so only the
   model is affected. Options for the refit: keep the 4090's concurrency in the model and let the constants absorb it (the fit takes
   `--ctas 1,2,2,3`), or extend `K4Cost` (issue 18's `K4Table` already carries `ctas_per_sm` per table). Registers per thread
   (from `cuobjdump --dump-resource-usage` of the sm_89 build): 248, 252, 168, 168; an sm_90 build may differ.

## What R1 adds

The R1 manifest gets the experiment `tiles.sweep` (`bench/sweep_k4_tiles.txt`, K4 phase marks, the median of 11 profiled calls per case (`--phase-reps 11`)) and the
decision `k4-tiles` (kind `tile_refit` in the campaign kit, which runs `decide` on the records). Offline, the analyst runs
`k4_fit.py fit` on the records for the table and `decide` for the verdict, per card (H100, H200 as a transfer test of H100's table,
RTX PRO 6000).

## Not done here

* **Plan-time timing above the 256 MiB cap** (`k4_tune` on a plan whose scratch exceeds `kK4TuneMaxBytes`). It needs a new value of
  `imd_options_t.k4_tune` (host API change) and so waits for the device-class change (issue 18, host API 14); noted for the same version bump.
* **Hopper tile timings from issue 22's ladder** (R2).
* **nsys kernel medians for the smallest views.** The phase marks are the only K4 timing in the R1 sweep; views whose K4 phase is under
  0.1 ms are not judged. If R1's data leave the small regime undecided, the A5 tooling (`bench/nsys_sweep.sh`) is the way to measure it in R2.

## R1 results (2026-10-19, folder `20261019-132130`, commit 05247d9)

* **H100: ADOPT** (held-out gain 2.12%, worst held-out slowdown 0.00%, leave-one-out regret 1.23%; the shipped table's own regret on the same rows is 2.21%). **H200: ADOPT** (2.98%, 0.80%; shipped 2.20%). **RTX PRO 6000: KEEP** (0.75%, 1.44%;
  its refit leaves one constant at zero, which a table cannot have).
* The refit changes only two picks (rows timed in all four tiles): r_mm N=14 from 128x128 to 64x64 (24% faster on H100, 32% on H200) and x512_32k N=8 from 128x128 to 64x128 (11% faster); every large case (sq4096, sq8192, t_cc,
  t_permout, s_cc16) keeps 128x256. The rule's power is low (a geometric mean over 13 held-out rows), so this is a small, solid gain, not a large one.
* The H100 table, fitted on H100 data and applied unchanged to the H200's data, gives the same two picks and lowers the mean regret from 2.20% to 0.80%: one table serves both. It is `kK4CostsSm90` in `src/host/k4_config.hpp`
  (round_ns | mac_fs): 128x256 3383|265, 128x128 2838|464/281, 64x128 737|584/414, 64x64 363|949/685/505, with the 4090's concurrency assumption (1, 2, 2, 3 CTAs per SM). `bench/test_k4_fit.py` checks the Python mirror
  of this table against the library's own pick on 150 random shapes.
* Not done: the real concurrency of the small tiles on Hopper is 3 and 6 CTAs per SM (the model absorbs it into the constants); an sm_120 table (its small-tile constants were not identified); the 0.1 ms cut removes about a third of the marks (186 of 280 usable on H100),
  so `round_ns` of the small tiles is the least certain constant.
