# The campaign kit: planned, rule-based cluster runs

Status: built for issue 15 (the version needed before the first cluster run, R1). Code: `bench/campaign/`. Gates: `bench/campaign/gates.py`.

## What this is and why

immerdink is tuned on one GPU, the RTX 4090. Moving it to Hopper (H100, H200) and Blackwell (RTX PRO 6000) needs measurements on
those GPUs, and cluster GPUs (the Georgia Tech PACE cluster) are scarce: the plan is three big shared runs (R1 to R3) plus one
spare, not many small ones. A run that fails half-way, or whose result nobody can trust, costs a whole run. The campaign kit
exists so that a run

- is **planned**: a *manifest* lists every experiment, its priority and its time estimate before anything is submitted;
- is **rule-based**: the formulas that turn numbers into a decision (adopt this tile shape? go or no-go for wgmma?) are written
  in a *decisions file* before the data exist, and are bound to the manifest by a hash;
- is **robust**: the *runner* resumes after a kill, copies results back as it goes, and records the GPU, clocks and tool versions;
- is **checked**: the *analyzer* gives one *verdict* per decision, and says INVALID instead of guessing when a result folder is
  damaged, from the wrong commit, or judged by an edited decisions file.

Terms used below. **Experiment:** one unit of work with its own output folder (a probe, a benchmark sweep, a command).
**Priority:** P0 must-have, P1 and P2 only with leftover time. **Class:** a GPU model the run targets (h100, h200, blackwell, rtx4090);
a run starts one job per class. **Pin:** a full commit hash that must be the one checked out. **Interleaved (paired) timing:** two
variants timed alternately call by call in one process, so both see the same clock and temperature; the unit of comparison is the
pair. **Spread:** how much repeated timings of one thing vary.

## The parts

| Part | File | Job |
|---|---|---|
| Manifest | `<run>.toml` (kept next to the cluster scripts, in generalWork, because it pins immerdink commits) | experiments by priority, estimates, class budgets, pinned commits, the decisions file and its sha256 |
| Decisions file | `<run>.decisions.toml` | one `[[decision]]` per question: kind, which experiments it needs, thresholds |
| Runner | `bench/campaign/runner.py` | runs a manifest for one GPU class; resumable; copy-back; environment capture |
| Analyzer | `bench/campaign/analyze.py` | integrity check, verdicts, regenerated tables |
| Statistics | `bench/campaign/stats.py` | paired interval, tie rule, regression guard |
| Gates | `bench/campaign/gates.py`, `archgate.py`, `estimates.py` | everything that must pass before a submission |
| Submission | `submit_campaign.sh` (generalWork, `phoenix/immerdink/`) | wrapper around the existing PACE submitter; checks out the pinned commits |

### Manifest

TOML. Experiment kinds: `probe` (built in: `env`, `clocks`, `tools`, `limits`), `cmd` (an argv; `{src}`, `{build}`, `{out}`,
`{class}`, `{exp}` are replaced) and `sweep` (a bench sweep file of the pinned source, run through `bench/run_sweep.sh`).
Every non-probe experiment lists the `outputs` it must leave behind; a missing or empty one makes it `failed`. An experiment may have
`needs = [other ids]`, `risky = true` (runs after the non-risky ones of its priority), `classes = [...]`, a `timeout_min`, and a `[experiment.smoke]` table that
overrides keys for `--smoke` runs. The order of execution is: priority, then non-risky before risky, then file order.
`python3 bench/campaign/manifest.py pin <manifest> --src <checkout>` rewrites the pins to a checkout's HEAD and refreshes the decisions hash.

### Decisions

Four kinds, each a pure function of the records (`decisions.py`):

- `threshold`: one number (an aggregate of a field) against two limits: GO, NO-GO, or INCONCLUSIVE in between.
- `paired`: interleaved A/B records. Per case, the median of the per-pair ratios t_A / t_B and a 95 % interval
  ([Thompson 1936](https://doi.org/10.1214/aoms/1177732502): the distribution-free interval of a median from the binomial order
  statistics; applied to log ratios). *Tie rule:* a case is a tie when 1 lies inside the interval, or when the case is noisy
  (range spread above 5 %) and the median ratio is within 10 % of 1. Verdict: ADOPT (enough cases faster, none slower),
  KEEP (none faster, or the variant is not bit-identical to its reference), INCONCLUSIVE (mixed, or too few pairs).
- `parity`: the parity gate of the tracking issue over comparison records (`sweep_compare`). Per case, the best configuration of each
  of two families at DGEMM-level accuracy (accuracy at or below cuBLAS DGEMM's on the same inputs), ratio A / B of TFLOPS-eq
  (default immerdink / GEMMul8 fast). PASS when the geometric mean over the cases (listed ones such as `s_batch` excluded) reaches
  `geomean_at_least` and no case is below `case_floor` once its larger spread is added; FAIL otherwise; INCONCLUSIVE when a case lacks
  a record. The report prints the per-case table, the class geometric means and the mean with every case. On the issue-11 folders
  it gives 0.84 (H100), 0.78 (H200), 1.35 (RTX PRO 6000) without `s_batch` and 1.09 / 1.01 with it, the numbers of the tracking issue.
- `guard`: records against a stored baseline. A slowdown beyond `max(2 %, 2 x spread)` is flagged. With the raw repetitions the
  spread is the *robust* one, the noise of the median (half the width of its interval); the bench's range spread is dominated by
  one outlier (8 to 12 % on a quiet RTX 4090 in the rehearsal) and would hide a 10 % slowdown. Without repetitions the range spread is used, which is
  more lenient.

New kinds are added as functions in `decisions.py`
and described in `REQUIRED` / `OPTIONAL`; the lint rejects any parameter it does not know.

### Interleaving different methods: `xduel`

`imd_bench --methods xduel --xduel-b dgemm,emu:fix55,gemmul8:fast:14` times immerdink (the `--duel-a` side) and each baseline
alternately, call by call, in one process (same protocol as `duel`, which only compares two immerdink configurations). The
baseline's record carries `xduel_ratio_median` (baseline time / immerdink time) and immerdink gets the record `xduel-imd`; with
`--keep-samples 1` it holds the paired times (`pair_t_a` baseline, `pair_t_b` immerdink), and a `paired` decision with
`method = "xduel-imd"` turns them into a verdict. This is how "alternate the compared methods" is met for the parity gate,
where sequential timing on a cluster node mixes clock and node drift into the ratio. (4090 check: immerdink / GEMMul8 fast N = 14
at 2048³ = 1.35, interval 1.351 to 1.355.)

### Runner and result folder

```
<out>/MANIFEST.toml DECISIONS.toml   copies of the inputs as run
      LOCK.json                      manifest + decisions hashes, class, GPU name, pinned and actual source commit
      env.json                       host, GPU, driver, SLURM ids        journal.jsonl  append-only event log (fsynced)
      exp/<id>/...                   the experiment's files, then DONE.json (status, argv, clocks before/after, size+sha256 of every file)
      COMPLETE.json                  written last: the job reached its end
```

DONE.json is the commit marker: written last, atomically (temp file, fsync, rename). **Resume** runs the same command again:
experiments with a valid DONE.json are kept, the one that was running is wiped and redone (the experiment process dies with the
runner, even on `kill -9`), and a folder that belongs to another manifest, decisions file, class or source commit is refused.
**Time:** an experiment starts only if it fits before the deadline (Slurm's job end time, or `--wall-min`) minus a copy-back
reserve (6 min); P1 and P2 need the manifest's margin on top. **Copy-back:** after every experiment, every 10 minutes, on SIGUSR1
(Slurm warns 5 minutes before the wall limit), on SIGTERM and at the end; DONE.json goes last inside an experiment folder and COMPLETE.json
last of all, so a copy cut short never looks finished. **Clocks:** a cluster job cannot lock GPU clocks (no root), so each
experiment records them before and after, and comparisons are paired.

### Verdicts and INVALID

`analyze.py verdicts <job folder> --decisions <the committed decisions file>`. INVALID (and then no decision is read from the
data) when: a file of an experiment is missing, truncated, resized, altered or unlisted; LOCK.json or an input copy is missing or changed; the
source commit is not the pinned one or the tree was dirty; the decisions file differs from the one the manifest pinned; the GPU
does not belong to the class; a record file does not parse. INCONCLUSIVE when the data are sound but absent or between the limits
(an experiment failed, was skipped for time, or never started). A job without COMPLETE.json is reported as unfinished.
`analyze.py tables` regenerates the benchmark tables; `analyze.py golden` regenerates an older job folder's tables and diffs them against the ones that job wrote
(the issue-11 folders reproduce exactly, all 15 table files).

## The gates

Run before every submission (`python3 bench/campaign/gates.py <gate> ...`).

| Gate | Command | Checks | Needs |
|---|---|---|---|
| G1 | `g1 MANIFEST --src DIR` | schema, ids, priorities, no placeholder text, pins are 40-hex and exist, CUTLASS and GEMMul8 pins equal what the pinned commit records, sweep files exist at the pinned commit, decisions file hash and content, P0 estimate x (1 + margin) inside each class budget | CPU |
| G2 | `g2 --build-dir DIR` | every library source compiles for sm_89, sm_90a, sm_120a; registers, stack, spills and tensor-core SASS families equal the committed table (`arch_table.json`); wgmma assembles for sm_90a only (about 30 s) | CPU, nvcc |
| G3 | `g3 MANIFEST --src DIR --build DIR --work DIR` | kill -9 during an experiment, resume, copy-back equals the run folder, expected verdicts, under Python 3.12 | the GPU (takes it like a benchmark) |
| G4 | `g4` | the unit tests under the current Python and 3.12; the issue-11 folders regenerate their tables | CPU |
| G5 | `g5 FOLDER` | every `bit_identical` / `elements_differing` field of every record says identical | a result folder |
| G6 | `g6 MANIFEST --logs DIR... --src DIR` | sweep estimates not below 80 % of what the issue-11 logs imply | issue-11 logs |
| G7 | `g7 MANIFEST` | `DRY_RUN=1 submit_campaign.sh` on the PACE login node: environment, pins, CUTLASS, GEMMul8 | the login node |

The G8 gate of the tracking issue (new sm_90a kernels) has nothing to check yet; `arch_table.json` has the rule slots for it
(`must_contain` / `must_not_contain` instruction families per architecture).

## What a run looks like

1. The assistant builds the manifest for the run and passes the gates.
2. Before the first real submission of a manifest, `PREPARE_ONLY=1 IMMERDINK_SRC=<clone> bash submit_campaign.sh campaigns/R<n>.toml` does the clone, the
   checkout of the pinned commit, the CUTLASS fetch and the lint for real, checks that the pinned commit's runner and analyzer start under the env's Python, and
   stops: no job, no GPU time. The owner then runs, on the PACE login node: `git -C ~/repos/generalWork pull && IMMERDINK_SRC=<an immerdink clone> bash ~/repos/generalWork/phoenix/immerdink/submit_campaign.sh campaigns/R<n>.toml` (add `DRY_RUN=1` first). The wrapper
   fetches the pinned commit into its own clone, lints the manifest with that commit's own tools, copies manifest and decisions next to the frozen sources and
   hands over to `submit_immerdink_metrics.sh`; each job then runs `runner.py` from the pinned commit (`pace_immerdink_metrics.sbatch`, campaign mode).
3. The owner downloads the result folder and names the run. The analysis is the assistant's.

## Rehearsal on the RTX 4090

`campaigns/R0-rehearsal.toml` (generalWork) exercises everything on this machine in about a minute of GPU: four probes, a three-case sweep
with raw repetitions, an A/A duel (must be a tie), a g=1 against g=0 duel (bit-identical, about 2.9 x), kill -9 and resume. Expected verdicts: sweep-sanity GO,
aa-resolution KEEP, g-grouping ADOPT. `bench/imd_bench --keep-samples 1` adds the raw repetition times (`samples_s`; duel-b: `pair_t_a`, `pair_t_b`) that the paired and guard computations need.

## Environments

`environment.yml` (4090) and `hpc_env.yml` (PACE) carry `cuobjdump`, `nvdisasm`, `compute-sanitizer`; the PACE one also numpy and pytest (the 4090 one already had them).
The PACE env check (`hpc_env_check`) only warns about the new tools, so an existing env keeps working until `FORCE_UPDATE=1 bash setup_hpc_env.sh`.

## Limits of this version

- The kit contains no R1 experiments. R1's content (baselines, INT8 ceilings, tile table, pack/CRT diagnosis) is defined by the issues that use it; they add experiments and decisions.
  A decision kind for the K4 tile-table fit belongs to the tile issue; the others (threshold, paired, guard, parity) are general.
- The GPU-side steps of a real submission (the sbatch campaign mode, the runner inside a job) are untested on PACE; the login-node dry run (G7) and `PREPARE_ONLY` pass.
- `xduel` against GEMMul8 was tested on the RTX 4090 only (sm_89).
- The runner protocol has no TLA+ model; the kill -9 test is the check.
