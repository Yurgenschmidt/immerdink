# Tuning immerdink for a GPU: the method

This guide explains how immerdink's kernels are made faster on a particular GPU. It covers how to find
what limits a kernel, how to change it without risking correct results, how to predict the gain, and how
to decide whether the change becomes the default. It is written for someone who has done GPU performance
work but is new to immerdink. The worked example throughout is the H100/H200 round of November 2026: it
took the sm_90 default from instruction set 447 to 8127, worth 1.82× on the r_mm shape on the H200.

The method has six steps:

1. Compute the floors.
2. Name the limiter, with evidence.
3. Put the change behind a switch.
4. Write the prediction.
5. Measure in one job.
6. Explain the gap and decide.

Steps 1, 2 and 4 cost no GPU time. That matters because cluster time is hard to get.

## Terms

- **Emulation.** immerdink computes an FP64 matrix product (or tensor contraction) with INT8 tensor-core
  products. Each operand is split into residues modulo N small primes ("moduli"). Each residue product is
  an INT8 GEMM, and the result is rebuilt with the Chinese remainder theorem (CRT). This is the Ozaki
  scheme II [1].
- **The kernels of one call.** A call runs several kernels:
  - **K1:** scaling.
  - **KL / KF / K2 / K3:** the pack kernels. They scale each operand and write its residues as INT8
    planes. KL handles operands read along rows, KF handles fused permutations.
  - **K4:** the INT8 GEMM.
  - **K5:** the CRT and the rounding back to FP64.
  Most tuning work has been on the pack kernels and K5, because K4 is already a tensor-core GEMM.
- **Device class.** A compute capability, for example sm_89 (RTX 4090), sm_90 (H100, H200) or sm_120
  (RTX PRO 6000 Blackwell). Defaults are chosen per class.
- **Instruction-sequence set ("seq bits").** An integer whose bits switch alternative code paths on or off.
  Examples: bit 1 is the E1 emitter, bit 512 is kllong, bit 4096 is klcs. It is defined in
  `src/host/seq_config.hpp`. Each class has one default set, its row in `kSeqClasses`.
- **Twin.** A kernel compiled for one setting of the seq bits. A set costs nothing in the others, because
  the bits are template parameters. Builds with `IMMERDINK_SEQ_TWINS=OFF` keep only the baseline.
- **Duel.** `imd_bench --methods duel` runs two configurations (`--duel-a`, `--duel-b`) interleaved in one
  process on the same inputs. It reports `ratio_median` = time(a) / time(b) with a p10-p90 range, and
  whether the outputs are bit-identical. The speed-up of a over b is 1 / ratio_median.
- **TFLOPS-eq.** 2mnk (times the batch count) divided by the time: the FP64 rate the call is equivalent to.
  Results are reported in TFLOPS-eq first, time alongside.

## Step 1: compute the floors before measuring

For each kernel, write down two lower bounds on its time:

- **DRAM floor.** Bytes that must move divided by the achievable bandwidth (about 90 % of the peak).
  Example, r_mm (1000 × 64 × 50000) on the H100: read A as FP64 (400 MB), write its INT8 planes (750 MB),
  read them back in K4 (750 MB). That is 1.9 GB, or about 0.57 ms at 3.35 TB/s. The measured call was
  1.89 ms, 3.3× the floor. The gap was the first clue that something other than bandwidth was limiting.
- **Issue floor.** Thread instructions per element (from `ncu` dynamic counts, or SASS weighted by loop
  trip counts) times elements, divided by the issue rate: 32 threads × 4 schedulers × SMs × clock, at the
  achievable issue efficiency (about 0.7–0.8).

A useful card-level number is the **issue budget per DRAM byte**:

  B = 32 × SMs × 4 × f_SM / BW_DRAM

This is how many thread instructions the GPU can issue per byte it reads at full bandwidth. A conversion
kernel that needs I instructions per byte is bandwidth-bound if I ≤ ~0.7 B, and issue-bound otherwise.

| card | B (instructions per DRAM byte) |
|---|---|
| RTX 4090 | 41 |
| RTX PRO 6000 Blackwell | 35 |
| H100 SXM | 10 |
| H200 | 7 |

The same pack code that is bandwidth-bound on the RTX cards is issue-bound on Hopper, and worst on the H200
(faster HBM, same SMs). This one number explained why the tuning done on the 4090 did not carry over to
Hopper, and it told us that Hopper needed fewer instructions, not fewer bytes.

The larger of the two floors is the target. The gap between it and the measured time is what tuning can
still win. Example: in October the KL time at r_bigk on the H200 was 588 µs, against an issue floor of
about 390 µs and a DRAM floor of about 337 µs, so about a third was still available.

## Step 2: name the limiter, with evidence

Do not change code until the limiter is named and backed by a measurement:

1. **Phase split.** Which kernel dominates the call? Use nsys or serialized timing. Before changing KL,
   make sure KL is the problem.
2. **Stall reasons and pipe use** of that kernel (`ncu`): issue active, the per-pipe utilisation
   (fmaheavy, fmalite, ALU, FP64), and the warp stall breakdown (long_scoreboard = waiting for memory,
   barrier, math-throttle, and so on).
3. **Instructions per element**, attributed to source lines (`ncu` source counters, or `cuobjdump -sass`
   with the loop trip counts). This is where most levers came from: the emit loop's per-modulus sign
   multiply (removed by O1), and a 64-bit multiply that only the heavy FMA pipe can run (moved to a float
   multiply-add by fq).

### Traps we fell into, and the checks that catch them

- **Co-bound pipes.** KL was limited by issue and by the fmaheavy pipe together (issue ≈ 2 × fmaheavy).
  Moving work from one pipe to another then gains nothing; only cutting total instructions helps. klf64 was
  predicted at −40 % KL time and measured at −10 to −15 % for this reason. Check: compare issue active with
  each pipe's share before predicting a pipe move.
- **Predicting a pipe move.** The gain from moving work off a saturated pipe shows up as a drop in the
  math-throttle stall. fq's float multiply-adds split about 50/50 between fmaheavy and fmalite, not all
  onto fmalite. Assume a 50 % lite share unless measured.
- **Lockstep (phase-locked SMs).** With one block per SM and uniform tiles, every SM runs the same phase at
  the same time. In KL's pass 1, DRAM was busy (65–97 %) while issue sat at about 12 %. In pass 3 the
  pattern reversed. A per-SM Little's-law argument [2] about memory parallelism then fails, because the
  shared DRAM is the limit. A prefetch (klpf) was built on that argument and made KL 1.8 % slower. Check:
  `ncu --section PmSampling` with a short interval shows the DRAM and issue time series. An
  autocorrelation of 0.85–0.89 at the row period confirmed the lockstep on both cards. Note that ncu
  stretches the time axis by about 1.3×.
- **Cache working set.** Rows in flight × row bytes against the L2 size (H100: 132 rows × 400 KB = 53 MB,
  more than the 50 MB L2) tells you whether re-reads hit L2 or DRAM.
- **Check every memory level.** NVIDIA GPUs have no L3. The levels, fastest first, are:
  1. **Registers per thread.** They set occupancy. KF at 112 registers fits only 2 blocks per SM on the
     H200.
  2. **L1 / shared memory per SM.** 256 KB combined on Hopper. klload keeps rows of up to two chunks on
     chip; klsplit was killed on its L1 counts; K4's tiles live here.
  3. **L2.** Shared by all SMs. On Hopper it is split into two partitions, and traffic across the
     partition link costs energy; that is the lead for K4's higher energy per operation at n = 32768.
  4. **DRAM / HBM.**

  Hopper also lets the blocks of a thread-block cluster read each other's shared memory, and the TMA unit
  can multicast loads across them. Neither is used yet.
- **Power cap.** PACE's H200s run under an administrative 560 W cap. Long dense runs drop the clock
  (1290 MHz against 990 MHz between two sizes). Read sustained results per clock, not per second.

A lever whose premise fails these checks is killed before it is built. That is cheaper than a cluster job.
klsplit was killed this way: L1 counts showed it could win only about 23 %.

## Step 3: put the change behind a seq bit

Every change goes in as a new seq bit, off by default, so that the old and new code run side by side in
one binary:

- `src/host/seq_config.hpp`: the bit constant, `kSeqAll`, any dependency (for example klf64 needs klwarp),
  and how the bit maps to the kernel's template value (`seq_kl_kb`, `seq_pack_s`).
- `src/contract.cu`: the kernel twin (template value), its launch case, and the host rule for when it
  applies.
- Tests:
  - `tests/test_seq_table.cpp`: CPU-only; the table and its rules, in both twin configurations.
  - `tests/test_seq_select.py`: forced sets read back, invalid sets rejected.
  - `tests/test_kl_bits.py`: the kernels a forced set really launches.
  - The bit-identity suites (`contract_*_kl<set>`, `output_hashes_kl<set>`): add the new set to the
    CMake suites.
  - Bench and harness masks: bench's `--seq` digits, `harness.py`'s hook arrays.
- **Before any cluster job, compile every changed `.cu` file locally for the target architecture**
  (`nvcc -c --generate-code=arch=compute_90,code=sm_90 ...`). Do it three ways: plain, with
  `-DIMD_BUILD_TEST_HOOKS=1`, and with `-DIMD_NO_SEQ_TWINS=1`. That takes about 2.5 minutes per file. A
  misplaced template header once cost a whole H200 job.

A bit can be a runtime argument instead of a twin when it changes no instruction. klph, for example, is a
start delay passed as a kernel argument.

## Step 4: write the prediction before the job

Write predictions in the findings file before the job is submitted:

- **One number (or range) per shape and per card.** State which kernels the bit touches and which shapes
  take its path. sq4096 does not take the long-row path, so kllong predicts 1.00 there.
- **Combinations as an explicit product** of the single-bit factors for each shape. The bits turned out to
  be nearly independent: on the H200, the product of the single-bit duels matched the combined duel within
  1 % on every shape:

  | shape | kllong | O1 | fq | klcs | product | measured 8119 vs 439 |
  |---|---|---|---|---|---|---|
  | r_mm | 1.623 | 1.038 | 1.069 | 1.015 | 1.828 | 1.820 |
  | r_bigk | 1.283 | 1.024 | 1.044 | 1.015 | 1.392 | 1.397 |
  | r_bigk, N=8 | 1.406 | 1.022 | 1.046 | ~1.01 | 1.518 | 1.516 |

  A written range that left out one factor (kllong) was the only large miss in that round. The product
  form makes that mistake visible.
- **A kill criterion.** For example: "if the pass-1 long_scoreboard stall does not drop, the premise is
  wrong."

## Step 5: measure in one job

- **One job per card, run once, covering every question.** Run duels first (they are the result), then
  `ncu` (the explanation), then the tests for the new sets (correctness). Cap each step so that the tests
  always get their reserved time.
- **Duels:** interleaved, the same inputs, `--duel-a seq=<new>,crt=f64 --duel-b seq=<old>,crt=f64`.
  `crt=f64` is needed because the class default's limb-CRT bit (8) cannot be forced. The forceable twin of
  a default is therefore the default minus 8: 8119 for 8127, 439 for 447.
- **An untimed warm-up first.** The first bench process after a build pays a one-time start cost of
  100–150 s.
- **Job hygiene:**
  - a private git worktree when other jobs share the source checkout;
  - `timeout -k 30` on every step;
  - a pre-submission review of the script by a cheap model.
- **Results:** duels and their bit-identity go in the PR comment, the analysis in the findings file. Data
  are not committed to this repository.

## Step 6: explain the gap, then decide

- Compare each measurement with its prediction. Explain every miss in a sentence, and write the lesson into
  this document when it is general.
- **Adoption rule for a class row:** the new set must be bit-identical on every case, and no shape may lose
  more than 3 % on any card of the class. The sm_90 row serves both the H100 and the H200, so both need a
  duel. The class row in `kSeqClasses` gets an evidence comment: the jobs, the per-shape speed-ups, and
  what it replaced.
- The row change also updates the tests that pin the default (`test_seq_table.cpp`, `test_seq_select.py`).
  Run the default-set tests on that class before the merge.

## Porting to another card

What carries over unchanged: the six steps, the seq-bit mechanism, the duel and test harnesses, and the
traps in step 2. What must be re-derived per card:

- **The floors and B:** SMs, clock, bandwidth, L2 size.
- **The hardware assumptions of each sequence.** `docs/NOTES-21.md` ("Hardware-specific assumptions") lists
  them:
  - The FP64 rate. The RTX cards run FP64 at 1/64 rate, so Q1, F1 and the limb CRT stay off there.
  - `F2I.U64` and `I2F.F64.U64` rates.
  - Register limits per kernel.
  - The compiler version: a different ptxas can emit longer sequences.
- **Which pipe and which phase bind**, on that card's own `ncu` data.

### Proposed: one-job tuning of a class row

Because the bits multiply, a new card's default row can come from a single job:

1. Duel each bit alone against the current row, on one shape per family (square, long-k, low-k, skinny,
   batched, tensor).
2. Compute the predicted best set as the product of the winning factors per shape, respecting the
   dependencies in `seq_valid`.
3. Duel the top one or two predicted sets against the current row in the same job, with the bit-identity
   tests for those sets.

This automates what took several jobs on Hopper. It is not built yet.

### Where solvers and provers could help

- **SMT (for example Z3).** Prove that an arithmetic rewrite is exact for every input in range before
  building it. Examples are the E1 range condition (`s' + h < 2^24`), O1's offset-binary emit and fq's
  float quotient. Today this is done by hand and by tests.
- **Constraint programming (for example OR-Tools CP-SAT).** Choose discrete launch settings under hard
  limits: blocks per SM, register caps, shared memory, tile sizes, and per-plan memory footprints. This
  fits once a cost model exists. The first candidate is KF's register cap (112 registers today, which
  allows only 2 blocks per SM).
- **Equality saturation or superoptimization** of the inner instruction sequences, with an SMT check of
  each rewrite. This is research-grade. The levers so far were found by reading SASS against the floors.

## Worked example: KL on Hopper, November 2026

| bit | limiter evidence | change | KL time (H200) | call speed-up (H200 r_mm / r_bigk) |
|---|---|---|---|---|
| klf64 (256) | instructions in the sigma filter | the filter in binary64 | −10 to −15 % (−40 % predicted: co-bound pipes) | in 447 |
| kllong (512) | barrier stall on long rows, 1 block per SM | direct loads for rows longer than two chunks | barrier stall halved | 1.623 / 1.283 |
| O1 (1024) | emit loop bound on fmaheavy | offset-binary emit, no per-modulus sign multiply | | 1.038 / 1.024 |
| fq (2048) | 64-bit multiply on fmaheavy only | quotient as one binary32 FFMA | −9.1 %, math-throttle stall 16.6 → 10.6 % | 1.069 / 1.044 |
| klcs (4096) | DRAM 55 % plus long_scoreboard, reads evicted by the plane writes | streaming stores (`st.global.cs`) | −7.8 %, DRAM reads 791 → 543 MB | 1.015 / 1.015 |
| klpf (8192) | assumed per-SM memory parallelism | prefetch | +1.8 %, falsified by lockstep | dropped |

The H100 agreed on every shape (r_mm 1.665, r_bigk 1.381). The row moved from 447 to 8127 in PR #64. The
next lever, de-phasing the first wave (klph), tests the lockstep finding directly.

## References

1. K. Ozaki, Y. Uchino, T. Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix
   multiplication using an integer modular technique", *The International Journal of High Performance
   Computing Applications* (2026). doi:[10.1177/10943420261467787](https://doi.org/10.1177/10943420261467787),
   [arXiv:2504.08009](https://arxiv.org/abs/2504.08009).
2. J. D. C. Little, "A proof for the queuing formula: L = λW", *Operations Research* 9(3):383–387 (1961).
   doi:[10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383).
3. S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore
   architectures", *Communications of the ACM* 52(4):65–76 (2009).
   doi:[10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785). The floors in step 1 are a
   per-kernel roofline with an issue ceiling added.

Detailed records: `docs/NOTES-21.md` (the seq mechanism and per-class assumptions), `docs/BENCHMARKING.md`
(the bench and clean-number rules), `docs/CAMPAIGN.md` (cluster campaigns), and the immerdink-eval findings
(`findings/hopper-tuning-20261108.md`, `analysis/20261110-k15/findings.md`).
