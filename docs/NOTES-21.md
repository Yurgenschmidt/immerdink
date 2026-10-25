# Issue 21: pack and CRT on Hopper — what is done offline, and what R1 must tell us

Status: the offline part is done (profile on the RTX 4090, the L2 knobs, a bandwidth microbenchmark and its analysis). The Hopper limiter is
**not named yet**: that is what R1 is for. Written for a reader new to the project.

## The question

Around the INT8 multiply (K4) sit two kinds of memory-bound kernels. The **pack** (kernels KF / KL, K2, K3) reads the FP64 operand and writes N
narrow integer planes (one residue per modulus). The **CRT** (K5) reads the N planes of K4's result and writes the FP64 answer. At 4096³ with
N = 14 they take 1.2 ms of an H100 call (38%) and 61% at 1536³. On the H200, whose memory is 43% faster (datasheet), they take the same time within 1%,
so bandwidth is probably not what limits them there. Issue 14's arithmetic says K4 alone cannot reach parity at 4096³, so pack and CRT are on the
critical path.

## On the RTX 4090 they ARE bandwidth-bound (ncu, 2026-10-15)

`ncu --set full` on the first call of 4096³, N = 14 (`imd_bench`; ncu fixes the clock, so durations compare with each other):

| kernel | time | DRAM throughput | DRAM bytes | SM throughput | L2 hit |
|---|---|---|---|---|---|
| KL pack of A | 437 µs | 85% of peak (837 GB/s) | ~370 MB | 41% | 65% |
| KL pack of B | 437 µs | 85% | ~370 MB | 41% | 65% |
| K5 CRT | 429 µs | 87% (858 GB/s) | ~370 MB | 66% | 50% |
| K4 (for reference) | 3.19 ms | 23% | | 94% | 98% |

The bytes are exactly what the algorithm must move (operand 134 MB + 14 planes × 16.8 MB). So on the 4090 pack and CRT take 1.3 ms because the memory
system delivers 0.85–0.9 TB/s, and nothing in the kernels needs fixing there.

## What a Hopper number must be compared with

If the same kernels also ran at ~0.86 TB/s on an H100 (3.35 TB/s datasheet) they would use a quarter of the memory system: **the same time on three
cards with very different memories is not a coincidence of bandwidth**. The candidates are (a) the launch shape asks for too little data at once
(Little's law: bytes in flight per SM = bandwidth × latency / SMs; a kernel of 64 registers and 8 values per thread at the occupancy ncu reports may
not cover H100's latency), (b) a per-launch or per-tile fixed cost, (c) the instruction stream (SM throughput was 66% for the CRT already on the 4090),
(d) L2 effects (the K1 rows-in-flight budget below). The bandwidth microbenchmark separates (a) from the rest.

## `bench/membw/imd_membw.cu` (R1) and its analysis

Patterns shaped like the library's kernels, on buffers far larger than L2: `rd`, `wr`, `cp`, `fan` (read 16 B, write 2 B into each of 14 planes: the
pack) and `fanin` (read 2 B from each of 14 planes, write 16 B: the CRT), `fan16` / `fanin16` (the same bytes with 16-byte stores / loads per plane, the library's shape), swept over blocks per SM {1..16}, threads {128..1024}, loads in flight per
thread {1..8}; plus `rd_small`, a read of L2/4 … 8 × L2 (the L2 cliff). One JSON line per shape with GB/s and % of the datasheet-derived peak
(memory clock × bus width × 2). `tools/build_membw.sh OUT run` builds it with one nvcc call.

`bench/membw/analyze.py MEMBW.jsonl BENCH.jsonl` takes the pack / CRT phase times of `imd_bench` records, converts them to bandwidth with the bytes the phase
must move (beta = 0; KL's operand counted once, valid while K1's second pass hits L2, which the K1 budgets below are for; KF records are not judged), and compares with the
best this program's kernels deliver: the pattern's own shape, the library-shaped variant (`fan16` / `fanin16`: 16-byte stores / loads per plane), a plain copy, a pure read
and a pure write. **BANDWIDTH-BOUND** (≥ 85% of that best; only fewer bytes help), **SHAPE-BOUND** (< 85% and the best does ≥ 1.15× more: the launch shape leaves
bandwidth unused), SMALL (< max(32 MiB, 2 × L2): partly L2-resident, latency, not judged). The reference is made of this program's kernels: the percentages say
"relative to these kernels", not "relative to the hardware"; `pct_of_peak` (memory clock × bus width × 2, unreliable for GDDR6X / GDDR7) and `pct_of_rd` are printed beside it.

On the RTX 4090 (full sweep, 68 shapes per pattern, run 2026-10-19 on a box whose CPUs were busy with other jobs; raw lines in `immerdink-eval/results/20261019-membw-4090/membw.jsonl`):
rd 956 GB/s (94.8% of the 1008 GB/s peak), wr 947, cp 907, fan 821, fan16 887, fanin 880, fanin16 929; the launch shape matters by up to 30% (fanin 631–880).
Narrow 2-byte accesses lose 7–8% against 16-byte ones at the same traffic. The library's phases at 4096³ (earlier run): KL A 867 GB/s, KL B 876, K5 922, i.e. 91–96% of the best
read (956) and 98–99% of the library-shaped kernels (887, 929): **bandwidth-bound** (ncu agrees: 85–87% of the DRAM peak). Writes are not flushed at the end of a timed kernel
(up to one L2 of dirty data is left), the clocks are not locked, and the library phases have the same biases.
L2-resident reads run at 3.0–4.3 TB/s up to 72 MiB and fall to 0.94 TB/s at 144 MiB (8 × L2: 0.95).

## K1's L2 budgets are now derived from the card's L2

K1 (the scaling kernel) reads a row twice; pass 2 hits L2 only while the rows in flight fit. Two byte budgets decide how many rows it keeps in flight, and
both were constants for the 4090: 40 MiB (the grid cap of a two-pass kernel) and 64 MiB (the long-row variants; "the L2 less what the rest of the GPU keeps there").
They are now **5/9 and 8/9 of the device's L2** (`host/analysis.hpp` `k1_l2_fit_bytes`, `k1_l2_budget_bytes`; the plan reads the L2 size with the device class of issue 18):

| card | L2 | fit (was 40) | budget (was 64) |
|---|---|---|---|
| RTX 4090 | 72 MiB | 40.0 MiB | 64.0 MiB, exactly the old values (`test_l2_knobs`), so every 4090 launch is unchanged |
| H100 | 50 MiB | 27.8 | 44.4 |
| H200 | 50 MiB (the H100's die; not read from a card yet) | 27.8 | 44.4 |
| RTX PRO 6000 | 128 MiB | 71.1 | 113.8 |

The change affects only the schedule of K1, never its result (S and γ are exact and order-independent), so output hashes are unchanged by construction. The 8/9 and 5/9
are the 4090's ratios, not fitted for the other cards: R2's grid tests them (below). The other launch settings are already derived from the SM count
(`kKfSmallBlocksPerSm = 16` per SM, K5's one-element-per-SM rule) or from the shape (KL's tile).

## The target X

"H100 pack plus CRT at sq4096, t_df, r_lowk and s_cc16 is at most X ms": X per case is the time left once K4 is as fast as the design reaches:
X = T_target − N·2mnk / R_K4, with T_target the time at which immerdink matches GEMMul8 fast (61.6 TFLOPS-eq at 4096³: 2.23 ms) and R_K4 the K4 rate the wgmma
decision of issue 16 projects (`bench/campaign/required_rate.py`). With the 1.5× faster K4 of the issue text this gives about 0.9 ms at 4096³ (1.2 ms now). It is written
here as a formula because R_K4 comes from issue 16's verdict; R2 fixes the number before its grid runs.

## The CRT with the sums in binary64 limbs (implemented, off by default, not yet measured on Hopper)

Design direction 1 above, built and checked offline.

* **Algorithm** (`spec/crt_f64.py`, the exact model; `spec/params.py` certificate `cert_crt_f64` and the generated table `kF64Crt`; device code `crt_f64_elem` in `src/contract.cu`). The reference adds `M_l y_l` to a running word with a conditional subtraction of P at every one of the N
  steps. Here `K_l = q_l M_l < P` is cut into J limbs of 40 bits (J = 3 up to N = 15, 4 up to N = 20), `S = sum_l K_l W_l` is the sum of the residues times these constants, congruent to the reference's value modulo P (`K_l W_l - M_l ((q_l W_l) mod p_l)` is a multiple of P),
  and the J limb sums `s_j = sum_l W_l k_lj` are formed by one fused multiply-add per limb per term: they are integers below 2^53 whatever the order (every term is nonnegative), so they are exact. The quotient `t = floor(fl(S) / P - 2^-30)` comes from the Horner sum of the limbs and one fused
  multiply-add with 1/P; it is never above the true quotient and at most one below (the error is below 2^-37, the bias 2^-30 absorbs it; without the bias the estimate overshoots, 13,248 cases found). `r_j = s_j - t P_j` is exact, the limbs are carried into 32-bit words and one conditional subtraction of P leaves the same `x` in [0, P) as the
  reference; `crt_finish` and `round_scaled` follow unchanged, so the result is bit-identical by construction.
* **Checks.** The model equals `oracle.crt` on 63,841 residue vectors (every N from 2 to 20, the boundaries of the range, random points); the certificate asserts every bound for every N; `crt_f64_gpu` runs the library with each CRT and compares the output hashes (the 13 small and mid committed cases and every
  N from 2 to 20 in both input families, RN and TRUNC, batched, alpha and beta): identical. A gross error (e.g. a bias of +0.5) fails every case, but the test cannot reach the edge the bias exists for (z0 within 2^-30 P of 0 or P happens with probability about 2^-30 per element in a GEMM) and it cannot see errors in the low 50 bits of the 128-bit value (the final double hides them): the edge vectors are covered only by the CPU model (`spec/test_crt_f64.py`, which runs the shipped 40-bit width and compares the generated tables with the model), not by the device code. **Open (before adoption): a device-level unit hook running `crt_f64_elem` on caller-supplied residues against the model's edge vectors, and forced E = 1, 2, 4 and bfast variants for a few N.** The kernel is chosen by a benchmark and test hook only (`imd_test_set_k5_crt`, `imd_prof_set_k5_crt`, the duel side key `crt=int|f64`); the default is the integer CRT.
* **On the RTX 4090 it is slower** (2048³ N = 14: about 1.18 ms against 0.735 ms in one interleaved local duel on a shared machine; no record was kept, so treat it as indicative; FP64 runs at 1/64 of the FP32 rate there) and exact. It is meant for devices whose FP64 pipe matches the integer pipe (H100, H200).
* **What it saves, from the SASS of the sm_90 build.** The summing part of the CRT is about 365 of the roughly 650 dynamic thread-instructions per output element of `k5_crt_n<14,4>` (14 terms of about 26 integer instructions); the rest is `crt_finish`, `round_scaled`, flag checks and the store. With the limbs the summing part is about 120
  (14 byte extractions and conversions, 42 DFMA on the FP64 pipe, the reduction), so K5 falls from about 650 to about 400 thread-instructions per element, i.e. from 0.46 ms to about 0.28 ms on the H100 at the same issue utilization (derived, not measured), 5% of the call. Static counts of `k5_crt_n<14,4>` for sm_90: 3832 instructions with the integer CRT, 2848 with the limbs (counted from `cuobjdump -sass`; the dynamic figures are hand estimates from the loop structure). The 5% is conditional on K5 being issue-bound (an inference from R1) and on a DFMA costing what an integer instruction costs; a fair range is 0 to 8% of the call at 4096³ (about 3.4% at 8192³).
  **The next target is the finish** (`round_scaled`: the top-limb scan, the 64-bit extraction with runtime shifts, the rounding and the exponent logic, about 285 dynamic instructions per element), and the pack (direction 2).
* **To measure on Hopper (R2)**: `--methods duel --duel-a default --duel-b crt=f64` on sq4096, sq8192 and the other cases; adopt for a device class if the call improves by at least 3% on two cases and no case is slower by more than 2%, with the output hashes unchanged.

## R1 and R2

* **R1**: `membw.main` (the microbenchmark, about 1 minute per card) and the phase marks already recorded by every sweep. `analyze.py` names the limiter per card.
* **R2** (grid): the settings the analysis points at, on sq4096, sq8192, t_df, r_lowk and three small shapes; adopt a setting if the call time improves ≥ 3% on 2 of the 4
  named cases and none is more than 2% slower. If R1 does not name the limiter, a repair needs the spare run R4 (issue 14).
* The emulation-versus-native crossover per card for issue 18 is in `docs/SUPPORT.md`.

## R1 result (2026-10-19): on Hopper the pack and CRT are limited by instruction issue, not by memory

**What R1 measured.** On H100 and H200 the library's pack (KL) and CRT (K5) phases run at 0.8 to 1.1 TB/s, while the microbenchmark's best for the same traffic is 2.5 to 2.9 TB/s on H100
(3.2 to 3.8 on H200): 25 to 34% of the best on H100, 18 to 25% on H200. H100 and H200 take the same time although the H200 has 43% more bandwidth. A simple fan-out kernel with 16-byte stores per plane, one block of 256 threads per SM
and one load in flight (a shape the pack could use) reaches 2767 GB/s on H100, so the launch shape and the memory-level parallelism of that pattern do not explain it. On the RTX PRO 6000 the phases run at 76 to 89% of the peak:
bandwidth-bound there. (Candidates (a) launch shape and (d) L2 effects of the list above are out; (b) per-launch cost is unlikely: 8192³, with four times the bytes per launch, reaches only 10% more bandwidth than 4096³ (1078 against 977 GB/s for KL A on H100).)

**What limits them: the number of instructions.** There are no Hopper counters (the PACE environment has no `ncu`), so this is derived. `ncu` on the RTX 4090 (`immerdink-eval/results/20261019-ncu-pack-4090/`, sq4096 N = 14) gives
the executed warp instructions of the two kernels, which are the same program on Hopper (static SASS of the sm_90 build: 6192 instructions for `kl_scale_pack<8,2>` against 5920 for sm_89, 3832 against 3744 for `k5_crt_n<14,4>`):

| kernel (sq4096, N = 14) | warp instructions | thread instructions per output element | 4090: time, issue active, ALU pipe | H100: instructions / (528 issue slots per clock at 1.98 GHz) | H100 measured | implied H100 issue utilization |
|---|---|---|---|---|---|---|
| KL A (pack) | 230.5 M | 417 | 0.405 ms, 45%, 40% (DRAM-bound at 85%) | 0.220 ms | 0.378 ms | 58% |
| K5 (CRT) | 346.2 M | 650 | 0.397 ms, 70%, 72% | 0.331 ms | 0.459 ms | 72% |

The CRT kernel is not DRAM-bound on the 4090 (it issues 70% of the time and keeps the ALU pipe 72% busy), and on the H100 it reaches 72% of the issue rate. The 4090 clocks 2.7 GHz and has 128 SMs, the H100 clocks 1.98 GHz and has 132, with the
same 4 issue slots and 16 integer lanes per SM sub-partition: the same instruction stream is about 1.5 times slower per SM on the H100 and cannot be faster because of the wider memory. That is what R1 shows (the same 1.2 ms of pack and CRT).
The sq4096 pack takes 30 integer instructions per (element, modulus) and the CRT 46 per (element, modulus term).

**Consequence.** Neither wider loads nor more bytes in flight will speed them up on Hopper; fewer instructions will. Design directions, none of them implemented or measured, each needing an exactness argument checked offline first
(the existing oracle and exhaustive tests are the gate):

1. **CRT in FP64 limbs on devices where FP64 is fast.** The H100 FP64 pipe issues 64 lanes per SM per clock, as many as the integer pipe (the 4090 has 1/64 of that). A residue r (|r| at most 128) times a 26-bit limb of the constant of its modulus
   is below 2^34, and 14 such terms sum below 2^38 in a double exactly; three limbs per term replace the multi-word integer multiply-add, and the carries and the final reduction modulo the product of the moduli run once per element instead of once
   per term. It would be a device-class choice (table sets from issue 18 already select per card).
2. **Residue emission**: the 30 instructions per (element, modulus) are `dp4a` on the words of X, one `umulhi` for the quotient and the symmetric-range correction; measuring where the other ~20 go (the SASS shows 356 branches and 159 BSSY/BSYNC pairs in
   `kl_scale_pack<8,2>`, which a branch-free emitter should not have) is the first step.
3. Do both in one pass with the FP64 pipe doing part of the residues, so that the integer and FP64 pipes issue together.

Expected size if the instruction count halves at the same issue utilization: pack and CRT of sq4096 on H100 from 1.22 ms to about 0.65 ms (call 3.2 to 2.6 ms); with a wgmma K4 at 1.4 POPS the call would pass GEMMul8's 2.23 ms.

## Not done here

The instruction-issue explanation is derived from 4090 counters and public H100 figures, not measured on Hopper: a Hopper `ncu` run (a module on PACE, if one exists) would confirm or refute it. None of the three directions is implemented.

## Pack emitter and finish (E1, Q1, F1: leaner instruction sequences, one set per device class)

Source: the offline study `immerdink-eval/findings/offline-theory-20261025.md` section 5 (models `analysis/20261025-offline/pack/`: `emit_model.py` E1, `quantize_model.py` Q1, `finish_model.py` F1; each asserts bit identity with `spec/oracle.py`, all three pass unchanged). Nothing here changes a result: the sequences compute the same bytes and the same doubles, and the baseline sequence of each stays in the code.

### Step 0: what nvcc 13.4 emits (micro-check, `nvcc -cubin`, `cuobjdump -sass`)

| question | sm_89 | sm_90 | sm_120 |
|---|---|---|---|
| `hi32(s * m32 + h)` with a 64-bit addend (`(uint64_t(s) * m32 + hm) >> 32`) | one `IMAD.HI.U32 Rd, Ra, Rb, Rc` where Rc is a **register pair** (the 64-bit addend); not `IMAD.WIDE.U32` | same | same |
| `(double)uint64_t` | `I2F.F64.U64` (one native instruction) | same | same |
| `(uint64_t)double`, `__double2ull_rn` / `__double2ull_rz` | `F2I.U64.F64` / `F2I.U64.F64.TRUNC` (native) | same | same |

So the study's "IMAD.WIDE.U32 with a 64-bit addend" is `IMAD.HI.U32` with a pair; the instruction count is what the study assumed. Native does not mean full rate: the conversions and DMUL share the FP64 / XU pipes, which are 1/64 rate on GeForce-class parts (RTX 4090, sm_120 workstation cards) and full width on H100. That is why Q1 and F1 are Hopper-only defaults (below).

### Selection: one set of sequences per device class

* `src/host/seq_config.hpp` (CUDA-free): the bits `kSeqEmit` (E1), `kSeqQuant` (Q1), `kSeqFinish` (F1) and `kSeqClasses`, the default set per compute capability. `imd_plan_create` stores the device class's set in `ExecPlan::seq` (in the padding after `trunc`; `sizeof(ExecPlan)` is unchanged); the pack launches (`launch_kl`, `launch_kf`, K2/K3) and `launch_k5` read it. A class that is not listed, or not counted, gets 0 = the baseline. The hooks `imd_test_set_seq(bits)` / `imd_prof_set_seq(bits)` (and the harness env `IMD_TEST_SEQ`, the duel key `seq=default|0..7`) force a set for every plan; -1 restores the class's.
* The kernels take the set as a template value (`kl_scale_pack<E, W, Seq>`, `kf_scale_pack<Cap, E, W, Seq>`, `k23_pack<W, Seq>`: 0 baseline, 1 E1, 3 E1 + Q1 (W = 2 only); `k5_crt_n<N, E, F64, F1>`, `k5_crt_bfast<N, F64, F1>`), so a set costs no instruction in the other sets. The baseline instances are the original code: their SASS counts equal the pre-change counts to within 2 instructions (table below). The occupancy rows of the performance model (`imd_prof_kernel_occupancy`) describe the baseline instances; the E1 / Q1 instances have different register counts (not modelled).
* Shared math in one place each: E1 `device/residue.hpp` (`PackEmit`, `kPackEmit`, `pack_emit_bounds_ok`) and `device/modp.hpp` (`pack_dot`, `pack_smod`; CPU and GPU run the same code); Q1 `kernels/numerics.cuh` (`quant_scale`, `quantize_cvt`, with the sigma bound as a `static_assert` against the certified `kSigmaMin`..`kSigmaMax`); F1 `kernels/numerics.cuh` (`finish_f1_try`, `crt_round`; its fallback is the existing `crt_finish` + `round_scaled`). A walk-back to a more general version is a change of `kSeqClasses` (or of one function); nothing else refers to the new sequences.

### Hardware-specific assumptions (what must be re-checked for a new class)

| sequence | math | assumes about the hardware |
|---|---|---|
| E1 | `d = sum dp4a`, `s' = sigma d + C` (C a multiple of p, `>= dmax`), `q = hi32(s' m32 + h m32)`, `u = s' - p q = SMod(sigma d, p)`; exact because `s' + h < 2^24` (A5) | the compiler fuses the 64-bit addend into `IMAD.HI.U32` (step 0); `dp4a` is native (sm_61+). Integer pipe only. |
| Q1 | `X = F2I.U64.{RNI,RZI}(\|x\| 2^s1 2^s2)`, `s1 = sigma >> 1`, `s2 = sigma - s1`; `X := 0` if `x` is Inf or NaN. W = 2 only (`X < 2^63`) | native `F2I.U64.F64` (step 0), DMUL at the integer rate. **1/64 FP64 rate on the 4090 and sm_120 cards: off there.** |
| F1 | `D = RN(t) 2^kk` from the top 64 bits plus a sticky bit; scaling is an exponent-field add | native `I2F.F64.U64`; same FP64 / XU rate caveat. Off on the 4090 and sm_120. |

### Defaults and the SASS evidence per class

Counts are `cuobjdump -sass` of `src/contract.cu` built for each architecture (`tools/sass_seq_counts.sh`; static instruction counts, NOPs excluded) and, for sm_89 only, executed thread-instructions from `ncu` (`smsp__thread_inst_executed.sum`, 4096^3 N = 14, RN, family u: deterministic counts, not timings; no Hopper counter run is possible here).

Modulus loop of the KL pack (instructions per iteration; the iteration emits 8 or 4 elements of one modulus):

| kernel | sm_89 base / E1 | sm_90 base / E1 | sm_120 base / E1 |
|---|---|---|---|
| `kl_scale_pack<8,2>` (8 elements) | 99 / 63 | 98 / 63 | 103 / 74 |
| `kl_scale_pack<8,3>` | 109 / 74 | 110 / 72 | 112 / not measurable (the loop is replicated) |
| `kl_scale_pack<4,2>` (4 elements) | 106 / 68 | 106 / 68 | 57 / 135 for 4 iterations (34 per iteration; unrolled) |
| `kf_scale_pack<8,16,2>` (16 elements) | 220 / 106 | 220 / 106 | 203 / 149 |
| `k23_pack<2>` (16 elements) | 207 / 107 | 207 / 107 | 202 / 151 |

Whole-kernel static counts (the loop is a small part of them, so these move little): E1 is lower on sm_89 in all 10 pack kernels counted (the K2/K3, KF E = 16 and 4, KL E = 8 and 4 kernels, W = 2 and 3; -20 to -274), on sm_90 in 9 of 10 (the exception is `kf_scale_pack<8,16,2>`: 6799 -> 6827, +28 while its modulus loop falls 220 -> 106), and on sm_120 in 8 of 10: `kl_scale_pack<4,2>` 4082 -> 4510 and `<4,3>` 4220 -> 4730 grow because the compiler unrolls the modulus loop there (per iteration it is shorter), and `kf_scale_pack<8,4,3>` is +57. Executed thread-instructions per KL element on sm_89 (ncu): baseline 417.8, E1 356.3 (-61.5; predicted -72), E1 + Q1 311.4 (a further -44.9; predicted -45).

| class | E1 | Q1 | F1 | reason |
|---|---|---|---|---|
| sm_89 (RTX 4090) | on | off | off | E1: fewer instructions everywhere, integer only. Q1, F1: FP64 at 1/64 rate. |
| sm_90 (H100, H200) | on | on | (F1 follows in the next commit) | fewer instructions in every pack kernel; FP64 pipe as wide as the integer pipe. **Not timed on Hopper**: the XU throughput of `F2I.U64` / `I2F.F64.U64` is unverified, R2 duels `seq=0` against `seq=default` decide whether it stays. |
| sm_120, anything else | off | off | off | counted (table above): E1's loop is shorter per iteration but two kernels grow because of unrolling and nothing was timed; a class without its own row in the K4 registry stays on the baseline. |

### Tests

* `pack_emitter` (GPU unit test, `tests/test_pack_emitter.cu`): every N = 2..20, E = 16, 8, 4, RN and TRUNC, now through the baseline, the E1 and (W = 2) the E1 + Q1 instances, against an exact host reference (148 M residues), with a new input mode that covers the whole certified sigma range (`kSigmaMin`..`kSigmaMax`: Q1's two factors, subnormals, the largest double); **every reduced value** `d` in `[0, dmax]` for both signs, all 20 moduli and both word counts through `pack_smod` on the real `__constant__` table (11,442,942 values); eight **mutations** of that table (`hm - m32`, `m32 - 1`, `C - p`, `negp + 1`, a weight lane, `C + 1`, `hm + m32` at p = 256, a W = 2 weight lane), each of which must be caught.
* `contract_{e2e,tensor,every_n,eps,bigk}_seq{1,7}`, `output_hashes_seq{1,7}`, `output_hashes_large_seq7`: the existing oracle / committed-hash suites run with the other sequences forced for every plan (`IMD_TEST_SEQ`). The committed hashes are unchanged.
* The Python models (`nice -n 19 python3 -I`): `emit_model.py` (11.4 M + 0.97 M cases), `quantize_model.py` (787 k), `finish_model.py` (457 k): pass.
