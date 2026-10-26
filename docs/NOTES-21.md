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

## The CRT with the sums in binary64 limbs (the sm_90 default since R2; off on every other class)

Design direction 1 above, built and checked offline.

* **Algorithm** (`spec/crt_f64.py`, the exact model; `spec/params.py` certificate `cert_crt_f64` and the generated table `kF64Crt`; device code `crt_f64_elem` in `src/contract.cu`). The reference adds `M_l y_l` to a running word with a conditional subtraction of P at every one of the N
  steps. Here `K_l = q_l M_l < P` is cut into J limbs of 40 bits (J = 3 up to N = 15, 4 up to N = 20), `S = sum_l K_l W_l` is the sum of the residues times these constants, congruent to the reference's value modulo P (`K_l W_l - M_l ((q_l W_l) mod p_l)` is a multiple of P),
  and the J limb sums `s_j = sum_l W_l k_lj` are formed by one fused multiply-add per limb per term: they are integers below 2^53 whatever the order (every term is nonnegative), so they are exact. The quotient `t = floor(fl(S) / P - 2^-30)` comes from the Horner sum of the limbs and one fused
  multiply-add with 1/P; it is never above the true quotient and at most one below (the error is below 2^-37, the bias 2^-30 absorbs it; without the bias the estimate overshoots, 13,248 cases found). `r_j = s_j - t P_j` is exact, the limbs are carried into 32-bit words and one conditional subtraction of P leaves the same `x` in [0, P) as the
  reference; `crt_finish` and `round_scaled` follow unchanged, so the result is bit-identical by construction.
* **Checks.** The model equals `oracle.crt` on 63,841 residue vectors (every N from 2 to 20, the boundaries of the range, random points); the certificate asserts every bound for every N; `crt_f64_gpu` runs the library with each CRT and compares the output hashes (the 13 small and mid committed cases and every
  N from 2 to 20 in both input families, RN and TRUNC, batched, alpha and beta): identical. A gross error (e.g. a bias of +0.5) fails every case, but the test cannot reach the edge the bias exists for (z0 within 2^-30 P of 0 or P happens with probability about 2^-30 per element in a GEMM) and it cannot see errors in the low 50 bits of the 128-bit value (the final double hides them): the GEMM test does not reach them; the CPU model (`spec/test_crt_f64.py`, which runs the shipped 40-bit width and compares the generated tables with the model) and the device-level unit test below do.
  **Device-level unit test** (`tests/test_crt_f64_unit.py`, ctest `crt_f64_unit`; hook `imd_test_crt_elem`, compiled only with `IMMERDINK_BUILD_TESTS` through the definition `IMD_BUILD_TEST_HOOKS`). A tiny kernel (`test_crt_elem_kernel` in `src/contract.cu`) runs the REAL `crt_f64_elem<N, L>` and the REAL integer running CRT (`crt_planes_n`) of the same file on caller-supplied residue vectors, for every N from 2 to 20 and each byte position e = 0..3 of the plane word (the other three bytes hold unrelated data), and returns the integer x in [0, P) as 32-bit words before `crt_finish` and `round_scaled`, so every bit of the 128-bit value is compared (with `z`, the CRT of the residues, which is `oracle.crt` before centring and `spec/crt_f64.py`'s value; the oracle and the model are also evaluated on a sample of the very vectors). Vectors per N: the edge vectors of `spec/test_crt_f64.py` (z in {0, 1, 2, 3, 255, 256, 65535, H-1, H, H+1, P-1, P-2, P-3, P-256, P/2, P/3, P/255}, the 300 smallest and 300 largest z, W_l = p_l - 1), z = P 2^-b and P - 1 - P 2^-b with offsets -2..2 for b = 1..46 (b = 30 is the bias edge; z0 below P 2^-30 is the t = -1 situation when S < P), random z of every bit length near both ends, random z within a few multiples of P 2^-30 of both ends, and uniform random z. The test counts the vectors that have t = -1 and those for which the estimate is one low (R >= P before the last subtraction) and prints them.
  **Mutation check.** `crt_f64_elem` has a compile-time knob `Mut` (default 0; no other value is instantiated outside test builds; the SASS of every shipped kernel in `contract.cu`, including all 76 `k5_crt_*` kernels with the limb CRT, is identical before and after, compared instruction by instruction with `cuobjdump -sass` on an sm_89 build). The test instantiates eight defects: bias 0, bias 2^-40, bias 2^-46, a wrong low limb constant (`k[N][N-1][0] + 1`), a wrong `Pj[0]` (+1), the bottom limb's carry dropped, the bottom limb's carry taken by a logical shift (wrong for negative `r_0`), and R == P not reduced. Each mutated device result must equal a bit-exact Python model of the mutated arithmetic on every vector (so the mutation does what it says), and must differ from z on some vector at every N where it is a defect; where it is provably none (`benign()` in the test: one limb, a word of at most 64 bits, `r_0` never negative, the estimate's error bound below the bias plus 1/P) it is reported as such. Result on the RTX 4090 build: bias 0 and bias 2^-46 are caught at N = 6..20 (no defect up to N = 5, where 1/P exceeds the estimate's error); the low limb constant, `Pj[0]` and R == P at every N from 2 to 20; the dropped carry at N = 6..20 (no defect with one limb, N up to 5); the logical shift at N = 8..20 (no defect with a word of at most 64 bits, N up to 7, nor at N = 11, where `r_0 >= 0` for every residue vector, shown exactly). **Bias 2^-40 is not caught at any N and is not a shown defect:** the proved bound on the estimate's error (2^-37) does not exclude it, but the observed overshoot with no bias at all is at most about 2^-43 in every search (the quotient is below 2^11 and its ulp 2^-42), so 2^-40 works on all vectors found; the test reports it and does not require it. The shipped 2^-30 has a margin of 2^13 over the largest overshoot seen. Differential search on the device (100,000 vectors per family per N: z within 2^-s P of P for s = 30..46, sparse residue vectors, uniform residues) adds the vectors on which a mutation differs from the shipped code to the set; the vectors that maximize and minimize each limb remainder are added too. About 3.5 million device evaluations run in half a minute.
  **What it can and cannot see.** It sees any wrong bit of x for these vectors, including the low bits, the t = -1 and R >= P cases, the byte extraction of E = 4 and the limb carries. It does not see: the later stages (`crt_finish`, `round_scaled`, store; the GEMM tests do), a defect that appears only for vectors off the edge lists and the random draws (about 4,000 per N and byte position; the CPU model covers 63,841 and the argument is `cert_crt_f64`), the register allocation, scheduling or inlining of the real K5 kernels (the test kernel is a different kernel around the same function), and the other devices (it runs on the build's architectures only; run it on sm_90 on the cluster).
  **K5 variants.** `crt_f64_gpu` also forces E = 1, 2, 4 (`imd_test_set_k5_elems`) and the batch-fastest kernel for N = 5, 8, 12, 16, 20 (shapes with tails), and checks with a launch counter (`imd_test_k5_launch_counts`, recorded by `launch_k5` in test builds) that exactly one K5 launch of the intended variant and CRT ran, and that the output hashes of the two CRTs are equal. The kernel is chosen by the plan's class set (the bit `kSeqLimbCrt` = 8 of `kSeqClasses`: set on sm_90 only, see "R2 outcome and the sm_90 default") unless the benchmark and test hook forces it (`imd_test_set_k5_crt`, `imd_prof_set_k5_crt`, the duel side key and `--crt` with `crt=default|int|f64`; the hook is tri-state: -1 follow the set (the process default), 0 integer, 1 limbs). Before R2 the integer CRT was the default everywhere.
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

So the study's "IMAD.WIDE.U32 with a 64-bit addend" is `IMAD.HI.U32` with a pair; the instruction count is what the study assumed. Native does not mean full rate: the conversions and DMUL share the FP64 / XU pipes, which are 1/64 rate on GeForce-class parts (RTX 4090, sm_120 workstation cards) and full width on H100. On H100 the DMUL is full rate but the 64-bit conversions (`F2I.U64`, `I2F.F64.U64`) are probably a quarter of the integer rate (unmeasured). That is why Q1 and F1 are opt-in everywhere for now (below).

### Selection: one set of sequences per device class

* `src/host/seq_config.hpp` (CUDA-free): the bits `kSeqEmit` (E1), `kSeqQuant` (Q1), `kSeqFinish` (F1), `kSeqLimbCrt` (the binary64-limb CRT of K5, a class-default-only bit: the hooks force the sets 0..7, which carry none and so run the integer CRT unless the K5 hook forces the limb CRT) and `kSeqClasses`, the default set per compute capability (`seq_default_for` masks it to what the build can run: with `IMMERDINK_SEQ_TWINS=OFF` the F1 bit is dropped, the limb CRT is not a twin and stays). `imd_plan_create` stores the device class's set in `ExecPlan::seq` (in the padding after `trunc`; `sizeof(ExecPlan)` is unchanged); the pack launches (`launch_kl`, `launch_kf`, K2/K3) and `launch_k5` read it. A class that is not listed, or not counted, gets 0 = the baseline. The hooks `imd_test_set_seq(bits)` / `imd_prof_set_seq(bits)` (and the harness env `IMD_TEST_SEQ`, the duel key `seq=default|0..7`) force a set for every plan; -1 restores the class's.
* The kernels take the set as a template value (`kl_scale_pack<E, W, Seq>`, `kf_scale_pack<Cap, E, W, Seq>`, `k23_pack<W, Seq>`: 0 baseline, 1 E1, 3 E1 + Q1 (W = 2 only); `k5_crt_n<N, E, F64, F1>`, `k5_crt_bfast<N, F64, F1>`), so a set costs no instruction in the other sets. The baseline instances are the original source: the 14 pack kernels (KL, KF, K2/K3, both word counts) have the same SASS count as before to within 2 instructions on sm_89, sm_90 and sm_120. The K5 baseline instances (`k5_crt_n<N,E,F64>`, `k5_crt_bfast`) have identical source too, but their counts differ from the pre-change build by -27 to +33 instructions (45 to 69 of the 152 instances differ by more than 2, depending on the architecture): instruction-scheduling noise from the changed template parameter list, not a different sequence. The occupancy rows of the performance model (`imd_prof_kernel_occupancy`) describe the baseline instances; the E1 / Q1 instances have different register counts (not modelled).
* Shared math in one place each: E1 `device/residue.hpp` (`PackEmit`, `kPackEmit`, `pack_emit_bounds_ok`) and `device/modp.hpp` (`pack_dot`, `pack_smod`; CPU and GPU run the same code); Q1 `kernels/numerics.cuh` (`quant_scale`, `quantize_cvt`, with the sigma bound as a `static_assert` against the certified `kSigmaMin`..`kSigmaMax`); F1 `kernels/numerics.cuh` (`finish_f1_try`, `crt_round`; its fallback is the existing `crt_finish` + `round_scaled`). A walk-back to a more general version is a change of `kSeqClasses` (or of one function); nothing else refers to the new sequences.

### Hardware-specific assumptions (what must be re-checked for a new class)

| sequence | math | assumes about the hardware |
|---|---|---|
| E1 | `d = sum dp4a`, `s' = sigma d + C` (C a multiple of p, `>= dmax`), `q = hi32(s' m32 + h m32)`, `u = s' - p q = SMod(sigma d, p)`; exact because `s' + h < 2^24` (A5) | `dp4a` is native (sm_61+). Integer pipe only. **Compiler dependence**: that the 64-bit addend fuses into one `IMAD.HI.U32` was measured with nvcc 13.4 only; the PACE modules use CUDA 12.9, whose ptxas may emit the longer `IMAD.WIDE` + `IADD3.X` form (a few instructions more per element-modulus; re-run `tools/sass_seq_counts.sh` with that toolchain before trusting the counts). **Registers**: KL is at 64 registers (the launch bound) with or without E1; `k23_pack<2>` is 64 baseline, 80 with E1 left alone (3 CTAs/SM instead of 4), so the E1 instance is held to 64 by its own launch bounds (`k23_pack_cap`, no spills). |
| Q1 | `X = F2I.U64.{RNI,RZI}(\|x\| 2^s1 2^s2)`, `s1 = sigma >> 1`, `s2 = sigma - s1`; `X := 0` if `x` is Inf or NaN. W = 2 only (`X < 2^63`) | native `F2I.U64.F64` (step 0), 2 DMUL. 1/64 FP64 rate on the 4090 and sm_120 cards; on H100 DMUL is full rate but the 64-bit conversion is probably quarter rate (unmeasured). **Registers**: `k23_pack<2,3>` takes 94 registers (alloc 96) against 64, so 2 CTAs per SM instead of 4 on sm_90; capping it spills 128 B, so it is not capped. A weak-FP64 sm_90-class part (H20) would suffer from Q1 and F1 if they were defaults; the class table is keyed by compute capability only, which is one more reason they are not. |
| F1 | `D = RN(t) 2^kk` from the top 64 bits plus a sticky bit; scaling is an exponent-field add | native `I2F.F64.U64`, same rate caveat. **Divergence**: a warp pays the fast path *and* the fallback if any lane has mag = 0, a subnormal or an overflowing result (zero outputs are common in sparse data), so F1 is worth less on such data than its count says. **Code size**: both paths are in the kernel (K5 static counts grow by ~200 per instance); the instruction cache is not modelled. |

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

Whole-kernel static counts (the loop is a small part of them, so these move little): E1 is lower on sm_89 in all 10 pack kernels counted (the K2/K3, KF E = 16 and 4, KL E = 8 and 4 kernels, W = 2 and 3; -20 to -274), on sm_90 in 9 of 10 (the exception is `kf_scale_pack<8,16,2>`: 6799 -> 6827, +28 while its modulus loop falls 220 -> 106), and on sm_120 in 6 of 10: `kl_scale_pack<4,2>` 4082 -> 4510 and `<4,3>` 4220 -> 4730 grow because the compiler unrolls the modulus loop there (per iteration it is shorter), `kf_scale_pack<8,4,3>` is +57 and `k23_pack<2>` (capped, see below) is +48. The sm_89 range is -20 to -274 (`kl_scale_pack<4,2>`: -274 against the pre-change count, -272 against the same-source baseline instance; `k23_pack<2>` is -32 with the cap, -82 without). Executed thread-instructions per KL element on sm_89 (ncu): baseline 417.8, E1 356.3 (-61.5; predicted -72), E1 + Q1 311.4 (a further -44.9; predicted -45).

| class | E1 | Q1 | F1 | reason |
|---|---|---|---|---|
| sm_89 (RTX 4090) | on | off | off | E1: fewer instructions everywhere counted, integer only. Q1, F1: FP64 at 1/64 rate. |
| sm_90 (H100, H200) | on | off | **on (since R2; plus the limb CRT, see below)** | E1 as on sm_89 (the review's register finding handled by `k23_pack_cap`). Q1 and F1 have fewer instructions but cost registers (Q1: `k23_pack<2,3>` 94 vs 64, 2 CTAs/SM instead of 4) and use conversions that are probably quarter rate on Hopper: opt-in (`seq=`, `imd_prof_set_seq`, `IMD_TEST_SEQ`) until R2 measured them: R2 measured F1 as an adopt (and the limb CRT), Q1 as a tie (0.98x at g = 7), so on sm_90 F1 and the limb CRT became defaults and Q1 stayed opt-in (this paragraph described the pre-R2 table; the shipped row is `E1 + F1 + limb CRT = 13`). |
| sm_120, anything else | off | off | off | counted (above): E1's loop is shorter per iteration but 4 of 10 kernels grow (compiler unrolling, the capped `k23_pack<2>`) and nothing was timed; a class without its own row in the K4 registry stays on the baseline. |

Registers and spills (`ptxas -v`, same builds). sm_90: `k23_pack<2>` 64 baseline / 80 E1 uncapped / 64 `k23_pack_cap<2,1>` (no spills; 2597 instructions against 2680 baseline) / 94 E1 + Q1; `k23_pack<3>` 116 / 126 (2 CTAs per SM either way); `kf_scale_pack<8,16,2>` 128 / 126 / 112; every `kl_scale_pack` 64. sm_89: `k23_pack<2>` is 84 registers at baseline (3 CTAs/SM), so the capped E1 instance (64) is better than the baseline there. Spill bytes (stores/loads): `kl_scale_pack<4,3,1>` sm_90 28/36 (7 STL, 9 LDL) against 4/4 at baseline, sm_89 20/32 against 12/16. The SASS of the sm_90 instance puts every STL/LDL in the per-tile prologue (row set-up, `divmod`) and the chunk set-up, none inside the modulus loop (the IDP loop at 0xce20 has none), so they are executed once per tile and chunk, not per modulus; E1's `sg[E]` is not the cause (the baseline keeps `neg` as one word) but the extra live values around the setup are, and the 1024-thread KL<4> instance is at its 64-register cap.

### Tests

* `pack_emitter` (GPU unit test, `tests/test_pack_emitter.cu`): every N = 2..20, E = 16, 8, 4, RN and TRUNC, now through the baseline, the E1 and (W = 2) the E1 + Q1 instances, against an exact host reference (148 M residues), with a new input mode that covers the whole certified sigma range (`kSigmaMin`..`kSigmaMax`: Q1's two factors, subnormals, the largest double); **every reduced value** `d` in `[0, dmax]` for both signs, all 20 moduli and both word counts through `pack_smod` on the real `__constant__` table (11,442,942 values); ten **mutations** of that table (see below), each of which must be caught.
* `contract_{e2e,eps}_seq0`, `contract_{e2e,tensor,every_n,eps,bigk}_seq7`, `contract_{e2e,eps}_seq1`, `output_hashes_seq{0,1,7}`, `output_hashes_large_seq7`: the existing oracle / committed-hash suites run with the baseline (seq0: the production path on every class without a default set, and the reference now that sm_89 and sm_90 default to E1), E1 and the full set forced for every plan (`IMD_TEST_SEQ`; the harness raises if the variable is set and the library has no test hooks or does not report that set back through `imd_test_get_seq`). `seq_select` pins the per-class default against this GPU's compute capability, reads the effective set of a plan back (`imd_test_plan_seq`) and checks that the invalid sets (Q1 without E1, out of range) are rejected. The committed hashes are unchanged.
* **Mutants** (the compile-time `Mut` template value of the production templates, zero in every shipped instance; each must produce mismatches): E1 sign polarity (`sg`), the `byte_perm` assembly, `store_slots` not advancing to the start slot (the launches' addressing, now in `kernels/pack_emitter.cuh` and driven by the unit test in two parts), Q1 without the `f2` factor, Q1 with RN for TRUNC, Q1 without the Inf/NaN mask, Q1 `s2` off by one; two table mutants at p = 256 (`m32 - 1`, `C - p`, the low-byte path) beside the eight above; F1: `x == H` counted as negative, the 2-limb normalisation stage miscounted, the exponent taken without the `Dm = 2^64` carry, besides the five of `finish_f1`. Two review items are equivalent mutants and are not bugs: `sigma / 2` for `sigma >> 1` (any split with |s1|, |s2| <= 1022 is exact; the test asserts zero mismatches) and a missing 1-limb normalisation stage (the result falls back to the existing path).
* The Python models (`nice -n 19 python3 -I`): `emit_model.py` (11.4 M + 0.97 M cases), `quantize_model.py` (787 k), `finish_model.py` (457 k): pass.

### F1 (the CRT finish through `I2F.F64.U64`)

`kernels/numerics.cuh` `finish_f1_try<L, N>` / `crt_round<L, N, F1>` replace `crt_finish` + `round_scaled` in `k5_elems_n` for both CRTs (the integer one and the binary64-limb one, `k5_crt_override()`): both leave `x` in `[0, P)` as `L` limbs. The model's sequence, with the two borrow chains as PTX `sub.cc` / `subc` carry chains (one `IADD3` per limb; the C++ 64-bit borrow compiled to three instructions per limb), 4 / 2 / 1-limb normalisation stages, `clz`, funnel shifts, a sticky bit, `I2F.F64.U64`, and an add of `kk` to the exponent field of the high word. Mag = 0, subnormal results, overflow and an exponent outside `1 <= 1086 + kk <= 2045` fall back to the existing path (`fast` fraction in the unit test: 78% of an adversarial set, 100% of typical exponents; the model's 457 k cases agree).

* **Test** `finish_f1` (`tests/test_finish_f1.cu`): for every N = 2..20, 34,170 cases (649,230 in all) of x around 0, H, P - 1, all-ones, powers of two, exact ties and neighbours, random of every bit length, and exponents that put the leading bit at the subnormal and overflow edges, through the device `finish_f1_try` against `crt_finish` + `round_scaled` on the same limbs: 0 mismatches, fast and fallback both exercised for every N. Five **mutations** (`Mut` template value: sticky bit dropped, exponent +1, upper bound 2047, sign dropped, lower bound admitting field 0): all caught. `crt_f64_gpu` now also runs the limb CRT, F1, and both together (and F1 with all other sequences) against the integer CRT with the baseline finish, N = 2..20 in both families: identical to each other and to the committed hashes.
* **Counts.** Static counts of K5 *grow* with F1 (the fast path and the fallback are both in the kernel: `k5_crt_n<14,4,false>` 3821 -> 4030 on sm_90, 3728 -> 3943 on sm_89), so only executed counts say anything. sm_89, `ncu`, 4096^3 N = 14: `k5_crt_n<14,4>` 651.2 thread-instructions per output element with the baseline finish, 554.5 with F1 (-96.7; the study predicted 650 -> ~495, -155). The shortfall is the estimate of the old finish: the typical `round_scaled` path executes about 150 instructions, not 200. The fast path alone is about 50 instructions (`k_f1` probe: 75 / 69 / 79 static including a 25-instruction harness on sm_90 / sm_89 / sm_120; the reference `crt_finish + round_scaled` probe is 264 / 253 / 210 static with both branch sides). H100 prediction at the same issue utilisation: 97 of 651 thread-instructions per element is 15% of K5 (0.07 ms of 0.46 ms), about 2% of the call; derived, not measured.

### Predicted against achieved (executed thread-instructions per element, sm_89 `ncu`)

| step | study prediction | achieved |
|---|---|---|
| E1 (KL) | -72 | -61.5 (417.8 -> 356.3); loop 98 -> 63 per 8 elements (predicted 96 -> ~55) |
| Q1 (KL) | -45 | -44.9 (356.3 -> 311.4) |
| F1 (K5) | -155 (650 -> ~495) | -96.7 (651.2 -> 554.5) |

### Discrepancies and remaining risk

* The study names `IMAD.WIDE.U32` for the 64-bit addend; the compiler emits `IMAD.HI.U32` with a register pair (same count).
* The E1 loop keeps ~10 instructions of 64-bit pointer arithmetic per iteration (`dst += plane` is rematerialised from three kernel arguments because KL is at 64 registers); a precomputed `plane` in `LargeArgs` would remove about 8 of 63 (a quick win, not done).
* (Written before R2; superseded by "R2 outcome and the sm_90 default" below.) Hopper is not measured. E1 is the sm_90 default on instruction counts and registers; Q1 and F1 (fewer instructions, more registers, probably quarter-rate conversions) are opt-in until R2 duels them (`--duel-a default --duel-b seq=7`, or `seq=1` against `seq=3` and `seq=1` against `seq=5` to separate them). `imd_pack` (a pack handle has no device class) and the performance model's occupancy rows use the baseline sequences.
* A forced set that is not a set is rejected (`imd_test_set_seq`, `imd_prof_set_seq`, the duel key): the Q1 bit needs the E1 bit (`seq=2`, `seq=6` would silently have run the baseline). `quant_scale` asserts (debug builds) that sigma is inside the certified `kSigmaMin`..`kSigmaMax`, as the baseline clamps its shift. The `Mut` knobs of `finish_f1_try`, `quantize_cvt`, `quant_scale`, `PackEmitterT` and `store_slots` stay compile-time template values (every shipped instance is `Mut = 0` and contains none of them); moving them into test-only copies would duplicate the functions under test.
* Not done (as asked): L1, P1, K1, E2.

### Build cost and the twins

The kernel count of `src/contract.cu` per architecture goes from 178 to 351 and its static SASS from 398,720 to 831,846 instructions on sm_89 (416,469 -> 867,207 on sm_90, 422,666 -> 886,317 on sm_120): every pack kernel exists for the baseline, E1 and (W = 2) E1 + Q1, every K5 kernel for F1 on and off, besides the F64 pair that already existed. Since R2 sm_90 defaults to the F1 twins (and to the F64 K5 instances, which are not twins and always compiled); the Q1 twins are reachable by hooks only. With the option off the sm_90 class default is masked to E1 + the limb CRT (`seq_default_for`; `imd_test_plan_seq` reads 9), so an sm_90 production build that drops the twins loses F1 and keeps the limb CRT. `-DIMMERDINK_SEQ_TWINS=OFF` omits them (`IMD_NO_SEQ_TWINS`: the Q1 pack instances and the F1 K5 instances are not instantiated; the hooks then refuse any set but 0 and 1, the ctest `seq7` / F1 cases are not registered, `crt_f64_gpu` and `seq_select` drop them). With the option off the sm_90 object has 192 kernels and 490,853 instructions (checked by compiling `contract.cu` with `-DIMD_NO_SEQ_TWINS=1`; the rest of the library was not rebuilt that way). Default ON, so a normal build and the CI keep testing every sequence; a production build that does not want the twins turns it off. The F64 instances (issue 21, before this section) are not covered by the option.

### Local regression check (4090, not a measurement)

`imd_bench --methods duel --duel-a seq=0 --duel-b default --reps 200` (the class default on the 4090 is E1), interleaved, GPU lock held, load below 8, the desktop session active, so these numbers are **not decision-grade**; they only answer "is the default grossly slower?" (the stop rule was more than 3%). ratio_median = time(seq=0) / time(default), so above 1 means the default is faster:

| case | ratio_median | p10 .. p90 | note |
|---|---|---|---|
| sq4096, N = 14 (KL for both operands) | 0.9998 | 0.9988 .. 1.0011 | 4.89 ms either way |
| sq4096, N = 18 (W = 3) | 0.9998 | 0.9984 .. 1.0010 | 6.25 ms |
| sq2048, N = 8 | 1.017 | 1.010 .. 1.023 | |
| sq1024, N = 14 (KF) | 1.052 | 0.933 .. 1.113 | spread 10-12 %: noise |

Every pair is bit-identical (`bit_identical: true`). The 4090 pack is DRAM-bound at these sizes (85%), so no change in the call is expected from fewer instructions; none was found and none was lost.

### Running the normal `imd` method under a sequence set and CRT variant (campaign manifest)

`imd_bench --methods imd ... --seq <default|0|1|3|4|5|7> --crt <default|int|f64>` (profiling build, which is what `bench/build.sh` makes) forces the sequence set (`imd_prof_set_seq`) and K5's CRT variant (`imd_prof_set_k5_crt`) for that method's plan and calls only: both are set before the plan is created and reset (-1 and -1) as soon as the record's fields are read, with a scope guard for the error paths, so later methods in the same process run the defaults. `--seq 2`, `6`, anything else out of range and a `--crt` other than `default`/`int`/`f64` are refused before any GPU work (the Q1 bit needs the E1 bit, as in the hooks). Omitting both options changes nothing (neither hook is called, the record has `seq_req: -1`, `crt_req: "default"`; `--crt default` is that).

The imd record gets `seq_req` (-1 = the device class's set) and `crt_req` (what was asked), `seq` and `crt` (what the plan read back: `imd_prof_plan_seq`, `imd_prof_plan_crt`: the CRT variant the plan's K5 runs, whatever decided it), `phase_seq` and `phase_crt` (the same, beside `phases`, which are the per-phase times of that configuration), and `seq_after_reset` / `crt_after_reset` (the plan read back after the reset: the class's set and the class's CRT (`f64` on sm_90, `int` elsewhere), the guard that nothing leaks). The output hash is the record's existing `c_hash` (FNV-1a over C's bytes): all configurations of one case must give one `c_hash`. The variant string gets ` seq=N` / ` crt=f64` or ` crt=int` appended when forced. `bench/test_seq_crt.py` is the smoke test (usage text, refused values, sets default/0/1/7 x default/int/f64 on a tiny shape with one `c_hash` and the recorded fields, the reset); it is a ctest (`bench_seq_crt`, GPU lock) when `IMMERDINK_BENCH_BIN` names the profiling-build `imd_bench`.

## R2 outcome and the sm_90 default

R2 (2026-10-25, commit e00bb92; H100 and H200, Blackwell RTX PRO 6000; findings in `immerdink-eval/findings/r2-20261025.md`) measured the sequences and the limb CRT as whole-call ratios (B against reference, above 1 is faster), everything bit-identical (all `*-complete` GO, 19 hashes reproduced on both Hopper cards and Blackwell):

| candidate | H100 / H200 | Blackwell |
|---|---|---|
| E1 against the baseline | +0.6..2.7%, ties (KEEP) | ties |
| Q1 alone | ties (1.009..1.018), 0.981..0.987 at g = 7 | not run |
| F1 alone | ADOPT: 2048 1.029, r_lowk 1.061, t_df 1.037, the rest 1.01..1.02 | not run |
| limb CRT | ADOPT: 4096 N = 14 1.049, 8192 N = 14 1.032, r_lowk 1.143, t_df 1.086 | **NO-GO, 0.44..0.84** (FP64 at 1/64) |
| limb CRT + F1 (`crtf1`) | ADOPT: 4096 N = 14 1.058, 8192 N = 14 1.041, 2048 1.074, r_lowk 1.183; floor 1.017 | not run |
| whole stack with Q1 (`crtall`) | 1.130 at 2048, 1.094 at 4096 N = 14, 1.066 at 8192 N = 14, r_lowk 1.261 | not run |

The pre-registered combination rule (rule 3 of the decisions file header) applied: Q1 KEEP (off) on both cards, F1 ADOPT (on), the limb CRT taken from the `crtf1` duel (ADOPT on both cards, so on), E1 stays (neutral, no regression). **sm_90 default = E1 + F1 + limb CRT = `kSeqEmit | kSeqFinish | kSeqLimbCrt` = 13, Q1 off.** The combination itself was not measured directly: it is judged by its components (`crtf1`, floor 1.017 on both cards, GO, no case slower). The figures usually quoted for it, +4..7% of the whole call at the N = 14 squares (1.041 / 1.058 / 1.074 at 8192 / 4096 / 2048) and 1.183 at r_lowk, are the `crtf1` duel ratios: the limb CRT's gain given E1 + F1, not E1 + F1 + limb CRT against the old default. The adopted set was not measured directly against the old default; chaining f1 x crtf1 gives about +6..10% at the same cases. Everything else is unchanged: sm_89 stays E1 = 1 (the RTX 4090 has the same 1/64-rate FP64 pipe as Blackwell: the limb CRT and F1 would run on its slow FP64 pipe), sm_120 and every unlisted class stay 0.

* **Where it lives.** One table row: `kSeqClasses` in `src/host/seq_config.hpp` (`{9, 0, kSeqEmit | kSeqFinish | kSeqLimbCrt}`). The limb CRT is bit 8 of the class set (`ExecPlan::seq` is a `uint8_t`; the plan's reported set, `imd_test_plan_seq` / `imd_prof_plan_seq`, now reads 13 on sm_90). `launch_k5` takes the limb CRT when the set it runs has bit 8, unless the K5 hook forces a CRT; `imd_test_plan_crt` / `imd_prof_plan_crt` report which CRT the plan's K5 runs.
* **Forcing.** The hooks still force every combination. `seq=default crt=int` is the integer CRT on every class (the "baseline CRT" on sm_90, with E1 + F1); `seq=0..7` keep their meaning and carry no limb-CRT bit, so a forced set runs the integer CRT unless `crt=f64` is given (`seq=7 crt=f64` is the old R2 "crtall", `seq=0 crt=int` the old default baseline, `seq=5 crt=f64` the adopted set spelled out). The K5 hook is tri-state: -1 (the process default) follows the plan's set, 0 forces the integer CRT, 1 the limb CRT; `imd_prof_get_k5_crt` returns that state (it used to return 0 or 1; the effective CRT is `imd_prof_plan_crt`). The harness's `IMD_TEST_SEQ` forces a set 0..7 and so runs the integer CRT unless a test forces limbs; the sm_90 default path (13) is exercised by the unforced runs.
* **Duel baselines move on sm_90.** `--duel-a default` is now the adopted set there, not the old baseline: to duel against the pre-R2 behaviour use `seq=1,crt=int` (the old class default) or `seq=0,crt=int` (the baseline of the R2 duels). The duel side key `crt=` has a third value `default`.
* **What stays opt-in and why.** Q1 (`seq=3`, `seq=7`): alone it ties on H100 and H200 and is 0.98x at g = 7 (the floor), it costs registers (`k23_pack<2,3>` 94 against 64, 2 CTAs per SM instead of 4) and the rule does not allow keeping a candidate on its own evidence when it only ties. The apparent oddity that the whole stack with Q1 (`crtall`, 1.094 at 4096 N = 14) beat `crtf1` (1.058) is a baseline artefact, found by the independent review of the R2 data: the duels have different A sides (`crtf1` is seq 5 against seq 5 + crt=f64, `crtall` seq 1 against seq 7 + crt=f64). Chaining f1 x crtf1 x (Q1 alone) reproduces `crtall` (implied Q1 1.012 against 1.010 measured alone at 4096, 0.980 against 0.981 at g = 7, same on H200): no interaction. The R3 duel that matters is the adopted default against the old one, `seq=5,crt=f64` against `seq=1,crt=int`.
* **Reading R2's records after this change.** On sm_90, R2's `--duel-a default` / `crt=int` decisions (`duel.crt`, `duel.default`, `e1-default-guard`, and the `imd.phases.seq` readback `crt_after_reset == "int"`) mean something different now: `default` is the adopted set 13 and `crt_after_reset` reads `f64`. The R3 manifest must therefore spell the sides out (`seq=1,crt=int` against `seq=5,crt=f64`) and update the readback check; the generalWork manifests are not edited here. Duel records now carry `seq_eff` and `crt_eff` per side (`duel-a` / `duel-b`: the sequence set and CRT the plan ran, read back with the side's hooks set; `duel_config` still only says what was asked), so a re-analysis can see what each side ran.
* **Walking it back.** One table row: change `{9, 0, kSeqEmit | kSeqFinish | kSeqLimbCrt}` to `{9, 0, kSeqEmit}` (the pre-R2 row), or drop `kSeqLimbCrt` or `kSeqFinish` alone; then `DEFAULTS` in `tests/test_seq_select.py`, `LIMB_CRT_CLASSES` in `tests/test_crt_f64_gpu.py` and the expectations in `tests/test_seq_table.cpp` (CPU-only, ctest `host_seq_table`) follow the table, and nothing else refers to it. The sm_89 SASS is unchanged by this change (the new code is host-side; the K5 kernels are the same instances).
* **Not verified here.** No Hopper hardware on the dev box: the sm_90 build was cross-compiled and the table is pinned by the CPU-only test, but the runtime path with the limb CRT as the default (and the default's bit-identity on an H100) is checked by the same tests only on the cluster (`seq_select`, `crt_f64_gpu`, `bench_seq_crt`, and the oracle / committed-hash suites, which now run the sm_90 default). The performance model's K5 occupancy rows still describe the integer-CRT instances, as they described the baseline instances before; its K5 time on sm_90 is the integer CRT's (the calibration will absorb it when re-fitted).
