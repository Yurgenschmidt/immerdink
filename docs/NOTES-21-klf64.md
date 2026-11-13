# Issue 21 'klf64': KL's sigma from a binary64 square sum with an exact fallback (Hopper)

Status: implemented as instruction-sequence bit 256 (`kSeqKlF64`, host/seq_config.hpp), **off in every device-class row** until a Hopper duel
measures it. Bit-identical to the exact kernel by construction (a rigorous filter, below) and by test. Written for a reader new to the project.

## Terms

* **KL** (DESIGN §4): the kernel that scales one long row (or column) of an operand by a power of two 2^sigma and writes its N residues. One
  CTA owns a tile of rows; the per-row *fixed part* (finding sigma) does not depend on N, the *emit* part (residues) scales with N.
* **sigma** (DESIGN §3.2 step 4): for a row with largest exponent gamma = max ilogb(x) and the exact integer square sum
  S = sum ceil(m^2 2^(2(e - gamma) + 60)) over its finite nonzero entries x = m 2^e, u = the largest integer with c S 4^u <= R
  (c, R plan constants), and sigma = u - gamma. The kernels compute S exactly in 128-bit integers.
* **Instruction-sequence set** (docs/NOTES-21.md): bits that select bit-identical kernel twins per device class (E1 = 1, F1 = 4,
  klwarp = 16, ...). The test hook `imd_test_set_seq` / env `IMD_TEST_SEQ` forces a set on any card.

## Why (the limiter, with evidence)

The method is in immerdink-eval `methods/conversion-kernel-budget.md`. A streaming kernel is memory-bound on a card if its thread
instructions per DRAM byte I stay below about 0.7 B, where B = 32 x SMs x 4 x f / BW is the card's issue budget per byte:
B = 41 on the RTX 4090, 35 on the RTX PRO 6000 Blackwell, **10 on H100 and 7 on H200**. ncu on the 4090 (sq4096, N = 14) gives KL I = 21:
memory-bound there, **issue-bound on Hopper**. The model predicts the H100 KL time at full issue as 0.36 ms against 0.527 ms measured
(68 % issue efficiency), consistent with the October timers. So on Hopper only fewer instructions per element help.

Bucketing KL's executed instructions by source (ncu SourceCounters, E1 set) gives per element: emit ~110, gamma scan ~52, the exact
square sum ~51 plus its 128-bit reduction ~26, quantize ~62, setup and reductions ~43. Hopper's FP64 pipe is full rate (1/64 on the 4090),
and an integer-free square sum in binary64 is ~3 instructions per element. That is the cut.

## What the bit does

With klf64 (needs klwarp: it uses the per-row sigma of that variant), KL first runs two cheap passes over the same loads in the same order
as the exact passes (so the emit pass finds the registers and the shared tile in the state it expects):

1. **gamma**: the max of |high word| (sign masked) and the OR of the low words. A non-finite entry has |hi| >= 0x7FF00000, so the flag is
   exact; if the max is a normal number's, gamma = its exponent field - 1023.
2. **S estimate**: y = x 2^(30 - gamma) by two exact power-of-two multiplies, St = sum fma(y, y, acc), xor-butterfly sums across lanes and warps.
   Every operation uses an explicit `_rn` intrinsic.
3. **filter** (`numerics.cuh largest_u_filtered`): with n = k_hat + 2,
   `hi = RU(RU(St (1 + n 2^-52)) + k_hat + 1)` and `lo = max(RD(St (1 - n 2^-52)), 2^60)`. Then S is in [lo, hi] (proof below), u(S) does not
   increase with S, so **u(hi) == u(lo) decides u exactly**.
4. **fallback**: if any row of the tile is undecided, has a non-finite entry, or has a subnormal largest entry, the whole tile
   (`__syncthreads_or`, block-uniform because the exact passes use full-mask shuffles and barriers) runs the original exact passes.

The host drops the bit for k_hat >= 2^50 (the constants above are then no longer exact doubles).

### Why [lo, hi] contains S

* Higham's bound for any summation order: |St - sum y^2| <= gamma_n sum y^2, gamma_n = n u / (1 - n u), u = 2^-53, where n counts the
  inexact roundings on any path; zero operands are exact, so n <= nnz <= k_hat. 1 + 2 n u >= 1 / (1 - gamma_n) and 1 - 2 n u <= 1 / (1 + gamma_n)
  for 4 n u <= 1, and 2 n u <= (k_hat + 2) 2^-52.
* sum y^2 <= S < sum y^2 + nnz: each term of S is the ceiling of the exact y^2 (terms >= 1 are exact integers).
* Underflow in y (rows with very small entries next to a large one) adds absolute errors below 2^-1000 per term: covered by the + 1.
* S >= y_max^2 >= 2^60, which justifies the clamp; every double >= 2^60 is an integer, so lo and hi convert to 128 bits exactly (hi < 2^115),
  by exponent and mantissa (never `__double2ull`: S can exceed 2^64).

The filter was modeled in Python against `spec/oracle.py` before any device code (immerdink-eval `analysis/20261108-hopper/sfilter_model.py`:
1320 random and threshold rows bit-identical) and reviewed independently (verdict "sound with fixes": the directed rounding, the u128
conversion, non-finite rows to the exact path, the low-word zero test ([5e-324] has a zero high word), the block-uniform fallback, explicit
`_rn`, the k_hat limit; all in the code; ~7,400 extra adversarial rows bit-identical).

## Expected effect (prediction, before any Hopper measurement)

KL fixed part per element ~234 -> ~115 (scan 52 -> ~6, square sum + reduction 77 -> ~8, gamma / sigma reductions shrink); total with E1
~344 -> ~225 thread instructions per element (-35 %). H100 sq4096 KL 0.527 -> ~0.34 ms (about -10 % of the call). On the 4090 the bit can only
cost (FP64 at 1/64 rate, and KL is memory-bound there): it is a per-device-class choice, sm_90 only.

Fallback rate: random data almost never straddles a threshold (the interval is ~k_hat ulps of S wide against thresholds a factor 4 apart);
the threshold suite below forces it on purpose.

## Tests

* `test_kl_bits.py`: every combination of klwarp / klload / klshape / klf64 over the emitter sets, one hash per case, and the launch counter
  (`imd_test_kl_launch_counts`, now int64[24]: [E][Kb 0..7]) proves the klf64 kernel ran.
* `test_contract.py --suite klf64` (`contract_klf64_seq{-1,273,305,272,437}`): operands whose rows' square sums sit on, or 1 to 2^30 units
  from, a threshold of largest_u (many equal entries make the binary64 sum round many times), mixed in tiles with random, zero, lone-5e-324,
  subnormal-only, Inf / NaN rows; one, two and more than two chunks per row; RN and TRUNC; bit-exact against the oracle.
* The oracle suites and committed hashes under 273 (E1 + klwarp + klf64), 305 (+ klload), 272 (baseline emitter), bigk under 273, and the
  R4 forced set 437 (the sm_90 default 181 + klf64).

## Follow-up: the emit loop's per-modulus overhead (klwarp + E1)

**Evidence.** After klf64, the 4090 ncu counts of `kl_scale_pack<8,2,1,7>` (sq4096, N = 14) are 219 issued instructions per element, 108 of them
on the fmaheavy pipe (IMAD, IMAD.WIDE, IDP, IMAD.MOV). IMAD and IDP issue at half rate, so the per-pipe bound is t ~ max(issue, 2 x fmaheavy)
= max(219, 216): both bind on Hopper. The hottest loop (one iteration per modulus, ~1.75 iterations per element) is 63 instructions. 9 of them are
bookkeeping on the fmaheavy pipe: `IMAD.MOV 0x20` and `IMAD l, 32, table` re-derive the constant-table address of modulus l, and two IMAD,
an IMAD.WIDE, an IMAD.IADD and three IMAD.MOV redo the 64-bit `plane = (b rows_pad) k_hat` multiply on every iteration (the compiler
rematerializes it instead of holding it in registers).

**Change.** With klwarp the host now passes `plane` in LargeArgs, and the loop (`store_slots_stepped`, pack_emitter.cuh) walks the
modulus' `PackEmit` entry by pointer. Both advance by additions on the ALU pipe. The bytes and addresses are the same; only kernels with klwarp
change (they are off by default on every row except sm_90).

**Prediction (before measuring).** Per iteration -9 fmaheavy, +~3 ALU (IADD3 for the pointer, IADD3 / IADD3.X for dst). Per element:
fmaheavy 108 -> ~92, issued 219 -> ~208, so the bound becomes max(208, 184): issue alone, about -5 % of KL time on H100 / H200 at
unchanged issue efficiency (~-1 % of an sq4096 call). On the 4090, KL is memory-bound: no change.

**SASS check (sm_90a, still before measuring).** The loop of `kl_scale_pack<8,2,1,7>` is 60 instructions (the same loop without klwarp,
`<8,2,1,0>`, is 63). Its bookkeeping is now VIADD x2, ULDC.64, IADD3 / IADD3.X and two IMAD.MOV (copies of dst kept for the store, which the
scheduler issues after the next address is formed), against nine IMAD-family instructions before. The refined prediction is therefore
-7 fmaheavy and -3 issued per iteration: per element fmaheavy 108 -> ~96 and issued 219 -> ~213. That is about -3 % of KL time on Hopper,
smaller than the first estimate because the replacement additions still issue.

## KF twins: 'kff64' (bit 16384) and 'kfaff' (bit 32768), issue 21 task 28

**KF** is the fused pack for small k (k_hat <= 1024, DESIGN v1.4 §4): one launch reads both operands' tiles into shared memory, finds each
row's sigma (the same K1 integers as KL) and writes group 1's residues. It is the pack of every "remaining" Hopper shape: r_lowk (8192 x 8192 x 1024),
t_batch (16 x 1024^3), sq1024, s512, s256.

**Limiter, with evidence.** H200 job 15566591 (k16), r_lowk N = 14, KF with min-blocks 4: 0.232 ms, 61 % issue active, 48 % warps active,
143.3 M warp instructions for 16.8 M elements = **272 thread instructions per element**; DRAM floor 0.08 ms, issue floor 0.14 ms (t_batch: the
same 274 per element, 0.454 ms). The KF source counters (ncu SourceCounters of the k16 report, split at the kernel's barriers) give per element:

| part | instructions / element | stall samples |
|---|---|---|
| step 1, descriptor: the per-tile table of k offsets `s_vk` (one 32-bit division per column and tile), row bases | ~33 (the table ~20) | 10 % |
| step 2, tile load into shared memory | ~36 | 36 % |
| step 3, gamma scan + flag + their reductions | ~30 | 10 % |
| step 3, exact u128 square sum + reductions + largest_u | ~65 | 21 % |
| step 5, emit (14 moduli) | ~104 | 21 % |

So, as in KL before klf64, the fixed part (everything except the emit) is ~60 % of KF's instructions on a card whose FP64 pipe is full rate.

**What the bits do.** `kff64` is klf64 inside KF: gamma from the max high word, S as a binary64 sum of y^2 with y = x 2^(30 - gamma), and the
filter `largest_u_filtered` decides u; a tile with an undecided, non-finite or subnormal-max row runs the exact integers (block-uniform,
`__syncthreads_or`). The proof above does not depend on the summation order or on where the row is held, and KF's rows are whole tile rows
(k_hat <= 1024 << 2^50), so it carries over unchanged. `kfaff` replaces the table of k offsets by h * stride when the K class is one fused
mode (the usual GEMM; k0_offset of one mode is exactly that for h < k). Both are bit-identical twins (template value Kf of kf_scale_pack),
compiled for the sm_90 emitter sets only (pack S = 15 at W = 2, S = 1 at W = 3) and the small descriptor capacity; elsewhere the launch runs the
shipped KF. The 606 shipped sm_90 kernels are SASS-identical with and without the twins; the twins use 64 registers (spill: kff64 E = 16 W = 2
76 / 136 B against 76 / 116 B; the others equal or less).

**Prediction (before any measurement).** Per element: step 3 ~95 -> ~15 (kff64), step 1 ~33 -> ~13 (kfaff), total ~272 -> ~175 (-35 %).
KF is not purely issue-bound (61 % issue, 36 % of the samples wait on the tile's loads), so the time gain is smaller: KF -20 .. -30 % on H200
(r_lowk 0.236 -> 0.17 .. 0.19 ms, t_batch 0.462 -> 0.33 .. 0.37 ms), end to end t_batch 1.10 .. 1.13x, sq1024 1.05 .. 1.10x,
r_lowk 1.02x, s256 / s512 within noise or better. On the 4090 the bits can only cost (KF is memory-bound there, FP64 at 1/64 rate): sm_90 row only.

**Tests.** `test_kf_bits.py` (ctest kf_bits): every combination of the two bits over emitter sets with and without a KF twin, 12 contractions
(k_hat 16 .. 1024, N = 8 / 14 / 20, batched, g > 1, a transposed operand, an odd pitch, a K class of two modes, rows of special values), with
and without the small-problem variants, one hash per case, and the KF launch counter (`imd_test_kf_launch_counts`) shows the twin ran.
The klf64 threshold suite gained KF shapes (k_hat 64 / 900 / 1000 / 1024) and runs under 24503 (8119 + kff64), 57271 (+ kfaff) and
19895 (3511 + kff64); e2e / tensor / eps and the committed hashes run under 57271 and 40887 (8119 + kfaff).

**Review.** An independent read-only review (2026-11-13) found no blocker: barriers and full-mask shuffles block-uniform, every shared-memory
write-after-read separated by a barrier, the zero / invalid / padding / non-finite / subnormal-max rows equal to the exact path, the filter's
preconditions met (St >= 2^60, k_hat << 2^50), every lane of a row with the same sigma, kfaff equal to k0_offset for h < k, and no twin
launched without being instantiated. Acted on: `ExecPlan::seq` is 16 bits and now full (a static_assert guards the next bit), and the bench's
`--seq` accepts the new bits. Left as is: when a row spans two warps, both warps run the filter (klf64 runs it once per row; two extra 128-bit
compares per row of 1024 values).
