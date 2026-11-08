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
