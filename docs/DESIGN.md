# immerdink — design (v1.4)

This is the single source of truth for *what* v1 computes and *how*.
- TLA+ specs in `spec/` check the claims marked **[spec: X]** (see `spec/README.md`).
- `spec/params.py` checks the real-width arithmetic facts, marked **[cert]**.
- `spec/oracle.py` is the exact, bit-level executable reference of §3. GPU results must
  match it bit for bit.
- Hand proofs of the parametric lemmas are in `docs/PROOFS.md`; they are the blueprint for
  the Lean formalization.
- **[deviation]** marks a departure from published Ozaki-II, with the reason.

v1.1 folds in the corrections found by the spec phase: adversarial review plus mutation
testing of every artifact. v1.2 changes execution details after M4 (K0 generated on the
device or plan-resident, GEMM parameters built at plan creation; a K1 split was specified
and measured but not adopted) and the §6 thread and lifetime rules that follow from them; changed text is marked **[v1.2]**. v1.3 adds
§6a, the CUDA-free host query API (no semantic change). v1.4 adds the fused small-k launch KF
(§4, chosen at plan time; same results, same workspace) and reports the launch structure in
the host API (§6a); changed text is marked **[v1.4]**. The change lists are at the end.

Sources:
- Ozaki–Uchino–Imamura, arXiv:2504.08009 v4 [OZ2].
- Uchino et al., arXiv:2508.03984 [SC25].
- Error analysis, arXiv:2602.02549 [ERR].
- Complex variants: arXiv:2512.08321 [CPLX] and arXiv:2609.05419 [2M].
- GEMMul8 v3.5.2, commit 6a9eff8 [G8].
- CUTLASS v4.8.0 [CUT].

---

## 1. What is computed

Inputs:
- FP64 tensors A, B and C with arbitrary int64 strides;
- integer labels per mode;
- FP64 scalars α and β;
- options: `N` (the number of moduli, 2…20), `rounding` (RN by default, or TRUNC), and `g`
  (moduli per residue group, 1…N, default N).

The result:

```
C[I,J,L] ← α · Σ_K A[I,K,L] · B[K,J,L]  +  β · C[I,J,L]
```

The sum is computed by the integer pipeline in §3. Results are **bitwise deterministic**:
they are a pure function of (input bits, α, β, N, rounding, library version). They do not
depend on:
- launch configuration or tile shapes;
- K-chunking or the residue grouping `g`;
- the stream.

**[spec: Pipeline, OzakiCRT; oracle tests A8 and chunk/g invariance]**

### 1.1 The GEMM view (IndexMap, §5)

Label analysis lowers every legal contraction to `b` independent GEMMs over canonical
index spaces, with `ī ∈ [0,m)`, `k̄ ∈ [0,k)`, `j̄ ∈ [0,n)` and `λ ∈ [0,b)`:

```
C_λ[ī,j̄] ← α Σ_k̄ A_λ[ī,k̄] B_λ[k̄,j̄] + β C_λ[ī,j̄]
```

The canonical orders are a fixed function of the descriptors (§5 step 8). NaN results
are NaN, but their payload and sign are unspecified.

---

## 2. Constants (offline, exact; [cert])

| Symbol | Definition |
|---|---|
| `p_1..p_20` | Greedy pairwise-coprime moduli, from **exactly 2⁸ = 256** downward: **256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173**. This is identical to [G8] `table.hpp`. [OZ2] eq. int8_law8 omits 241; that is an erratum. All `p_l ≤ 256` is required by the int8 operand store (§3.4). |
| `P` | `∏_{l≤N} p_l`. It is always even. `log2 P` = 63.58 / 71.41 / 79.24 / 110.16 / 117.78 / 125.38 at N = 8 / 9 / 10 / 14 / 15 / 16. |
| `H` | `⌊(P−1)/2⌋ = P/2 − 1`. |
| `M_l, q_l` | `M_l = P/p_l` and `q_l = M_l⁻¹ mod p_l ∈ [1,p_l)`. Then `M_l q_l ≡ δ_lj (mod p_j)`. |
| `T` | `⌊√H⌋`, the per-side norm budget. `T² ≤ H < (T+1)²`. |
| `K_c` | `131071 = ⌊(2³¹−1)/128²⌋`. It is tight: `128²·(K_c+1) > 2³¹−1`. CUTLASS 2.x int8 MMA is `.satfinite` ([CUT] `mma_sm80.h`), so overflow **saturates silently**. **Every modulus, including p = 256, must be K-chunked.** The wraparound exemption [G8] relies on with cuBLAS does not apply here. **[spec: OzakiCRT ChunkFits, neg_chunk, neg_order]** |
| `K_c'` | `131056 = ⌊K_c/16⌋·16`, the chunk length used. Numerically any chunk length ≤ K_c is exact. The multiple of 16 is only there to keep CUTLASS pointers 16-byte aligned. |
| `L_w` | Width of the **unsigned** CRT word, `32·L32` with `L32` minimal such that `2P ≤ 2^{L_w}`. `L32` = 1 for N ≤ 3, 2 for N ≤ 7, 3 for N ≤ 12, 4 for N ≤ 16, 5 for N ≤ 20. The bound is exact: the reachable peak needs every bit. |

All constants are generated and checked by `spec/params.py` in exact integer arithmetic.
It is the only producer of the device constant tables (`--emit-header`, which writes
`include/immerdink/detail/constants.hpp`; `--json` for tooling). We use no
floating-point constants. [G8]'s N=2 `log2P` entry is mis-rounded.

---

## 3. Numerics, step by step

This is per batch λ. Rows of A are shown; columns of B are symmetric.

### 3.1 Non-finite scan (fused into §3.2's pass)
- `fA[ī] = 1` iff row ī of A_λ contains Inf or NaN. Detection uses the exponent field
  (`== 0x7FF`), never max/norm arithmetic. `fB[j̄]` is defined the same way for columns of B.
- Non-finite entries are treated as 0 by every later emulated step.
- Flagged outputs go through the native path (§3.7).

### 3.2 Scaling: exact-integer fast mode (Cauchy–Schwarz) — **[deviation]**
For row ī, over its finite entries `a_h = m_h·2^{e_h}`, where `m_h` is a 53-bit integer
significand:

1. **Zero rows.** If the row is all zero, then `σ := 0` and skip to step 5.
2. **Row exponent.** `γ = ilogb(max_h |a_h|)`, which lies in [−1074, 1023].
3. **Sum of squares.** `S = Σ_h ⌈x_h²·2^F⌉`, where `x_h = a_h·2^{−γ}` and `F = 60`.
   - Each term is computed **from integers**, as `ceil_shift(m_h², 2(e_h − γ) + F)`, never
     by FP scaling: `2^{−γ}` overflows, and ldexp underflows.
   - Any nonzero entry contributes at least 1.
   - Terms are < 2⁶³. S is exact, order-independent, and fits uint128 for `k < 2⁶⁶`.
   - `‖x‖²·2^F ≤ S < ‖x‖²·2^F + k`.
4. **Shift.** `σ = u − γ`, where `u` is the **largest integer** satisfying
   - RN: `4·S·4^u ≤ R·2^F`, with `R = (2T − κ)²`;
   - TRUNC: `S·4^u ≤ R·2^F`, with `R = T²`.

   Here `κ = ⌈√k⌉` is computed exactly in integers (`c = isqrt(k); κ = c + (c² < k)`).
   Using `⌊√k⌋` is **unsound** **[spec: OzakiScaling neg_kappa_floor; cert]**.

   Computing u:
   - R is a per-plan constant of at most 157 bits.
   - Estimate `c = (bitlen(R·2^F) − bitlen(LhsCoef·S) + 1) >> 1`. This must be an
     **arithmetic (floor) shift**, because the difference can be negative at real widths.
   - Then do one exact multi-word compare (about 220 bits at N = 20): `u = Rule(c) ? c : c−1`.
   - **[spec: OzakiScaling ShiftIsMaximal, UOf; cert]**
5. **Store.** `σ` is stored as int16.
   - The exact range is `[−1037, 1151]` for k ≤ 2⁴⁰, N = 2…20, both roundings.
   - `σ + τ ∈ [−2074, 2302]` also fits int16.
   - **[cert: cert_shift_range]**

**RN acceptance.**
- Soundness needs `2T − κ ≥ 1`. At `κ = 2T` no u exists; beyond that the squared rule is
  unsound. **[spec: OzakiScaling pos_kappa_2Tm1, neg_kappa_2T, neg_noreject]**
- As an accuracy policy, plan time rejects `κ ≥ T` (NOT_SUPPORTED: "N too small for this
  k"). It applies only to RN (TRUNC has no κ term), only in EMULATE with a non-empty
  output and k ≥ 2, and is independent of α. (At real sizes, T(N) ≥ 180 makes the k ≥ 2
  clause implied; it is kept for a symbolic T.) It is the **last** plan-time check, after
  every §5 row, so e.g. a GemmCoord overflow reports INVALID_VALUE first.
  **[spec: IndexMap pos_kappa, neg_kappa_{drop,direct,empty,k,trunc}]**
- The §3.8 bound already exceeds 1 well before that, once `κ ≳ 0.4T`. So choose N with
  `params.py --table`.

**Guarantee.** `‖a'‖₂ ≤ T` (RN: `‖a·2^σ‖ ≤ T − κ/2`, plus a rounding error of at most
`√k/2`). Hence `Σ_h |a'_h||b'_h| ≤ T² ≤ H` for every (ī, j̄).
**[spec: OzakiScaling NormBudget, CRTPrecondition; cert: cert_shift_rule_guarantee,
cert_kappa_slack_needed; proof: PROOFS L1]**

Why we deviate:
- [OZ2], [SC25], [ERR] and [G8] all pick the shift with floating-point `log2`/`__log2f`.
  [G8] also squares raw entries, which overflows for `|a| ≥ 2⁵¹²` and silently corrupts
  the row.
- The integer rule is overflow-free and order-independent, hence reproducible, and it can
  be checked and proved.
- It is at least as tight: it finds the maximal u exactly instead of rounding a log down.

### 3.3 Quantization — **[deviation: RN by default]**
- `a'_h = RN_even(a_h·2^σ)`; in TRUNC mode, `trunc(a_h·2^σ)`.
  - RN must be ties-to-even; ties-away fails **[spec: OzakiScaling QuantIsRNEven]**.
  - TRUNC must truncate toward zero, not floor.
- The value is held exactly as `μ·2^ε` with `|μ| ≤ 2⁵³` and `0 ≤ ε ≤ ⌊log2 T⌋`
  (`|a'| ≤ T` reaches 2⁷⁷ at N = 20). It is never forced into int64.
- **Why RN.** It is unbiased and halves the worst-case per-entry error.
  - TRUNC biases every product toward zero.
  - Measured with the oracle: RN is about 1.5–3× more accurate (median) on zero-mean random
    data, and 19–191× on same-sign data. One modulus is worth about 16× (≈4 bits per
    operand), so that is about 0.3 moduli on zero-mean data and 1–2 on same-sign data. At
    k = 1000 and φ ≤ 1, RN reaches native-FP64-level accuracy at N = 14, where TRUNC needs
    15.
- **The cost** is the `κ/2` slack in §3.2: a fraction `κ/(2T)` of the budget, which is
  `< 2⁻²⁰` for N ≥ 8 and k ≤ 2²⁴.
- TRUNC stays available for comparisons that follow the paper exactly.

### 3.4 Residues
- `r_l(a') = SMod(a', p_l) ∈ [−⌊p_l/2⌋, ⌈p_l/2⌉−1]`. This fits int8 (`[−128,127]` for
  p = 256).
- It is computed exactly as `SMod((μ mod p_l)·(2^ε mod p_l), p_l)`.
  - `mod` here is the **non-negative** mathematical mod. C `%` on a negative μ is wrong.
  - It uses a `2^ε mod p_l` table.
- Any exact reduction is allowed (Barrett, dp4a byte weights); only the congruence class
  matters.
- The int8 store is correct only because `p_l ≤ 256` **[spec: OzakiCRT neg_resbits_p9,
  neg_resbits_p9_store]**.

### 3.5 Per-modulus INT8 GEMM with modular epilogue
- Split canonical k̄ into chunks `c = 0, 1, …` of length `K_c'`. The last chunk may be
  shorter.
- For each chunk: `acc = Σ_{h∈chunk} r_l(a'_īh)·r_l(b'_hj̄)` in int32.
  - It is exact in any accumulation order, because the bound is on `Σ|products| ≤
    128²·K_c ≤ 2³¹−1`, which bounds every partial sum.
  - K_c must **not** be sized from data: a final sum that fits is not enough under
    `.satfinite` **[spec: OzakiCRT neg_order]**.
- The epilogue stores u8:
  - `c = 0`: `W ← acc mod p_l`;
  - `c > 0`: `W ← (W + acc) mod p_l`.
- Here `mod` is the **non-negative residue in [0, p_l)**. It must be exact over the
  reachable input `v ∈ [−K_c'·128², (p_l−1) + K_c'·128²]`.
  - Per-modulus exact ranges are in `params_out.json`.
  - The max of `W + acc` is 2147221759, which fits int32 **[cert: cert_kc]**.
- **Realization [cert: cert_barrett]:**
  - p = 256: `v & 255`;
  - otherwise: `n = v + bias` (u32, `bias = p·⌈2³¹/p⌉`), `q = umulhi64(⌈2⁶⁴/p⌉, n)`,
    `v mod p = n − p·q`.
  - A 32×32 Granlund–Montgomery multiply is **insufficient** for 7 of the moduli.
  - The CUTLASS `rem.s32` plus saturating u8 store would silently clamp negative
    remainders to 0. That is wrong.
- Result: `W_l ≡ (A'B')_īj̄ (mod p_l)`. **[spec: OzakiCRT WCongruent, CRTExact, neg_carry]**

### 3.6 CRT reconstruction (exact, multi-word) — **[deviation from D10 as first stated]**
```
x ← 0                                        (unsigned L_w-bit word: L32 little-endian u32 limbs)
for l = 1..N:  y ← (q_l·W_l) mod p_l         (16-bit value; same umulhi64(⌈2⁶⁴/p⌉) reduction as §3.5)
               x ← x + M_l·y                 (one fused carry chain: t = x_k + M_l,k·y + carry in u64)
               if x ≥ P: x ← x − P           (invariant x < P; peak < 2P ≤ 2^{L_w})
z ← (x ≤ H) ? x : x − P                      (z ∈ [H+1−P, H] = [−P/2, P/2−1])
```
- `Σ_h|a'||b'| ≤ H` implies `z = (A'B')_īj̄` exactly; at H+1 this can fail.
  **[spec: OzakiCRT CRTExact, neg_bound; CRTRealization Refines (3 layers: exact CRT →
  word → limbs); cert: cert_crt_peak, cert_crt_roundtrip; proof: PROOFS L2]**
- The word is **unsigned**. A signed `__int128` would need `2P ≤ 2^{L_w−1}`, and signed
  overflow is UB.
- In the centering step, compare the sign bit with the borrow-out
  **[spec: CRTRealization pos_wordref_nosub]**.
- "No division" means no wide division. `x + M_l y = P` can never occur exactly, so `≥` and
  `>` behave the same.
- [G8]'s Garner tree is an equivalent exact alternative. [OZ2]'s double-double FP
  reconstruction is superseded, and its error term R_64 disappears.

### 3.7 Output
- **Emulated:** `D = RN_double(z·2^{−(σ_ī+τ_j̄)})`, a **single rounding** directly at the
  final exponent, including the subnormal range and overflow to ±Inf.
  - [G8] rounds twice for subnormal results.
  - `z = 0` gives `D = +0`. A nonzero z that rounds to zero gives a zero with z's sign.
  - **[spec: OzakiScaling FinalRounding, neg double-rounding; oracle A1]**
- **Flagged** (`fA[ī] ∨ fB[j̄]`): `D = acc` after `acc = +0; for h in canonical k̄ order:
  acc = fma(a_h, b_h, acc)`.
  - D is always ±Inf or NaN here.
  - The order matters only if a finite partial sum overflows: `[MAX,MAX,−Inf]·[2,2,1]` gives
    NaN forward and −Inf reversed **[oracle A7]**.
- **Scaling.**
  - `E = (α = 1) ? D : RN(α·D)`.
  - `C ← (β = 0) ? E : fma(β, C, E)`, a single fma **[oracle A6]**.
  - C is **not read** when β = 0.
- **α = 0 or k = 0:** `C ← (β = 0) ? +0 : RN(β·C)`.
  - A and B are not read, so non-finite values in them are suppressed (BLAS semantics).
  - K1–K4 are not launched, and K5 runs in β-only mode, reading only C's scatter vectors
    **[spec: Pipeline neg_alpha0_k5_full]**.

### 3.8 Error contract (finite inputs, no over/underflow)
With `s = √(1 + k·2^{−F})`, which is `< 1 + 2⁻²¹` for k ≤ 2⁴⁰:
```
RN:     |D − Σ_h a_h b_h| ≤ 2√k·‖a_ī‖₂‖b_j̄‖₂·s / (T − κ/2)  +  k·2^{−σ−τ−2}  +  u·|D|
TRUNC:  |D − Σ_h a_h b_h| ≤ 4√k·‖a_ī‖₂‖b_j̄‖₂·s / T          +  k·2^{−σ−τ}    +  u·|D|
```
- These come from the exact ℓ1 form `|err| ≤ δ(2^{−τ}‖a‖₁ + 2^{−σ}‖b‖₁) + k·δ²·2^{−σ−τ}`,
  with δ = ½ (RN) or 1 (TRUNC), and from `2^{−σ} < 2‖a‖·s/(T − κ/2)` (RN; `/T` for TRUNC).
- Without the factor s the bound is **false**: S's ceiling slack is real, and there are
  counterexamples at N = 14 and N = 20.
- The bound is **normwise per (row, column)**, as for every Ozaki scheme.
- **[spec: OzakiScaling ErrorBound, SigmaBound, ContractL2, neg_sigmabound_literal; cert:
  cert_end_to_end; oracle; proof: PROOFS L3]**

Worst-case guidance, `2√k/T` relative to `‖a_ī‖‖b_j̄‖`:

| N | k = 10⁴ | k = 2¹⁷ |
|---|---|---|
| 6 | 1.8e−5 | 6.5e−5 |
| 8 | 7.6e−8 | 2.8e−7 |
| 9 | 5.0e−9 | 1.8e−8 |
| 10 | 3.3e−10 | 1.2e−9 |
| 12 | 1.5e−12 | 5.5e−12 |
| 14 | 7.4e−15 | 2.7e−14 |
| 16 | 3.8e−17 (< u) | 1.4e−16 (≈ 1.2u) |

Typical errors are smaller. Choosing N is the caller's job, guided by
`params.py --table`.

---

## 4. Execution pipeline (one stream, in order)

```
plan time:  IndexMap lowering → fused-mode descriptor, canonical orders, chunking, launch
            split; K4 GEMM parameters (CUTLASS device properties, occupancy, smem attribute)
K0 vec      kernel: generate the scatter vectors → ws.vec from the fused-mode descriptor
            (a kernel argument, by value); not launched with IMD_VEC_RESIDENT      [v1.2]
K1 scale    A rows, B cols: flags + σ, τ (int16)       one launch, two passes per row (γ, then S)
            (the split K1a/K1b below is specified and checked, not used: [v1.2])
for each residue group G (g moduli; ⌈N/g⌉ groups):
  K2 pack   A → A'_l[λ][ī][k̂]  for l ∈ G   (gather via scatter vectors, smem transpose,
  K3 pack   B → B'_l[λ][j̄][k̂]  for l ∈ G    K-contiguous int8, zero padding)  re-read A, B
  K4 gemm   for each K-chunk c: CUTLASS batched int8 TN over z = s·b + λ (s = slot in G),
            epilogue store-mod (c = 0) / add-mod (c > 0) → u8 planes W_l[λ][ī][n̂]
K5 crt      per (λ,ī,j̄): flagged ? native dot (reads A, B) : exact CRT; unscale; α/β; scatter
```
If `α = 0` or `k = 0`, only K0 and K5 (β-only) run.

**[v1.4] Fused path KF (plan-time choice).** When the emulated GEMM runs and
`k̂ ≤ 512` (`fused`; a pure function of the lowering, reported by `imd_lowering`, §6a), one
launch **KF** replaces K0, K1 and group 1's K2/K3; then group 1's K4 launches, groups 2.. as
above (K2, K3, K4, reading the σ/τ and vectors KF stored), and K5:
```
KF          per pack tile (th rows × all of k̂, one tile row = one whole operand row) and
            moduli part y: offsets from the fused-mode descriptor (kernel argument, as K0);
            the tile into shared memory; per row the §3.1/§3.2 integers of K1 (flag, γ, S, σ:
            the same exact integers, reduced across the row's lanes); part 0 stores σ/τ and
            the flags; residues of part y's slots of group 1 (as K2/K3); then part-0 blocks
            zero RB rows n..n̂−1 of group 1 and, with IMD_VEC_GENERATE, write the vec region
```
Rules **[spec: Pipeline Fused, MParts; pos_fused_*, neg_fused_*]**:
- KF reads **nothing another block of KF writes**: its row and k̂ offsets are computed from
  the descriptor, never read from the vec region its own vec blocks write
  (`neg_fused_reads_vec`), and every moduli part computes σ itself from its tile instead of
  reading part 0's stored σ (`neg_fused_shared_sigma`).
- KF still produces every output of the kernels it replaces, for the later kernels: σ/τ and
  the flags (`neg_fused_no_sigma`: group 2's packs and K5 read junk) and the vec region
  (`neg_fused_no_vec`).
- KF packs **group 1 only**: the RA/RB slots are reused per group, so later groups are packed
  after the previous group's GEMMs retire, as before (`neg_fused_all_groups`).
- Exactly the §4a workspace and the same bits in every region (σ, flags, vec, RA, RB, W) as
  the unfused sequence; `α = 0` / `k = 0` calls run K0 + the β kernel as before.
- Why k̂ ≤ 512: a tile of 4096 FP64 holds `th = 4096/tk` whole rows (tk = pow2 ≥ k̂), and a
  row's `tk/16` converting threads are lanes of one warp, so K1's reductions are warp
  shuffles. The point is fewer, fuller launches for small problems (docs/NOTES-A1.md):
  K1 and the packs no longer read the operands separately and K0's launch disappears.

**[v1.4] Moduli parts.** A pack launch (K2/K3 and KF) with fewer tiles than about two per
SM splits the group's slots over `gridDim.y` block rows: part y of a row writes the slots
`[y·|G|/ny, (y+1)·|G|/ny)` and re-reads its tile. Disjoint outputs, same values
**[spec: Pipeline MParts, pos_pack_parts]**. K5 takes one or four j̄ per thread (a launch
choice; with four, a warp's results are transposed through shared memory so C stores are
coalesced). Neither changes any result.

**K0 and the scatter vectors [v1.2].** The plan option `vec_mode` chooses where the
vectors live; both modes give bitwise identical results.
- The scatter vectors are a pure function of the fused modes of §5 step 9 (per class I, J,
  K, L: extent and the stride in each tensor that carries it). The plan keeps that
  descriptor (O(rank) host bytes). At most 62 fused modes per class (each has extent ≥ 2
  and the class's extent product fits int64), so it is bounded.
- **`IMD_VEC_GENERATE` (default).** K0 is a kernel that receives the descriptor **by
  value** (its kernel argument) and writes exactly the §4a vec contents of §5 step 10 into
  the workspace, one thread per entry. No PCIe copy, no pinned host buffer, no plan-owned
  device memory: the workspace stays the only device memory a call uses, and no GPU work
  reads plan memory (§6 lifetime). `imd_workspace_size` is unchanged from v1.1.
- **`IMD_VEC_RESIDENT` (opt-in).** `imd_plan_create` allocates `8·(2m + 2n + 2k + 3b)`
  device bytes, fills them with the same kernel and waits for it; the kernels of every
  call read that copy, K0 is not launched, and the workspace has **no vec region** (§4a).
  `imd_plan_device_bytes` reports the plan's device bytes exactly. The plan must outlive
  the GPU work of its calls (§6).
- The v1.1 pinned host buffer + `cudaMemcpyAsync` is gone (measured slower than both
  modes on every benchmark shape; docs/NOTES-M4.md "v1.2").

**K1 in two launches [v1.2]: specified and model-checked, measured, NOT adopted.**
The split below is a valid alternative (the spec checks both); implemented, it was slower
on every benchmark shape (K1 +17 % at r_bigk, +79 % at sq8192: two full streaming reads
cost more than one read plus a re-read that mostly hits L2; docs/NOTES-M4.md "v1.2"), so
v1.2 keeps the one-launch K1.
- K1a reads every A row and B column once: the flags into fA/fB and the row exponent γ
  (§3.2 step 2, int16; zero rows marked) into sA/sB, used as **scratch**.
- K1b reads them again with γ from sA/sB, computes S and overwrites γ with σ (τ).
- Rules **[spec: Pipeline neg_k1_merged, neg_k1_no_sigma, neg_k1_sigma_row_split,
  neg_k1_pack_waits_gamma]**:
  - K1b starts after K1a retires (a kernel boundary; in one launch an S-pass CTA can read
    γ before it is written);
  - each row / column is owned by **one** K1b CTA, which reads its γ before it writes σ over
    it (two CTAs sharing a row would read σ as γ);
  - nothing but K1b reads γ: the packs and K5 read σ only after K1b retires (with a second
    stream, the packs' event edge is from K1b, not K1a).
- No extra workspace: γ ∈ [−1074, 1023] fits the int16 σ slots.

**Per-batch modulus in K4.**
- One batched launch covers several moduli. A plain `VisitorScalarBroadcast` cannot
  express `slot = z div b`, because that is not linear in z.
- The epilogue therefore reads `p` and the Barrett constants from a `__constant__` table
  indexed by `g0 + (z0 + z) / b` (z: the tile's batch in the launch, z0: the launch's first
  batch), once per tile.
- That table is at most 20 entries and needs no workspace.

**Padding.**
- Numerically only one operand needs zero K padding, and RB rows n…n̂−1 only feed W
  columns ≥ n, which K5 never reads.
- v1 still zeroes all padding, so every byte the GEMM reads is written in this call. That
  keeps runs `compute-sanitizer --tool initcheck` clean.

Properties checked by **[spec: Pipeline]**:
- **Exactly once.** Every C element is written exactly once, by K5, after all N planes are
  final (each plane is tagged with the K positions folded in: `PlanesFinalAtCRT`).
- **No stale reads.**
  - No buffer is read before it is written for the current group or chunk (the scatter
    vectors per tensor, as K0's CTAs write them).
  - With the (unused) K1 split: σ/τ are final before a pack or K5 reads them; K1b reads γ,
    never junk or σ (`SigmaFinalAtUse`) **[v1.2]**.
  - Residue buffers are reused only after the group's GEMMs retire.
  - Chunks partition `[0,k̂)`.
- **Exact layout.** Regions are disjoint, 256-aligned, gap-free per region, and equal to
  §4a (`LayoutExact`).
- **Read rules.** The β = 0 / α = 0 / k = 0 read rules of §3.7 hold.
- **Aliasing.** C may not alias A or B, because K5's native path reads A and B while other
  threads write C. The workspace may not alias A, B or C. Both are checked at execute
  time → ALIASING.
- **Streams.**
  - Two streams without events produce a hazard.
  - Safe two-stream operation needs three event edges: K1 → pack (RAW; from K1b with the
    split), pack(G) → gemm(G)
    (RAW), and last gemm(G) → pack(G+1) (WAR). With single-buffered RA/RB those edges
    serialize everything anyway.
  - So v1 is single-stream.

**Launch limits.**
- (A2: the persistent K4Gemm grid has no gridDim limits of its own; the two splits below
  were CUTLASS grid limits and are kept unchanged, so the launch structure, the host
  queries that count it and the Pipeline spec are the same.)
- The CUTLASS batch count `|G|·b` must be ≤ 65535 (gridDim.z), so plan time splits it.
- The N-tile count must be ≤ 65535 (gridDim.y), so n̂ is split across launches if needed.
  The last launch may end in a partial N tile; CUTLASS predicates columns past n̂, and
  the threadblock N (128 or 256 today) must be a multiple of 16 so every launch's B and W
  pointers stay 16-byte aligned. **[spec: Pipeline pos_ysplit*, neg_ysplit_*, neg_ytile_misaligned]**
- `m, n̂ ≤ 2³¹−1` (`GemmCoord` is int), and the per-launch K is ≤ K_c'.
- `lda = ldb = k̂` and `ldc = n̂`. The batch strides are int64, so k̂ itself may exceed 2³¹.

**GEMM configuration [CUT]:**
- Our kernel `K4Gemm` (`src/kernels/k4_gemm.cuh`, A2) around CUTLASS 2.x's threadblock
  main loop (`DefaultMma` → `MmaMultistage`, `arch::Sm80` on sm_89), batched.
- int8 `mma.sync.m16n8k32`, `OpMultiplyAddSaturate` (`.satfinite`; measured free against
  the wrapping form, docs/NOTES-A2.md).
- TN layout: A RowMajor, B ColumnMajor, both K-contiguous. Alignment is 16, hence
  `k̂ = ⌈k⌉₁₆` and `n̂ = ⌈n⌉₁₆`.
- Persistent CTAs (grid = SMs × occupancy) over the launch's tiles in swizzle-8 order;
  each CTA issues the next tile's cp.async prologue before the current tile's epilogue.
- Epilogue in registers: each thread reduces its accumulators with §3.5's exact reduction
  (chunk 0: `acc mod p`; later chunks: `(W + acc) mod p` with the old W bytes loaded and
  shuffled to the accumulator layout), packs u8, and a 4×4 word transpose within each lane
  quad gives each thread 16 consecutive bytes of a row (one 128-bit store). No shared
  memory, no barriers. (Until A2: CUTLASS `GemmUniversal` with an `Sm80EVT` epilogue —
  compute node, `VisitorAuxLoad<u8>`, `VisitorAuxStore<u8>` — kept in
  `kernels/mod_epilogue.cuh` and still tested against the same reference.)
- One tile: 128×256×64, 3 stages, 8 warps of 64×64 (72 KB smem, 1 CTA/SM). The menu
  below was measured (M4, A2) and is not instantiated: 256×128×64 s3 and 128×256×64 s4
  within noise; 128×128×64 s3 (2 CTAs/SM) L2-bandwidth-bound.
  Menu (99 KB of smem): 256×128×64 s3 or 128×256×64 s3; 128×128×64 s3–5;
  64×128×64 s6; 64×64×64 s6–10.
- cuBLASLt (int32 out) is used only as a test oracle and baseline.

**DIRECT path.**
- Native FP64, our own small kernel.
- Used for K-empty contractions (outer/Hadamard; k = 1 after lowering) and when
  `mode = DIRECT`.
- K-empty → DIRECT is **documented dispatch by descriptor**, not a hidden switch.
- No cuTENSOR/cuBLAS runtime dependency.

---

## 4a. Workspace layout (exact; [cert: cert_workspace], [spec: Pipeline LayoutExact])

```
align(x) = ⌈x⌉₂₅₆      k̂ = ⌈k⌉₁₆      n̂ = ⌈n⌉₁₆      g ∈ [1, N]

region      bytes                       contents
sA          2·b·m                       int16 shifts σ  (A rows)
sB          2·b·n                       int16 shifts τ  (B cols)
fA          1·b·m                       u8 non-finite flags (A rows)
fB          1·b·n                       u8 non-finite flags (B cols)
vec         8·(2m + 2n + 2k + 3b)       int64 scatter vectors (A: row,k,batch; B: k,col,batch;
                                        C: row,col,batch), written by K0
RA          g·b·m·k̂                     int8 residues of A for one group
RB          g·b·n̂·k̂                     int8 residues of B for one group
W           N·b·m·n̂                     u8 modular products, all N planes

workspace_size = Σ_regions align(bytes), regions in this order at 256-aligned offsets
               = 0                if b·m·n = 0 (empty output: no-op)
               = align(vec)       if mode = DIRECT, K-empty, k = 0
[v1.2] vec_mode = IMD_VEC_RESIDENT: the vec region is omitted everywhere (it is in the
       plan's device memory instead), so DIRECT, K-empty and k = 0 need 0 bytes.

plan device bytes (imd_plan_device_bytes) [v1.2]
               = 8·(2m + 2n + 2k + 3b)   if vec_mode = IMD_VEC_RESIDENT and b·m·n > 0
               = 0                       otherwise
```
**[spec: Pipeline LayoutExact, RegionsUsed, PlanBytesExact, pos_resident_*,
neg_resident_ws_vec, neg_resident_reads_ws]**
- The caller must pass `ws_bytes ≥ workspace_size`.
- The result includes the tail padding after W, up to 255 bytes.
- If any product or the total overflows uint64, the call returns INVALID_VALUE.
- Worked example: 8192³, N = 14, g = 14 needs 2,819,014,912 B (2.625 GiB).

---

## 5. IndexMap: label semantics and legality (v1)

Lowering, in order **[spec: IndexMap]**:
1. Check that extents agree per label.
2. **Repeated label within A or B** → diagonal view (sum its strides).
3. **Repeated label within C** → NOT_SUPPORTED (any extent).
4. Drop extent-1 modes.
5. Classify each label by its membership set: {A,C} → I, {B,C} → J, {A,B} → K, {A,B,C} → L.
6. **A label only in A** (or only in B) → K, with stride 0 in the other operand. This sums
   over it; k grows by its extent.
7. **A label only in C**:
   - extent > 1 → NOT_SUPPORTED. This is a v1 policy: broadcasting is exact
     (`pos_conly`), and a later version may allow it.
   - extent 0 → C is empty → no-op.
8. **Canonical orders**, a pure function of the descriptors. The coalescing role swap
   below is applied **first**; orders are then computed on the post-swap roles, with
   "position" taken from the caller's original label order:
   - I by (|s_C|, |s_A|, position); J by (|s_C|, |s_B|, position); L by (|s_C|, |s_A|,
     |s_B|, position).
   - K by (|s_big|, |s_other|, position), where "big" = A if m ≥ n, else B.
   - "Position" is the first occurrence in `A.labels ++ B.labels ++ C.labels`.
   - **Coalescing:** among C's modes with extent > 1, if the smallest-|stride| one is in I,
     swap the operand roles (compute Cᵀ = BᵀAᵀ: A↔B, I↔J) so that j̄ is C's fastest axis.
     The swap happens **before** §4a sizing: §4a's m and n are post-swap. That matters
     because only n is padded to n̂. The smallest |stride| is unique whenever rule 4 holds.
9. **Fusion (after step 8):** within each group, in canonical order, merge consecutive
   labels g, g′ iff `s_X(g′) = s_X(g)·e(g)` for **every** tensor X that carries the group.
   - That includes operands that carry it with stride 0 from step 6.
   - The v1 wording ("adjacent in every tensor where they appear") was **unsound**: it fused
     an A-only label with a K label **[spec: IndexMap neg_fusion]**.
10. Build int64 additive scatter vectors (TBLIS style): `off_X = vL[λ] + vRow[·] + vCol[·]`.

Legality is always checked (no debug-only asserts, unlike TBLIS). Statuses are reported in
this precedence order; the first failing check wins:

| # | Case | Status |
|---|---|---|
| 1 | extent mismatch for a label | INVALID_VALUE |
| 2 | repeated label in C | NOT_SUPPORTED |
| 3 | label only in C with extent > 1 | NOT_SUPPORTED |
| 4 | non-empty C fails the **sorted-\|stride\| rule**: over C's modes with extent > 1, sorted by \|stride\|, each \|s₍t₎\| > Σ_{u<t} \|s₍u₎\|(e₍u₎−1) (sufficient for injectivity) | INVALID_VALUE |
| 5 | overflow: extent products or spans exceed int64; workspace total exceeds uint64; in EMULATE mode with a non-empty output and k ≥ 2, m or n̂ > 2³¹−1 (CUTLASS limit; not applied to DIRECT, K-empty or k = 0) | INVALID_VALUE |
| 6 | C ∩ A or C ∩ B: byte intervals of the non-empty tensors, checked even when A or B is not read (α = 0 or k = 0); workspace ∩ A, B or C, using the bytes actually used, `[ws, ws + workspace_size)`. Conservative: interleaved but disjoint tensors are rejected (an exact lattice test is a possible later refinement) | ALIASING (execute time) |
| 7 | A, B or C not 8-byte aligned; workspace not 256-byte aligned | MISALIGNED (execute time) |

Allowed:
- strides of any sign (int64), including zero strides in A and B;
- A and B aliasing each other (e.g. `W = BᵀB`);
- an empty I, J or L, which is a no-op with 0 workspace;
- `k = 0`, which gives `C ← β·C` (§3.7).

**[spec: IndexMap]** checks, on tiny integer tensors (extents up to 3) and including
fusion at extent 3:
- classification is a partition;
- `Scatter ∘ GEMM ∘ Gather` generates exactly the einsum's multiset of (C, A, B) address
  triples;
- the canonical index encodings are bijections onto RA, RB and W, and the label decode that
  generates the scatter vectors is a bijection;
- the gather reads exactly the einsum's elements (many cells may read one source element
  under diagonals or stride-0 lowering);
- the sorted-stride rule implies injective C writes;
- the status precedence above holds;
- the canonical orders are sorted as specified.

Negative models:
- the C rule dropped;
- extent consistency dropped;
- the stride-0 lowering dropped;
- the literal v1 fusion rule;
- the repeated-C rule dropped;
- the aliasing test dropped;
- the C rule used as if it were necessary as well as sufficient.

---

## 6. API (C ABI, ctypes-friendly; Python package later)

```c
typedef struct { int32_t rank; const int64_t* extent; const int64_t* stride; /* elements */
                 const int32_t* label; } imd_tensor_t;
typedef struct { int32_t num_moduli;      /* N, 2..20                        */
                 int32_t residue_group;   /* g, 0 = N                         */
                 int32_t rounding;        /* IMD_RN (default) | IMD_TRUNC     */
                 int32_t mode;            /* IMD_EMULATE | IMD_DIRECT         */
                 int32_t vec_mode;        /* [v1.2] IMD_VEC_GENERATE (0, default)
                                             | IMD_VEC_RESIDENT                */
               } imd_options_t;

imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                const imd_tensor_t* C, const imd_options_t*, size_t* bytes);
imd_status_t imd_plan_create(imd_handle_t, const imd_tensor_t* A, const imd_tensor_t* B,
                             const imd_tensor_t* C, const imd_options_t*, imd_plan_t*);
imd_status_t imd_contract(imd_plan_t, double alpha, const double* A, const double* B,
                          double beta, double* C, void* ws, size_t ws_bytes, cudaStream_t);
imd_status_t imd_create(imd_handle_t*);           /* lifecycle */
imd_status_t imd_destroy(imd_handle_t);
imd_status_t imd_plan_destroy(imd_plan_t);
imd_status_t imd_plan_device_bytes(imd_plan_t, size_t* bytes);   /* [v1.2] §4a */
const char*  imd_status_string(imd_status_t);
```

- **[v1.2] API additions** (approved): the `vec_mode` option and `imd_plan_device_bytes`.
  `imd_options_t` grows by one int32 (20 bytes); a zero-initialized field is the default,
  so `{N, g, rounding, mode}` initializers keep their meaning (`-Wextra` now warns about
  the missing initializer), but callers built against the v1.1 header must be rebuilt
  (the struct size changed). An unknown `vec_mode` is
  INVALID_VALUE (in `imd_workspace_size` too, which depends on it).

- `imd_workspace_size` is a **pure host function**: no GPU, no allocation, and exactly
  §4a. **[v1.3]** It and the other host queries (§4a table, plan bytes, lowering summary)
  live in the CUDA-free `libimmerdink_host` (§6a), which libimmerdink links.
- Statuses: SUCCESS, INVALID_VALUE, NOT_SUPPORTED, ALIASING, INSUFFICIENT_WORKSPACE,
  MISALIGNED, ALLOC_FAILED (host allocation in plan creation), CUDA_ERROR.
- `imd_plan_create` reports the §5 statuses and the RN acceptance rule (§3.2: κ ≥ T →
  NOT_SUPPORTED). That rule applies only when the emulated GEMM would run;
  `imd_workspace_size` is exactly §4a and never applies it.
- `imd_contract` checks, in order:
  1. null pointers → INVALID_VALUE. C must always be non-null. A and B must be non-null
     only when α ≠ 0 and k ≠ 0, and ws only when `workspace_size > 0`. The rule uses the
     plan's k, α and `workspace_size` only, not whether the output is empty or whether
     this call would actually touch the workspace (e.g. α = 0 with `workspace_size > 0`
     still needs a non-null ws). Conservative by design;
  2. an empty output → SUCCESS with nothing touched;
  3. ALIASING (§5 row 6);
  4. MISALIGNED;
  5. `ws_bytes < workspace_size` → INSUFFICIENT_WORKSPACE.
- `imd_contract` does no host synchronization and no allocation, and has no hidden mode
  switches. (Tested: an LD_PRELOAD shim counts CUDA allocations; guard zones around the
  workspace.)
- Test-only hooks (launch-limit overrides, lowering dump) are not part of the production
  library: they are built only with `IMMERDINK_BUILD_TESTS=ON`. The launch-limit override
  applies to plans created after it (the K4 launch split is fixed at plan creation)
  **[v1.2]**; the hooks are not thread-safe.
- A workspace belongs to at most **one** in-flight `imd_contract` at a time.
- **Lifetime [v1.2]:**
  - `IMD_VEC_GENERATE`: a plan may be destroyed as soon as every `imd_contract` that uses
    it has **returned**; the GPU work already enqueued reads no plan memory (K0's
    descriptor and K4's parameters are kernel arguments, copied at launch).
  - `IMD_VEC_RESIDENT`: the kernels read the plan's device vectors, so the plan must
    outlive the GPU work of every call that uses it (synchronize those streams before
    `imd_plan_destroy`), as v1.1 required for its pinned buffer.
  - The handle must outlive every `imd_plan_create` that uses it.
  - **[spec: HostAPI pos_shared_plan, neg_witness_early_destroy, pos_resident,
    neg_resident_early_destroy, neg_pinned_early_destroy]**
- **`imd_plan_device_bytes`** reports the device bytes the plan owns, exactly §4a's "plan
  device bytes" (the bytes requested from `cudaMalloc`; the driver's allocation granularity
  is not included). With the workspace and the user tensors this is all the device memory
  immerdink uses (MEMORY_MODEL §3).
- **ALLOC_FAILED** covers host allocation failure in plan creation and, with
  `IMD_VEC_RESIDENT`, the plan's device allocation; any other CUDA failure, including plan
  creation's device-property and kernel-attribute calls, is CUDA_ERROR. **[v1.2: no
  pinned allocation any more]**
- **Threads [v1.2] [spec: HostAPI]:**
  - `imd_plan_create` does all host-side GEMM setup: it reads the SM count and the K4
    kernels' occupancy, sets their dynamic shared-memory attribute (CUDA runtime calls,
    thread-safe), and stores the parameters of every K4 launch shape in the plan.
    `imd_contract` copies them, patches this call's pointers and launches; it never calls
    CUTLASS's operator initialization and writes no plan or process-global host state.
  - Hence any number of threads may call `imd_workspace_size`, `imd_plan_create` and
    `imd_contract` concurrently, on distinct plans or on **one shared plan**, on any
    streams, provided:
    1. every in-flight `imd_contract` has its own workspace. Two concurrent calls on one
       workspace are unsafe even on one stream: their launches interleave on it
       (`neg_ws_shared_same_stream`). Calls ordered on the host (one thread, or
       externally synchronized) on one stream may reuse a workspace;
    2. no in-flight call writes memory another in-flight call reads or writes (its C vs
       the other's A, B, C), as for any stream-ordered library;
    3. a plan is not destroyed while a call that uses it is running on another thread;
    4. the calling thread's current device is the one the plan was created on.
  - Each rule is shown necessary by a negative model: a parameter cache filled by the
    first `imd_contract` races (`neg_lazy_cache`); one mutable operator per plan lets a call
    launch another call's arguments (`neg_shared_op`); lazily initialized process-global
    state races even with a plan per thread (`neg_global_lazy`).
  - (v1.1 said concurrent first calls were unsupported because CUTLASS caches device
    properties in unsynchronized statics. CUTLASS v4.8.0 declares them `thread_local`
    (`CUTLASS_THREAD_LOCAL`), so that race did not occur (`pos_v11`); v1.2 no longer
    depends on it.)
  - Not thread-safe: the test hooks and the benchmark-only phase marks
    (`IMMERDINK_PROFILE`).
- **Kernel-level scope and stream capture [v1.2].** immerdink moves no data between host
  and device in `imd_contract` (no per-call copy at all in `IMD_VEC_GENERATE`), owns no
  streams, events or graphs, and does no internal tiling; the caller orchestrates.
  `imd_contract` only launches kernels on the caller's stream, so it may be captured into a
  CUDA graph (`cudaStreamCaptureModeGlobal`) and replayed: the graph rewrites everything it
  reads in the workspace (tested in both vec modes: `tests/test_graph.cpp`). A replayed
  graph keeps the kernel arguments of the capture (pointers, α, β), and a resident plan
  must outlive every replay. (`imd_plan_create` with `IMD_VEC_RESIDENT` builds its vectors
  on `cudaStreamPerThread` and waits for them; plan creation is not capturable.)
- Callers tile I/J/L themselves, since sub-views are valid inputs. Tiling over I/J/L leaves
  results bitwise unchanged; tiling over K by the caller does change them. The library
  handles K chunking and the batch-launch split internally.
- **Reserved, not v1** (the API and specs must not preclude these):
  - a packed-operand API `imd_pack` / `imd_contract_packed`. Fast-mode σ depends only on
    the operand itself, so packs are reusable across contractions with the same K labels,
    N and rounding;
  - complex values via the 2M method (2 GEMMs per modulus, moduli with √−1);
  - accurate-mode scaling, which is incompatible with independent packing.
  - adaptive precision, possibly via an Ozaki-I slice mode sharing this pipeline (PLAN, Later).

---

## 6a. Host query API (libimmerdink_host, CUDA-free) [v1.3]

Callers that own memory and orchestration (e.g. ehrenfest) must know exactly what
immerdink will need **without a GPU**: workspace bytes per (descriptors, N, g, vec_mode),
the §4a region table, plan device and host bytes, and the lowering a performance model
would consume. This section **only exposes** §3.2 (RN acceptance), §4 (dispatch, launch
structure), §4a (layout, plan device bytes) and §5 (lowering, legality); they remain the
specification, and no semantics change. The TLA+ specs are therefore unchanged
(`IndexMap`, `Pipeline LayoutExact / PlanBytesExact`, `HostAPI` cover the same functions,
now in another library).

**Single source of truth.** `libimmerdink_host.so` (plain C++17, no CUDA headers, no CUDA
libraries) holds the only implementation of the argument checks, the §5 lowering and
legality, the §3.2 plan-time constants and RN acceptance, the §4a layout, the K4 launch
structure and the plan sizes. `libimmerdink.so` links it (DT_NEEDED) and builds every plan
from it: `imd_plan_create` = host analysis + plan data, then only the device parts (SM
count, K4 kernel parameters, the resident vectors, allocated with exactly the host
library's plan device bytes). `imd_workspace_size` and `imd_status_string` moved into the
host library (same declarations, same header `immerdink.h`, which now includes
`immerdink_host.h`). `spec/params.py` and `spec/oracle.py` are not an API; they stay the
independent check of the C (tests compare them).

```c
#include "immerdink/immerdink_host.h"        /* no CUDA header; also included by immerdink.h */

uint32_t     imd_host_api_version(void);      /* IMD_HOST_API_VERSION of the loaded library */
imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                const imd_tensor_t* C, const imd_options_t*, size_t* bytes);
typedef struct { uint64_t offset, bytes; } imd_region_t;      /* bytes 0: region absent */
imd_status_t imd_workspace_layout(const imd_tensor_t* A, const imd_tensor_t* B,
                                  const imd_tensor_t* C, const imd_options_t*,
                                  imd_region_t* regions, int32_t capacity,
                                  int32_t* num_regions, size_t* total);
const char*  imd_region_name(int32_t region);  /* IMD_REGION_SA .. IMD_REGION_W: "sA" .. "W" */
imd_status_t imd_plan_bytes(const imd_tensor_t* A, const imd_tensor_t* B,
                            const imd_tensor_t* C, const imd_options_t*,
                            size_t* device_bytes, size_t* host_bytes);
imd_status_t imd_lowering(const imd_tensor_t* A, const imd_tensor_t* B,
                          const imd_tensor_t* C, const imd_options_t*, imd_lowering_t* out);
const char*  imd_status_string(imd_status_t);

typedef struct {               /* 136 bytes in version 1, 152 in version 2 [v1.4] */
  uint32_t struct_size;        /* in: sizeof(imd_lowering_t) as the caller compiled it  */
  int32_t  status, lowered, dispatch, swapped, num_moduli, residue_group, num_groups;
  int64_t  m, n, k, b, n_hat, k_hat, vec_entries, k_chunks, gemm_launches;
  uint64_t kappa, workspace_bytes, plan_device_bytes, plan_host_bytes;
  int32_t  fused, reserved;    /* version 2 [v1.4] */
  int64_t  kernel_launches;    /* version 2 [v1.4] */
} imd_lowering_t;
enum { IMD_DISPATCH_NOOP, IMD_DISPATCH_EMULATE, IMD_DISPATCH_DIRECT, IMD_DISPATCH_SCALE };
```

Semantics:
- **`imd_workspace_size`**: unchanged (§4a, §6): statuses SUCCESS, INVALID_VALUE,
  NOT_SUPPORTED from the argument checks and §5 rows 1–5; never the RN acceptance rule.
- **`imd_workspace_layout`**: the same status; on SUCCESS the §4a table as data:
  `regions[r]` for `r < min(capacity, IMD_REGION_COUNT)` in §4a order (sA sB fA fB vec RA
  RB W), each `{offset, bytes}` with `bytes = 0` for a region absent from this call (empty
  output, DIRECT / K-empty / k = 0, the vec region with `IMD_VEC_RESIDENT`);
  `*num_regions = IMD_REGION_COUNT`, `*total = imd_workspace_size`. `capacity < 0`, or
  `capacity > 0` with `regions = NULL` → INVALID_VALUE.
- **`imd_plan_bytes`**: returns the **plan-time status**, i.e. what `imd_plan_create`
  returns barring CUDA and allocation failures: the `imd_workspace_size` checks, then §3.2
  RN acceptance (κ ≥ T with the emulated GEMM → NOT_SUPPORTED). On SUCCESS:
  - `device_bytes` = §4a "plan device bytes" = `imd_plan_device_bytes` of that plan
    (the bytes requested from `cudaMalloc`);
  - `host_bytes` = the plan's **host payload**: `sizeof` the plan record + 20 bytes per
    descriptor mode (extent, stride, label copies) + the lowering tables (per distinct
    label, per label in the canonical orders, per fused mode) + the K0 fused-mode
    descriptor (unless NOOP) + on EMULATE the K4 parameter block and one CUTLASS parameter
    set per distinct K4 launch shape (store-mod and add-mod). Heap-allocator and container
    bookkeeping are not included (as the driver's granularity is not in the device bytes).
    It is a few KB; the exact `sizeof` values are those of the build, so it is reported by
    the library rather than written into this document.
  Either pointer may be NULL; outputs are written only on SUCCESS.
- **`imd_lowering`**: fills the summary and returns the same plan-time status (also in
  `status`). `out == NULL` or `out->struct_size < 136` → INVALID_VALUE with nothing
  written. `lowered = 1` iff the `imd_workspace_size` checks passed; then `m, n, k, b`
  (post-swap, §1.1), `swapped`, `n_hat`, `k_hat`, `kappa` (⌈√k⌉, 0 for k = 0),
  `num_moduli`, `residue_group` (g normalized to [1, N]), `dispatch`, `vec_entries`
  (2m + 2n + 2k + 3b, 0 on NOOP), `workspace_bytes`, and on EMULATE `num_groups` = ⌈N/g⌉,
  `k_chunks` (K4 K-chunks per group, ⌈k̂/K_c'⌉) and `gemm_launches` (K4 kernel launches per
  `imd_contract`, with the batch ≤ 65535 and n̂ ≤ 65535·256 splits) are valid even when the
  status is NOT_SUPPORTED (κ ≥ T), so a caller sees why. `plan_device_bytes` and
  `plan_host_bytes` are valid only on SUCCESS. With `lowered = 0` every field except
  `struct_size`, `status` and `lowered` is 0.
  - **[v1.4] Version 2** appends `fused` (1 iff EMULATE with k̂ ≤ 512: the §4 KF launch) and
    `kernel_launches`, the kernel launches of one `imd_contract` with α ≠ 0: NOOP 0; DIRECT
    and SCALE `v + 1`; EMULATE `(fused ? 1 : v + 2) + (⌈N/g⌉ − fused)·(2 + [n̂ > n]) +
    gemm_launches + 1`, with `v` = 1 for `IMD_VEC_GENERATE` (K0), 0 for resident vectors (a
    call with α = 0 launches `v + 1`). Both are valid whenever `lowered = 1`, like
    `k_chunks`. A version-1 caller (struct_size 136) gets exactly the version-1 fields.
    Tested: the LD_PRELOAD shim counts `cudaLaunchKernel` during every test call and
    compares it with the plan, and the plan with the host query.
  - `dispatch`: NOOP (b·m·n = 0), EMULATE (K0–K5; EMULATE mode, k ≥ 2), DIRECT (K0 + the
    native kernel; DIRECT mode or K-empty), SCALE (k = 0: K0 + the β kernel, §3.7). A call
    with α = 0 also runs only K0 and the β kernel; that is per call, not per plan.
- **Statuses** are those of §6; the queries can return SUCCESS, INVALID_VALUE,
  NOT_SUPPORTED and (host allocation of the O(rank) lowering tables) ALLOC_FAILED.
- **No GPU, no CUDA runtime, no driver.** The functions are **pure** (outputs depend only on
  the arguments; no allocation visible to the caller, no global or static mutable state)
  and **thread-safe** (any number of threads, concurrently with each other and with
  libimmerdink). They read the descriptor arrays only during the call.
- The K4 launch structure used for `k_chunks`, `gemm_launches` and the host bytes is the
  production one (K_c' chunks, 65535 batches, 65535 N-tiles of 256); the test-only
  launch-limit hook changes only libimmerdink's plans.

**The workspace/speed knob is g only.** Callers query sizes for each g ∈ [1, N] and choose;
there is no budget option (BENCHMARKS "Residue-group size g" gives the speed side).

**ABI stability.**
- Plain C: int32/int64/uint64/size_t/pointers, no bitfields, unions or enums in structs;
  `imd_region_t` is 16 bytes, `imd_lowering_t` 136 bytes in version 1.
- `IMD_HOST_API_VERSION` (header) and `imd_host_api_version()` (library) start at 1 and
  grow by one whenever a function, an enum value or a trailing `imd_lowering_t` field is
  added. Nothing that exists changes meaning, signature or layout; a change that would is
  a new function name.
- `imd_lowering_t` is versioned by `struct_size`: an older caller's smaller struct gets the
  fields it knows; a newer caller's larger struct keeps its unknown trailing fields
  untouched. Callers check `imd_host_api_version() >= ` the version they need.
- `imd_tensor_t` and `imd_options_t` are §6's; `imd_options_t` has no size field (v1.2 grew
  it; a further change would need a rebuild, as v1.2 did).
- Internal C++ entry points of the host library (lowering, layout, plan data) are exported
  for libimmerdink and the tests only; they are not in `include/` and carry no ABI promise
  (both libraries are built together).

**Python.** `python/immerdink/host.py` is a thin ctypes wrapper over these functions
(nanobind later), with an einsum-style front end (`"kjbc,kiac->abij"` + extents + optional
strides) and raw descriptors; it implements no sizing logic. Library lookup: argument,
`$IMMERDINK_HOST_LIB`, `$IMMERDINK_LIB_DIR`, the build directory, the system path.

**Tests** (`tools/test.sh`): `host_c_api` (C11, linked against libimmerdink_host only);
`workspace_vs_params` (all §4a vectors plus a g = 1..N × vec_mode sweep vs `params.py`:
status, layout, plan device bytes, lowering); `host_python` (wrapper == direct C calls ==
`params.py` + `oracle.py`, including the κ ≥ T boundary and K4 splits);
`host_vs_plan` (real plans: GEMM view, layout, device bytes, K4 parameter sets and
launches, host payload counted from the live containers == the host queries);
`host_cuda_free` (no CUDA in readelf/ldd/nm; the repository builds and passes its host
tests with `IMMERDINK_HOST_ONLY=ON`, the system compiler and no nvcc on PATH).

---

## 7. What was considered and rejected

| Option | Why not |
|---|---|
| GETT-style code generation | Residues must be rewritten anyway, so packing makes the layout free (TBLIS-style). GETT would add an autotuner for nothing. |
| Optimized CPU kernels | The dev box has no AVX-512/AMX, and emulation loses to native DGEMM there. The CPU's job is the exact Python oracle. |
| CUTLASS 3.x / CuTe collectives | No sm80/sm89 builders, and no SM80 EVT in 3.x ([CUT]). |
| V0 int32 planes | 4× the bytes per plane, for nothing, once the epilogue reduces mod p. |
| Register-fused CRT (V3) | Exceeds 255 registers/thread at useful tiles (MEMORY_MODEL §6). |
| FP CRT ([OZ2] double-double) | Exact integer arithmetic is simpler, checkable, and division-free. |
| Accurate mode in v1 | Needs the other operand (Ā·B̄), which breaks operand packing. Revisit for inputs with very wide dynamic range. |
| FP8 / INT4 paths | FP8 can't hold residues exactly at speed on Ada. INT4 forces p ≤ 16. |

---

## Changes v1 → v1.1 (from the spec phase)

- **§3.8 error contract.** It now carries the factor s; the old inequality was false.
  TRUNC contract added.
- **§3.2.** Terms are computed from integers; κ is the exact ceiling; the width and
  floor-shift rule for computing u are stated; RN soundness is separated from the accuracy
  policy; the exact σ range is given; the dead `S = 0` clause is removed.
- **§3.5.** The epilogue mod is non-negative and exact over the stated range; the
  `umulhi64` realization is specified; chunking is not data-sized; every modulus is
  chunked.
- **§3.6.** The word is unsigned, with a limb carry chain; the range of z is stated.
- **§3.7.** Sign of zero, α = 0 / k = 0 semantics, the native accumulator and order, and the
  single fma are all specified.
- **§4.** Added the K0 vector copy and a per-batch modulus table in `__constant__` memory.
  Stated the padding rationale, launch strides, workspace aliasing, and the one-call-per-
  workspace rule.
- **§4a.** Added the overflow status, the `ws_bytes ≥ size` contract, and that K-empty
  dispatches to DIRECT.
- **§5.** The fusion rule was rewritten (the old one was unsound) and moved after ordering.
  Fully specified the canonical orders and the coalescing role swap; added status
  precedence, the empty-C exemption, and C-only extent 0.

## Changes v1.1 → v1.2 (after M4; docs/NOTES-M4.md "v1.2")

- **§4 K0, §4a, §6 `vec_mode`.** Default `IMD_VEC_GENERATE`: a kernel generates the
  scatter vectors from the fused-mode descriptor (a kernel argument) instead of copying
  them from the plan's pinned buffer: no per-call PCIe copy, no pinned or device memory in
  the plan, §4a unchanged. Opt-in `IMD_VEC_RESIDENT`: the plan owns a device copy (exactly
  reported by the new `imd_plan_device_bytes`), K0 is skipped and the workspace has no vec
  region. The pinned copy is removed. **[spec: Pipeline neg_k0_no_c, pos_resident_*,
  neg_resident_*, PlanBytesExact; HostAPI pos_resident, neg_resident_early_destroy]**
- **§4 K1.** Two launches, K1a (flags, γ into sA/sB as scratch) and K1b (S, σ over γ), are
  specified with the rules that make the scratch reuse safe, but **not adopted**: measured
  slower on every shape. **[spec: Pipeline K1Split, SigmaFinalAtUse, pos_k1_split*,
  neg_k1_*]**
- **§4 GEMM setup / §6.** K4's parameters are built at plan creation and only copied by
  `imd_contract`. The thread-safety rules are stated precisely; the plan lifetime rule is
  relaxed to "until every call using it has returned" (generate mode). ALLOC_FAILED no
  longer covers a pinned allocation. Public API: `imd_options_t.vec_mode` and
  `imd_plan_device_bytes` added (user-approved); nothing else changes. **[spec: HostAPI]**

## Changes v1.2 → v1.3 (external memory API)

- **§6a (new).** The lowering, legality, RN acceptance, §4a layout, K4 launch structure and
  plan sizes move into the CUDA-free `libimmerdink_host` (C ABI, `immerdink_host.h`),
  which `libimmerdink` links and uses for every plan: one implementation. New queries:
  `imd_host_api_version`, `imd_workspace_layout`, `imd_region_name`, `imd_plan_bytes`,
  `imd_lowering`. `imd_workspace_size` and `imd_status_string` move to the host library
  with unchanged declarations. No computed value, status or kernel changes, so the TLA+
  specs are unchanged; §3.2, §4, §4a and §5 remain the specification.

## A2 (kernel-internal; no semantic, layout, API or spec change)

- **§4 K4.** The CUTLASS `GemmUniversal` + `Sm80EVT` kernel is replaced by `K4Gemm`: the
  same CUTLASS main loop in a persistent kernel with a register-only epilogue (see "GEMM
  configuration"). Same launches (host/k4_launches.hpp), same §3.5 arithmetic on the same
  values (the umulhi64 of cert_barrett is evaluated from 32-bit words), bit-identical W.
  No cert, spec or Apalache bound changes. Measurements: docs/NOTES-A2.md.

## Changes v1.3 → v1.4 (small-problem overhead, docs/NOTES-A1.md)

- **§4 KF.** A plan with the emulated GEMM and k̂ ≤ 512 runs K0 (generate mode), K1 and
  group 1's K2/K3 as one launch, which computes its offsets from the descriptor and σ per
  block, and stores σ/τ, the flags, the vec region and group 1's residues exactly as the
  separate kernels did. Results and the §4a workspace are unchanged. **[spec: Pipeline
  Fused; pos_fused_g3, pos_fused_g1, pos_fused_g2_parts, pos_fused_resident,
  pos_fused_batch2; neg_fused_reads_vec, neg_fused_shared_sigma, neg_fused_no_vec,
  neg_fused_no_sigma, neg_fused_all_groups]**
- **§4 moduli parts and K5 mapping.** Pack launches may split a group's slots over
  `gridDim.y`; K5 takes 1 or 4 j̄ per thread. Launch choices only. **[spec: Pipeline MParts;
  pos_pack_parts]**
- **§6a.** `IMD_HOST_API_VERSION` 2: `imd_lowering_t` grows to 152 bytes with `fused` and
  `kernel_launches` (appended; version-1 callers unaffected). libimmerdink_host stays the one
  implementation of the choice (`fused_path`) and the count (`plan_launches`).
