# immerdink — design (v1.13)

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
the host API (§6a); changed text is marked **[v1.4]**. v1.5 adds a small set of K4 tile
configurations chosen per plan (§4; bit-identical results, same launches and workspace), the
optional plan-time autotuning option `k4_tune` (§6) and the choice in the host API (§6a), and
takes the fused launch KF up to k̂ ≤ 1024 (§4); changed text is marked **[v1.5]**. v1.6 adds the
per-operand fused scale-and-pack launch KL for k̂ > 1024 (§4, chosen at plan time; same results,
same workspace, one DRAM read of the operand less) and reports it in the host API (§6a);
changed text is marked **[v1.6]**. v1.7 (§6b, D1 packed operands; design approved by Eric,
2026-10-03, all §6b.10 recommendations) is implemented (host API version 5; §6b.11 lists the
implementation choices and one clarification of §6b.5 to confirm). v1.8 adds §6c, the
performance model (ROADMAP B2: host API version 6; no semantic change). v1.9 adds §6d,
the Python front end over the C ABI (ROADMAP B1; no semantic, layout or C API change).
v1.10 extends §6c (host API version 7, calibration format 2): per-kernel wave and occupancy
features from launch rules shared with the kernels, BURST / SUSTAINED clock regimes with a
power-cap model (default SUSTAINED), COLD / WARM cache state (default COLD; the hold-out
accuracy is measured in the REPEAT state, COLD rests on a 6-shape cache sweep), prediction
intervals and rank ties (default order: time) (no semantic, layout, launch or spec change;
merged after v1.11, numbered as planned).
v1.11 adds §6e, fast host queries from Python for hot paths (`immerdink.planner`; no
semantic, layout, C API or spec change; v1.10 is the parallel performance-model revision).
v1.12 adds §6f, accuracy-aware planning (host API version 8): rigorous and typical error
estimates per configuration from caller-supplied operand statistics, `imd_err_min_moduli`, and
an error target in `imd_perf_rank` (no semantic, layout, launch or spec change).
v1.13 starts §4b, K4 load alignment. Its stage 1 sets the K-chunk length to `K_c' = 131008 =
⌊K_c/64⌋·64` (it was 131056), so every full chunk is a whole number of K4's 64-wide K tiles.
Results stay bit-identical; there is no layout, pack format or API change. Only the chunk
count of k̂ ∈ (j·131008, j·131056] changes (+1); changed text is marked **[v1.13]**.
The change lists are at the end.

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
| `K_c'` | **[v1.13]** `131008 = ⌊K_c/64⌋·64 = 2047·64` is the chunk length used (it was `131056 = ⌊K_c/16⌋·16`). Numerically any chunk length ≤ K_c is exact. A multiple of 16 keeps CUTLASS pointers 16-byte aligned (correctness). A multiple of 64, K4's threadblock K tile, keeps every full chunk free of a residue tile, so a row's K4 load phase depends on k̂ alone (§4b, performance). 131008 is the largest such multiple ≤ K_c **[cert: cert_kc, cert_row_phase]**. |
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
  - **[v1.13]** For p = 256 the range is `[−2129666048, 2146435327]` (−128·127·K_c' and
    128²·K_c' + 255); it fits int32 **[cert: cert_kc]** **[spec: apalache Epilogue]**. With
    the old 131056 the maximum was 2147221759.
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
`params.py --table`. **[v1.12]** It can also be left to the host library: §6f computes this
bound (and a typical estimate) per configuration from the caller's operand statistics, and
`imd_err_min_moduli` / `imd_perf_rank` return the cheapest N that meets an error target.

---

## 4. Execution pipeline (one stream, in order)

```
plan time:  IndexMap lowering → fused-mode descriptor, canonical orders, chunking, launch
            split; K4 configuration (rule, or timed with k4_tune) [v1.5] and its GEMM
            parameters (CUTLASS device properties, occupancy, smem attribute)
K0 vec      kernel: generate the scatter vectors → ws.vec from the fused-mode descriptor
            (a kernel argument, by value); not launched with IMD_VEC_RESIDENT      [v1.2]
K1 scale    A rows, B cols: flags + σ, τ (int16)       one launch, two passes per row (γ, then S)
            (the split K1a/K1b below is specified and checked, not used: [v1.2])
            [v1.6] per operand: K1 of that operand, or KL (below: K1 + its group-1 pack)
for each residue group G (g moduli; ⌈N/g⌉ groups):
  K2 pack   A → A'_l[λ][ī][k̂]  for l ∈ G   (gather via scatter vectors, smem transpose,
  K3 pack   B → B'_l[λ][j̄][k̂]  for l ∈ G    K-contiguous int8, zero padding)  re-read A, B
  K4 gemm   for each K-chunk c: CUTLASS batched int8 TN over z = s·b + λ (s = slot in G),
            epilogue store-mod (c = 0) / add-mod (c > 0) → u8 planes W_l[λ][ī][n̂]
K5 crt      per (λ,ī,j̄): flagged ? native dot (reads A, B) : exact CRT; unscale; α/β; scatter
```
If `α = 0` or `k = 0`, only K0 and K5 (β-only) run.

**[v1.4] Fused path KF (plan-time choice).** When the emulated GEMM runs and
`k̂ ≤ 1024` (**[v1.5]**; 512 in v1.4) (`fused`; a pure function of the lowering, reported by `imd_lowering`, §6a), one
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
- Why k̂ ≤ 1024: a tile of 4096 (or, for small operands, 1024) FP64 holds `th` whole rows
  (tk = pow2 ≥ k̂, `th·tk` = the tile; th ≥ 1), so a block owns its rows; their per-row K1
  reductions are warp shuffles plus, for rows spread over 2–8 warps, a shared-memory
  combine. The point is fewer, fuller launches for small problems (docs/NOTES-A1.md): K1 and
  the packs no longer read the operands separately and K0's launch disappears. **[v1.5]**
  1024 instead of 512 (A3, docs/NOTES-A3.md: sq1024-class shapes): tk = 1024 is the largest
  row a 4096-value tile holds 4 of, with KF's per-row offsets (8 KB) inside the 48 KB of
  static shared memory. The threshold is a launch-shape property only: Pipeline's `Fused`
  model has no k̂ in it (a KF CTA owns whole rows for any k̂), so the spec is unchanged.

**[v1.6] Fused large path KL (plan-time choice, per operand).** Without KF (k̂ > 1024), K1
reads every operand row twice (γ, then S; the second pass mostly from L2) and group 1's pack
reads it again from DRAM, so a large operand streams from DRAM twice before its first GEMM
(and once more per later group). For each operand x (the GEMM's A or B, post-swap) whose
`fused_large` bit is set (a pure function of the lowering, reported by `imd_lowering`, §6a),
one launch **KL_x** replaces K1 of x and group 1's pack of x (for B including the launch that
zeroes RB rows n..n̂−1, which KL_B does itself). K0 runs as before (generate mode); the other
operand keeps its K1 launch and its group-1 pack; groups 2.. are packed by K2/K3 from the σ/τ
KL stored, as after KF:
```
K0 vec      (unchanged)
per operand x = A, B:   fused_large(x) ? KL_x : K1 of x
for each residue group G:
  K2 / K3   for every operand, except group 1's pack of a KL operand
  K4 gemm   (unchanged)
K5 crt      (unchanged)

KL_x        per row tile (th rows × all of k̂; th·tk = 4096 FP64, tk = pow2 ≤ 4096; a row
            spans nc = ⌈k̂/tk⌉ chunks of tk): ONE CTA owns whole rows
  pass 1    chunks c = 0..nc−1 gathered through the scatter vectors (ws.vec or the plan's
            copy; h·stride when the K class is one fused mode) into shared memory, coalesced
            as in K2/K3; per thread E consecutive k̄ of one row (E = 8 with 512 threads, 2
            CTAs per SM, for nc ≤ 2; E = 4 with 1024 threads, 1 CTA per SM, for nc > 2):
            flag, γ; chunk 0 kept in registers, chunk nc−1 stays in shared memory
  reduce    per row over its tk/E threads (warp shuffles + a shared-memory combine): flag,
            γ = the §3.2 row exponent
  pass 2    S over chunk 0 (registers), chunk nc−1 (shared memory), then chunks nc−2 … 1
            re-read (most recently read first: from L2); per row σ = largest_u(S) − γ: the
            same exact integers as K1 (max, or, u128 sum: order-independent)
  pass 3    §3.3/§3.4 residues of every slot of group 1 (K-contiguous 8- or 4-byte stores
            per thread, coalesced per warp): chunk 0 (registers), the chunk in shared memory,
            then the others re-read ascending (from L2)
  store     σ/τ and the flag (one thread per row); KL_B also zeroes RB rows n..n̂−1 of
            group 1 (grid-strided, after its rows)
```
Rules **[spec: Pipeline FusedLarge; pos_large_*, neg_large_*]**:
- KL reads the scatter vectors from the vec region, so **K0 retires before KL starts** (a
  separate launch; `neg_large_with_k0`). Unlike KF it does not compute offsets from the
  descriptor: K0's grid is negligible for large problems, and a vec read is one cached load
  per k̄ instead of a mixed-radix decode.
- **One CTA owns each row**: the CTA that packs a row computes its σ from the whole row and
  never reads σ written in the same launch (`neg_large_row_split`: a long row split over two
  CTAs that pass σ through sA reads junk). A row cannot be split across CTAs without a
  grid-wide combine between pass 2 and pass 3 (S needs γ of the whole row first, and the
  ceiling terms of §3.2 cannot be rescaled from a per-segment γ); KL does not do that.
- KL stores σ/τ and the flags for the later groups' packs and K5 (`neg_large_no_sigma`).
- KL packs **group 1 only** (RA/RB slots are reused per group; `neg_large_all_groups`).
- KL_A and KL_B are independent of each other and of the other operand's K1 (disjoint
  outputs); exactly the §4a workspace and the same bits in every region as the unfused
  sequence; `α = 0` / `k = 0` calls run K0 + the β kernel as before.

DRAM traffic of operand x (|x| = 8·b·rows·k bytes), before → after, group 1 and K1:
`2|x|` (K1 pass 1 + pack; K1's pass 2 from L2) → `|x|`; the residue writes are unchanged.
Every later group still reads |x| once (below: g). On chip: chunk 0 in registers and the last
chunk in shared memory, so rows with nc ≤ 2 (k̂ ≤ 8192 read along k̄) are read exactly once;
longer rows re-read nc − 2 chunks from L2 in pass 2 and in pass 3. With nc > 2 one
1024-thread CTA per SM is resident, so the rows in flight are SMs × th × 8k̂ bytes (r_bigk
32 MB, r_mm-like 400 KB rows 51 MB: inside K1's 64 MiB L2 budget, A3). Rows beyond 64 MiB /
SMs (k̂ > 65536 on 128 SMs) partly miss L2 in passes 2/3; pass 2 walking backwards and pass 3
forwards keeps the most recently read chunks first, and the DRAM reads never exceed the
unfused path's. (The threads per CTA and E are launch choices measured in NOTES-D4 §2: 8
values per thread is 9–16 % faster than 16 on on-chip rows, 4 is 8–40 % faster than 16 on
re-read rows; they change no result.)

**Residue groups (g < N).** KL removes one DRAM read of x whatever g: `(1 + ⌈N/g⌉)|x|` →
`⌈N/g⌉|x|` (N = 14: g = 14: 2 → 1; g = 7: 3 → 2; g = 2: 8 → 7). Cutting the later groups'
reads as well would need the residues of all N moduli of x at once, i.e. N slots for x:
exactly the g = N workspace whose RA (or RB) term g < N was chosen to avoid (the large
operand's residues dominate the workspace). A smaller stand-in does not exist: the
quantized a' (|μ| ≤ 2⁵³, §3.3) needs as many bytes as the FP64 input, which is also smaller
than the remaining residues whenever N − g > 8. So the workspace formula (§4a) is
unchanged and later groups re-read x as before.

**Selection (`fused_large`, libimmerdink_host).** Bit 0 (A) / bit 1 (B) is set iff the plan
runs the emulated GEMM without KF (k̂ > 1024). It is decided at plan time from the lowering
only (no SM count, no data), so `imd_lowering` reports the plan's choice exactly.

**[v1.4] Moduli parts.** A pack launch (K2/K3 and KF) with fewer tiles than about two per
SM splits the group's slots over `gridDim.y` block rows: part y of a row writes the slots
`[y·|G|/ny, (y+1)·|G|/ny)` and re-reads its tile. Disjoint outputs, same values
**[spec: Pipeline MParts, pos_pack_parts]**. K5 takes one, two (A5) or four j̄ per thread
(a launch choice by the output count; with four, a warp's results are transposed through
shared memory so C stores are coalesced), and is compiled once per N (A5: the §3.6 constants
and the plane loop are compile-time). Neither changes any result.

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
  **[v1.5]** The split is in units of 65535 × 256 columns for every K4 configuration (a
  multiple of every tile width and of 16), so the launches do not depend on the
  configuration; a launch of a 64- or 128-wide configuration has more than 65535 N tiles
  only when n̂ > 65535·256, which the persistent kernel handles (it has no grid limit).
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
- **K4 configurations [v1.5]** (A3, docs/NOTES-A3.md). Until v1.4 one tile, 128×256×64,
  served every shape; it leaves most SMs idle when a group has few tiles (s128: 8 CTAs on
  128 SMs; r_mm, n̂ = 64: 3/4 of every tile is padding and half the SMs idle). v1.5
  instantiates a small table, `host/k4_config.hpp kK4Configs` (`IMD_K4_*`):

  | id | tile (M×N×K) | warps (warp tile) | stages | smem | CTAs/SM |
  |---|---|---|---|---|---|
  | `IMD_K4_128x256` | 128×256×64 | 8 (64×64) | 3 | 72 KB | 1 |
  | `IMD_K4_128x128` | 128×128×64 | 4 (64×64) | 3 | 48 KB | 2 |
  | `IMD_K4_64x128` | 64×128×64 | 4 (32×64) | 4 | 48 KB | 2 |
  | `IMD_K4_64x64` | 64×64×64 | 2 (32×64) | 4 | 32 KB | 3 |

  - Every entry is the same K4Gemm (persistent CTAs, swizzle-8 tile order, register
    epilogue with the in-quad transpose, which needs 64-column warp tiles) around the same
    CUTLASS main loop and MMA; only the tile → CTA mapping differs. Each W element is the
    same exact §3.5 reduction of the same int32 sum (the K order inside a chunk does not
    matter: integer addition is exact below K_c', cert_kc), so **every configuration
    gives bit-identical W** (tested: every configuration on the contract suites and on
    `cutlass_mod_epilogue`).
  - Same launches (host/k4_launches.hpp, all splits as above), same workspace, same kernel
    sequence: no spec change (Pipeline does not model tiles). The plan's K4 parameter
    sets have one size whatever the configuration, so the plan host bytes (§6a) do not
    depend on it.
  - **Selection (plan time, deterministic).** `k4_select` in libimmerdink_host — the one
    implementation; libimmerdink calls it with the device's SM count at plan creation and
    `imd_lowering` reports it (§6a `k4_config`, for a given `sm_count`) — is a pure function
    of (m, n̂, k̂, b, N, g, SMs): it estimates each configuration's K4 time from its tile
    count and picks the smallest estimate (rule and constants below), fitted on the A3
    measurements (RTX 4090, 128 SMs).
    - Per configuration c (tile tm×tn, o CTAs/SM): tiles = ⌈m/tm⌉·⌈n̂/tn⌉·g·b (one
      group's launch); the busiest SM gets j = ⌈tiles/SMs⌉, run as ⌊j/o⌋ rounds of o
      concurrent CTAs and, if r = j mod o > 0, one round of r. A round of q CTAs costs
      `round_ns + q·tm·tn·k̂·mac_fs[q]`; est_c = the sum, in femtoseconds, computed in
      unsigned 128-bit integers clamped at 2¹²⁷−1 after every operation.
    - Constants (`kK4Costs`, least squares in relative error on nsys K4 kernel times of
      every configuration on 84 (shape, N) cases):

      | c | round_ns | mac_fs[1] | mac_fs[2] | mac_fs[3] |
      |---|---|---|---|---|
      | 128×256 | 2716 | 418 | | |
      | 128×128 | 2550 | 453 | 456 | |
      | 64×128 | 1424 | 505 | 612 | |
      | 64×64 | 1333 | 944 | 1088 | 802 |
    - Choice: 128×256 unless another configuration's estimate is below 95 % of 128×256's
      (`kK4MarginPermille` = 50: model error); then the smallest estimate (ties: lower id).
      On the 84 fitted cases the rule's K4 total is 4.1 % below always-128×256 and 0.4 %
      above the per-case best, and it never picks a configuration slower than 128×256
      (NOTES-A3 §3).
  - **Autotuning (optional plan option `k4_tune`, default off; §6).** The plan times every
    configuration once on its device and keeps the fastest. Not the default: plan creation
    then runs kernels and allocates temporarily, and timing on a shared GPU is noisy.
  - Measured and not in the table: 256×128×64 (within noise of 128×256, M4/A2), 4 stages
    of 128×256 (1–3 % slower, A2), 32-row warp tiles at 128×128 (register-limited).
- cuBLASLt is not used (no runtime or test dependency).

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
- **[v1.6]** The fused paths (KF, KL) change no region: they write the same regions with the
  same bits (group 1's slots of RA/RB, σ/τ, flags) as the kernels they replace (§4 "Residue
  groups": why KL does not hold more than g slots).
- The caller must pass `ws_bytes ≥ workspace_size`.
- The result includes the tail padding after W, up to 255 bytes.
- If any product or the total overflows uint64, the call returns INVALID_VALUE.
- Worked example: 8192³, N = 14, g = 14 needs 2,819,014,912 B (2.625 GiB).

---

## 4b. K4 load alignment [v1.13] (stage 1 of a staged plan; NOTES-K4ALIGN)

**Mechanism (measured; CUTLASS `predicated_tile_access_iterator.h:204-213, 514-523`).**
- K4 (CUTLASS 2.x, int8 A row-major and B column-major, alignment 16, threadblock K tile 64)
  reads RA/RB rows with row stride `lda = ldb = k̂` (the region and pack bases are
  256-aligned).
- A launch covers the K-chunk `[k0, k0 + kc)`. It loads the chunk's residue tile
  (`kc mod 64` elements) **first**. Every later main-loop load of row r therefore starts at
  phase

  `φ(r, c) = (r·k̂ + e_c) mod 64`, where `e_c = k0 + kc` is the chunk end.

  So the phase depends on the stride and the chunk end together, not on the stride alone,
  and not on the chunk start.
- Measured K4 time per MAC, relative to class 0, for `k̂ mod 64` classes 32 / 16 / 48:

  | config | class 32 | class 16 | class 48 |
  |---|---|---|---|
  | 128×256 | 1.03 | 1.32–1.34 | 1.37–1.44 |
  | 128×128 | 1.10–1.11 | 1.55–1.58 | 1.61–1.73 |
  | 64×128 | 1.16–1.17 | 1.69–1.73 | 1.81–1.92 |
  | 64×64 | 1.15–1.19 | 1.69–1.76 | 1.83–1.94 |

  These cover 2048² and 16384×512, N = 8 and 14. ncu shows more L2 requests and sectors and
  more shared-memory store wavefronts; DRAM reads are flat.
- Two facts are measured but **not** explained by the phase rule:
  - class 48 is 5–15 % slower than class 16, although both have the same set of phases;
  - the sector multipliers are 1.5× with half of the rows misaligned and 2.0× with all of
    them (a 64-byte segment touching 3 sectors instead of 2 predicts at most 1.5×).

**Stage 1 (this version): `K_c' = ⌊K_c/64⌋·64 = 131008`** **[cert: cert_kc, cert_row_phase]**.
- Every full chunk has `k0 ≡ kc ≡ 0 (mod 64)`, so its phase is `r·k̂ mod 64`. With 131056
  it was `r·k̂ + 48·(c+1)`. The last chunk ends at k̂ and does not depend on K_c'.
- The phase class therefore depends on k̂ alone, and no chunk gets worse:
  - k̂ ≡ 0 (mod 64) is fully aligned in every chunk (before: full chunk c sat at phase
    48·(c+1) mod 64, so chunk 0 was fully misaligned);
  - k̂ ≡ 32: the even chunks move from phases {16, 48} to {0, 32} (the odd ones were
    already there);
  - odd multiples of 16 are unchanged (their phases are uniform either way).
- 131008 is the largest multiple of 64 that is ≤ K_c. 131040 (≡ 32 mod 64) would leave
  full chunks in class 32.
- Exactness is unchanged. CRT and add-mod do not depend on the chunking (any chunk length
  ≤ K_c is exact, §2/§3.5), so W and C are bit-identical to v1.12. The epilogue range
  shrinks to `[−2129666048, 2146435327]` for p = 256.
- What changes:
  - `k_chunks = ⌈k̂/131008⌉` gains one for `k̂ ∈ (j·131008, j·131056]`;
  - in those windows a call has one more add-mod launch per group and part (§4 launch
    counts), one more W read-modify-write (`N·b·m·n̂` bytes), and possibly one more add-mod
    shape key (plan host bytes);
  - workspace and pack layout, the pack format, the host API version and the calibration
    format do not change.
- **Measured** (same-session interleaved A/B of v1.12 and v1.13; NOTES-K4ALIGN §6):
  - 2048² × 196608, N = 8, 128×256: 23.5 → 32.0 TFLOPS-eq per call (70.1 → 51.5 ms); K4
    45.1 → 25.9 ms.
  - 2048² × 196640 (k̂ ≡ 32 mod 64): 25.8 → 31.8 TFLOPS-eq.
  - 512² × 262144: 11.1 → 11.8 TFLOPS-eq; K4 −18 %.
  - Odd multiples of 16 (196624, 262160): unchanged within 2 %.
  - Window cases (k̂ = 131040 and 131056, now two chunks): the call is unchanged within
    1 %; K4 changes by −2 to +3 %.

**Later stages (not in v1.13 stage 1; DESIGN text before code, review 2026-10-05):**
- Stage 0, done with stage 1, is a refactor with no behaviour change. `k̂ = ⌈k⌉_A` has one
  definition, `k_hat_of(k, A)`, in libimmerdink_host (`src/host/k_hat.hpp`), mirrored by
  `spec/params.py k_hat_of`. A = 16 for every plan.
- Stage 2: `k̂ = ⌈k⌉_A` with A ∈ {16, 64}, a pure function of k
  (`A = 64 iff k > K_PAD64_MIN`, the threshold frozen from a sweep).
  - k̂ is both the row stride and the GEMM K extent, with a zero tail.
  - It adds no `imd_options_t` field. A test hook forces A, and `imd_lowering_t` reports
    the resolved alignment (additive, host API 9).
  - The pack format word carries A (1 = A16, 2 = A64).
  - Padding only the stride while the K extent stays ⌈k⌉₁₆ is **rejected**: with the
    residue tile loaded first, it misaligns every row of every full chunk.
- The W/n̂ stride (`ldw = n̂`, ≡ 16 mod 32 possible) is excluded here and unmeasured.

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

## 6. API (C ABI, ctypes-friendly; Python: §6d)

```c
typedef struct { int32_t rank; const int64_t* extent; const int64_t* stride; /* elements */
                 const int32_t* label; } imd_tensor_t;
typedef struct { int32_t num_moduli;      /* N, 2..20                        */
                 int32_t residue_group;   /* g, 0 = N                         */
                 int32_t rounding;        /* IMD_RN (default) | IMD_TRUNC     */
                 int32_t mode;            /* IMD_EMULATE | IMD_DIRECT         */
                 int32_t vec_mode;        /* [v1.2] IMD_VEC_GENERATE (0, default)
                                             | IMD_VEC_RESIDENT                */
                 int32_t k4_tune;         /* [v1.5] IMD_K4_TUNE_OFF (0, default)
                                             | IMD_K4_TUNE_ON                  */
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
- **[v1.5] `k4_tune` (plan-time autotuning of the K4 configuration, §4; optional, not
  decided as a default).** The struct grows to 24 bytes, as in v1.2: zero is the default
  (off), callers must be rebuilt, an unknown value is INVALID_VALUE in every function that
  takes options. Off: the plan runs `k4_select`'s configuration for the device's SM count
  (the `imd_lowering` `k4_config` for that `sm_count`). On, for a plan with the emulated
  GEMM and `k4_tune_bytes > 0` (§6a; 0 above 256 MiB, where the rule is kept):
  `imd_plan_create` allocates `k4_tune_bytes` of device memory (group 1's RA, RB and W
  planes, one allocation), zeroes it and times group 1's K4 launches in every configuration
  on `cudaStreamPerThread`: one warm-up run each, which sizes a batch of back-to-back runs
  of about 0.1 ms (1–32 runs), then three rounds of one batch per configuration,
  interleaved; the minimum per run counts. It frees the memory and keeps the fastest
  configuration (the rule's unless another is strictly faster). It reads no user data and
  writes nothing the plan keeps but the choice. (A single timed run per configuration, the
  first version, picked 10–30 % slower configurations for small problems: NOTES-A3 §5.)
  - Results are bit-identical whatever the choice (§4), so tuning changes speed only;
    the choice itself may differ between plans (timing noise).
  - Costs: plan creation synchronizes `cudaStreamPerThread`, and `cudaMalloc`/`cudaFree`
    may synchronize the device; plan creation time grows by the allocation, the zeroing
    and about 4 × (0.1 ms or one run) of group 1's K4 launches per configuration
    (NOTES-A3 §5 measures it).
    Not capturable (as plan creation never was).
  - Statuses: the temporary allocation failing is ALLOC_FAILED, a launch or timing error
    CUDA_ERROR; the plan is not created in either case.
  - Thread safety is unchanged: the temporary memory is private to the call; concurrent
    tuned plan creations only perturb each other's timings.

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
  **[v1.2]**; the hooks are not thread-safe. **[v1.5]** A hook (and a benchmark-only
  profiling hook) forces the K4 configuration of later plans, for the invariance tests and
  the A3 measurements.
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
  immerdink uses (MEMORY_MODEL §3), except **[v1.5]** the `k4_tune_bytes` (§6a) a plan
  creation with `k4_tune` on allocates and frees before it returns.
- **ALLOC_FAILED** covers host allocation failure in plan creation and, with
  `IMD_VEC_RESIDENT`, the plan's device allocation (and **[v1.5]** with `k4_tune`, its
  temporary allocation); any other CUDA failure, including plan creation's device-property
  and kernel-attribute calls, is CUDA_ERROR. **[v1.2: no pinned allocation any more]**
- **Threads [v1.2] [spec: HostAPI]:**
  - `imd_plan_create` does all host-side GEMM setup: it reads the SM count and the K4
    kernels' occupancy, sets their dynamic shared-memory attribute (CUDA runtime calls,
    thread-safe), and stores the parameters of every K4 launch shape in the plan
    (**[v1.5]** of the plan's K4 configuration, which is fixed at creation).
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
- **[v1.7] Packed operands** `imd_pack` / `imd_plan_create_packed` / `imd_contract_packed`
  (fast-mode σ depends only on the operand itself, so a pack is reusable across
  contractions): §6b, implemented.
- **Reserved, not v1** (the API and specs must not preclude these):
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

typedef struct {               /* 136 bytes in version 1, 152 in v2 [v1.4], 168 in v3 [v1.5],
                                  176 in v4 [v1.6], 184 in v5 [v1.7] */
  uint32_t struct_size;        /* in: sizeof(imd_lowering_t) as the caller compiled it  */
  int32_t  status, lowered, dispatch, swapped, num_moduli, residue_group, num_groups;
  int64_t  m, n, k, b, n_hat, k_hat, vec_entries, k_chunks, gemm_launches;
  uint64_t kappa, workspace_bytes, plan_device_bytes, plan_host_bytes;
  int32_t  fused, reserved;    /* version 2 [v1.4] */
  int64_t  kernel_launches;    /* version 2 [v1.4] */
  int32_t  sm_count;           /* version 3 [v1.5]: in (0 = 128) and out */
  int32_t  k4_config;          /* version 3: IMD_K4_* of the rule (§4) */
  uint64_t k4_tune_bytes;      /* version 3: temporary bytes of a k4_tune plan creation */
  int32_t  fused_large;        /* version 4 [v1.6]: bit 0 / bit 1: KL for the GEMM's A / B */
  int32_t  reserved4;          /* version 4: 0 */
  int32_t  packed;             /* version 5 [v1.7]: bit 0 / 1: the caller's A / B packed (§6b) */
  int32_t  pack_mismatch;      /* version 5: IMD_PACK_MISMATCH_* of a PACK_MISMATCH */
} imd_lowering_t;
enum { IMD_DISPATCH_NOOP, IMD_DISPATCH_EMULATE, IMD_DISPATCH_DIRECT, IMD_DISPATCH_SCALE };
enum { IMD_K4_128x256, IMD_K4_128x128, IMD_K4_64x128, IMD_K4_64x64, IMD_K4_CONFIG_COUNT };
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
  - **[v1.4] Version 2** appends `fused` (1 iff EMULATE with k̂ ≤ 1024 (512 before v1.5): the §4 KF launch) and
    `kernel_launches`, the kernel launches of one `imd_contract` with α ≠ 0: NOOP 0; DIRECT
    and SCALE `v + 1`; EMULATE `(fused ? 1 : v + 2) + (⌈N/g⌉ − fused)·(2 + [n̂ > n]) +
    gemm_launches + 1`, with `v` = 1 for `IMD_VEC_GENERATE` (K0), 0 for resident vectors (a
    call with α = 0 launches `v + 1`). Both are valid whenever `lowered = 1`, like
    `k_chunks`. A version-1 caller (struct_size 136) gets exactly the version-1 fields.
    Tested: the LD_PRELOAD shim counts `cudaLaunchKernel` during every test call and
    compares it with the plan, and the plan with the host query.
  - **[v1.5] Version 3** appends `sm_count`, `k4_config` and `k4_tune_bytes`, valid
    whenever `lowered = 1`. `sm_count` is also an **input** (read only when
    `struct_size ≥ 168`): the SM count of the device the caller will run on; 0 means 128
    (the RTX 4090 the rule was fitted on); negative → INVALID_VALUE with nothing written.
    The library writes back the count it used. `k4_config` is `k4_select` (§4) for that
    count on EMULATE (`IMD_K4_128x256` otherwise): exactly the configuration
    `imd_plan_create` uses on a device with that many SMs and `k4_tune` off.
    `k4_tune_bytes` = Σ align₂₅₆ of g·b·m·k̂ (RA), g·b·n̂·k̂ (RB) and g·b·m·n̂ (group 1's W
    planes) on EMULATE when that sum is ≤ 256 MiB, else 0: the device memory a `k4_tune`
    plan creation allocates temporarily (0: no tuning would run), reported whatever this
    call's `k4_tune` (the option changes no host query). A version-2 caller gets exactly
    the version-2 fields. Tested: `host_c_api`, `host_python` (against an independent
    Python transcription of the rule for several SM counts), `host_vs_plan` (every real
    plan runs the rule's configuration for its device's SM count; tuned plans have the
    same sizes).
  - **[v1.6] Version 4** appends `fused_large` and `reserved4` (0), valid whenever
    `lowered = 1`: bit 0 (bit 1) is set iff the plan scales and packs the GEMM's A (B,
    post-swap) with KL (§4 "Fused large path"; 0 when `fused`, and 0 off EMULATE), and
    `kernel_launches` on EMULATE becomes `(fused ? 1 : v + 2) + (⌈N/g⌉ − 1)·(2 + [n̂ > n]) +
    (fused ? 0 : [A not KL] + [B not KL]·(1 + [n̂ > n])) + gemm_launches + 1` (with
    `fused_large = 0` this is the version-2 count: KL replaces K1 of its operand one for one
    and removes that operand's group-1 pack launches). A version-3 caller gets exactly the
    version-3 fields (its `kernel_launches` is the v4 count, which is what the library
    launches). Tested: `host_c_api`, `host_python`, `host_vs_plan` (every real plan's choice
    and launch count equal the query), and the launch counts of every contract test call.
  - **[v1.7] Version 5** appends `packed` and `pack_mismatch` (§6b.5): 0 from `imd_lowering`;
    `imd_lowering_packed` fills them (and every other field for the packed plan). A
    version-4 caller gets exactly the version-4 fields.
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
  launch-limit hook changes only libimmerdink's plans. **[v1.5]** It does not depend on
  the K4 configuration, and neither do the plan host bytes (one parameter-set size).

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
(the device API, §6d, is ctypes too), with an einsum-style front end (`"kjbc,kiac->abij"` + extents + optional
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

## 6b. Packed operands — **APPROVED design (2026-10-03), implemented in v1.7** (ROADMAP D1)

Implemented as written (host API version 5); §6b.11 lists the choices the design left to
the implementation and one clarification of §6b.5. Measurements: docs/NOTES-D1.md,
BENCHMARKS "D1". **Binding user decision:** an **explicit** packed-operand
handle, not implicit plan-level caching; kernel level only (caller-owned device memory,
the caller's stream, no allocation or synchronization in a call, capture-safe).
**[spec: PackedOperand, 11 pos / 17 neg]**

### 6b.1 What a packed operand is

A packed operand is one FP64 operand X in exactly the form the emulated GEMM consumes:
per row the shift σ (§3.2) and the non-finite flag (§3.1), and **all N** residue planes
(§3.3–§3.4) over the whole canonical K, in a byte layout (§6b.4) that is a function of X
alone. `imd_pack` writes it once, on the caller's stream, into caller-owned device memory;
any number of contractions read it in place of scaling and packing X themselves.

Per contraction that uses it, a packed operand saves X's K1 pass, X's ⌈N/g⌉ pack passes (or
X's share of KF) and the workspace regions sX, fX, RX. It costs the caller
`N·b·r·k̂` bytes (≈ N/8 × the FP64 operand, 1.75× at N = 14) for as long as it is kept.
(Measured share of what it removes: the pack is 10–15 % of large calls; at small shapes K1
+ pack of one operand is ≈ 15 µs of ≈ 50 µs at N = 8, NOTES-A1.)

Notation: X has three label classes as seen by the pack — **K** (contracted: X's labels not
in C), **L** (batch: in C and in the partner) and **R** (rows: the rest); `b`, `r`, `k` are
their extent products after §5 steps 2 and 4 (diagonals merged, extent-1 modes dropped),
`k̂ = ⌈k⌉₁₆`.

### 6b.2 What the packed bytes depend on (compatibility analysis)

From §3, §4, §5 and `spec/oracle.py` (`scale_vector`, `quantize`, `residue`, `lower`):

| Packed item | Depends on | Source |
|---|---|---|
| flag fX[λ,ρ] | X's values over the row's K cells | §3.1 |
| σ[λ,ρ] | X's values over the row's K cells, **with multiplicity** (S counts every canonical k̄, so a stride-0 K mode duplicates entries); **k** (RN only: `R = (2T − κ)²`, κ = ⌈√k⌉); **N** (T); **rounding** (rule and R) | §3.2 step 4 |
| a′ | value, σ, rounding (RN-even / trunc) | §3.3 |
| residue of plane l | a′, p_l: the moduli prefix of length N | §3.4 |
| byte position | the R, L and K **enumerations** (orders and extents), k̂ (= ⌈k⌉₁₆), the region/row/plane strides | §4a, §6b.4 |

σ does **not** depend on the partner: the fast-mode scaling is per side (Cauchy–Schwarz),
which is why §7 rejects accurate mode for v1. The bytes do **not** depend on: the
partner's values or strides, C, α, β, the residue-group size g (all N planes are stored; a
group is a contiguous slice, §6b.4), moduli parts (a launch split), the K-chunking (K4
reads chunks of the same rows), the K4 configuration and `k4_tune` (bit-identical W, §4),
`vec_mode`, KF vs unfused (same bits, §4), fusion (§5 step 9 merges modes without changing
the enumeration), the stream.

**Where v1.5 couples X's layout to the partner and C** (§5 step 8):
1. **K order** = (|s_big|, |s_other|, position), big = A iff m ≥ n: it depends on the
   partner's extents (which side is big), the partner's strides (second key) and the
   label positions in `A ++ B ++ C`.
2. **R order** (I of A or J of B) = (|s_C|, |s_X|, position): it depends on C.
3. **L order** = (|s_C|, |s_A|, |s_B|, position): it depends on C and the partner.
4. **The role swap** is decided by C: X may be post-swap A (m rows, RA layout) or B (n̂
   rows, RB layout: the workspace pads only B to n̂).
5. **k** includes **partner-only** labels (§5 step 6: summed, stride 0 in X): they change
   S (duplicated entries), κ, k̂ and the layout.

**Resolution.**
- **The pack fixes its own orders; a plan built against it adopts them.** For each class
  the packed operand carries (K, L, and X's own rows), the plan's canonical order is the
  pack's; the partner's gather and C's scatter vectors follow it. Any order within a class
  is a valid GEMM view (the enumeration of each class is a bijection whatever the order;
  fusion stays sound, it only tests stride products); the follow-up IndexMap model with an
  adopted order is listed in §6b.9.
- **Numerical consequence: none for finite data.** The emulated result is independent of
  the K order: S is an exact sum, residues are per element, W is an exact integer sum mod
  p (any order below K_c', cert_kc), CRT is exact **[spec: PackedOperand
  EmulatedOrderFree]**. Only the native path of **flagged** outputs (§3.7: an fma chain in
  canonical K order) is order-sensitive, when a finite partial sum overflows (oracle A7:
  NaN vs −Inf). Precisely: **a packed contraction is bit-identical to the unpacked
  contraction run with the pack's orders**; it is bit-identical to the plain unpacked
  contraction for every unflagged output, and for flagged outputs whenever the two K
  orders agree **[spec: PackedOperand CEqual; neg_witness_native_order]**. The default
  (AUTO, §6b.5) K order is X's stride order, which is also v1.5's K order whenever X is the
  "big" operand with distinct K strides; `IMD_PACK_ORDER_GIVEN` with the helper
  `imd_pack_desc_for` reproduces any one contraction's v1.5 orders exactly.
- **Performance consequence.** The partner's K gather (K1, K2/K3, KF) and C's scatter
  (K5) follow X's orders instead of their own: the partner's pack tiles may read across
  strides, and in the post-swap B role j̄'s fastest mode is X's fastest row mode, not
  necessarily C's (the swap's coalescing goal is lost; K5 writes C strided). The L order
  affects the partner and C the same way. Mitigation: GIVEN orders for the contraction
  that matters (or two packs of one tensor). To be measured in implementation.
- **The swap is unchanged** (decided by C, §5 step 8); the layout is **role-agnostic**
  (§6b.4), so one pack serves either post-swap role.
- **Partner-only K labels → PACK_MISMATCH.** The plan's K would contain a label X does
  not carry; X's rows would need duplicating along it. Not supported (the caller can sum
  the partner over that label first). X-only K labels are fine: they are in the pack's K
  set and the partner carries them with stride 0.

**The compatibility key** (the "packed layout descriptor"): a canonical sequence of int64
words, computed by `imd_pack_create` from the pack descriptor and by plan creation for each
packed slot from its own lowering, and compared **word for word**:
1. `IMD_PACK_FORMAT` (1): fixes k̂ = ⌈k⌉₁₆, the regions and their alignment, the strides,
   the tail, the zero-padding rule and the moduli table. Bumped by any change to the bytes
   (e.g. if D4 or E1 needs another K alignment).
2. N and rounding.
3. b, r, k.
4. The **FP64 element map**: for each class in the order L, R, K, the fused modes
   (extent, stride in X), in the pack's canonical order, fused over **X's strides alone**
   (maximal runs with `s′ = s·e`). It is label-free: the same tensor under renamed labels
   matches. The plan computes it from its classes restricted to X, in its (adopted) orders,
   with X's strides — so a partner-only K label appears as a mode (e, 0) and mismatches,
   and a label that is batch in the plan but a row in the pack mismatches.

The key is **sound** (equal keys ⇒ equal σ, flags and bytes for equal X values) but not
complete (two descriptors that enumerate the same elements but fuse differently are
rejected; safe). Strides are in the key so that the FP64 pointer the contract receives
addresses the elements the pack read (K5's native path). Not in the key: everything in
the "does not depend" list above, the role, the mode (a DIRECT plan accepts a matching
pack and does not read it).

**Each key part is necessary [spec: PackedOperand]:** binding a pack that differs in one
part gives a wrong result — a K order not adopted (`neg_order`), N (`neg_n`), rounding
(`neg_rnd`), the K set (`neg_kset`), the format's k̂ (`neg_kalign`), two packs with
different K orders (`neg_partner_order`); with the key check, every such pack is rejected
and every matching one accepted (`pos_key_check`, `pos_key_check_both`,
`AcceptedWhenKeyMatches`).

### 6b.3 κ ≥ T

k is fixed by the pack (the K set is in the key), N too. So `imd_pack_create` applies the
§3.2 RN acceptance itself: RN, k ≥ 2 and κ ≥ T → NOT_SUPPORTED (such a pack could never
feed an emulated GEMM). Plan creation keeps its rule and, with equal k and N, gives the
same answer; nothing is checked at contract time. A pack whose K is empty after lowering
(k = 1) is NOT_SUPPORTED too: every contraction with it dispatches DIRECT (§5) and would
never read it. k = 0 or b·r = 0: a valid pack of 0 bytes (`imd_pack` does nothing).

### 6b.4 Byte layout (format 1) — the contract

```
region  bytes                      contents
sig     2·b·r                      int16 σ of row (λ, ρ) at index λ·r + ρ
flag    1·b·r                      u8 non-finite flag, same index
res     N·b·r·k̂ + (⌈r⌉₁₆ − r)·k̂    int8 residue of plane l ∈ [0, N), batch λ, row ρ, k̄:
                                   offset l·b·r·k̂ + (λ·r + ρ)·k̂ + k̄
                                   k̄ ∈ [k, k̂): 0;  the last (⌈r⌉₁₆ − r)·k̂ bytes ("tail"): 0
pack_bytes = Σ align₂₅₆(region) in this order;  0 if b·r·k = 0
```
- Every byte of `[0, pack_bytes)` is written by `imd_pack`, alignment gaps with 0: the
  bytes are a pure function of (key, X's values), so packs can be compared or hashed.
- **Role A** (post-swap): K4 reads m = r rows per batch at batch stride r·k̂ — exactly the
  v1 RA layout `[slot][λ][ī][k̂]`.
- **Role B**: the GEMM's N extent is n̂ = ⌈r⌉₁₆; block (l, λ) is read from its start at
  batch stride r·k̂ (not v1's n̂·k̂), so its rows r..n̂−1 are the first rows of the next
  block, or the tail after the last block. Those rows only feed W columns ≥ n, which K5
  never reads (§4 Padding; Pipeline "B padding rows: nothing"), are finite int8 (no
  saturation: K_c' counts every position) and initialized, and the tail keeps the last
  block in bounds **[spec: NoOOBRead; neg_notail, neg_brole]**. Padding rows to ⌈r⌉₁₆
  instead would cost up to 16× for batch-heavy operands with few rows (r = 1); the tail
  costs at most 15·k̂ bytes once.
- **Residue groups and parts**: group G = planes [g₀, g₀ + |G|) is the contiguous slice at
  `res + g₀·b·r·k̂` with batch index z = s·b + λ at stride r·k̂, v1's z addressing; so the
  layout is independent of g and of moduli parts **[spec: CEqual over g ∈ Gs]**.
- **K padding is zero** in every pack. Numerically one zero side suffices, but two packed
  operands meet with no in-line partner to supply it **[spec: neg_nopad]**.
- **Alignment**: k̂ is a multiple of 16, so every row is 16-byte aligned (CUTLASS); the
  buffer must be 256-byte aligned (MISALIGNED).
- The packed buffer replaces the workspace regions sX, fX and RX of the operand's
  post-swap role: those have 0 bytes in the packed plan's §4a layout (§6b.6).

### 6b.5 API

**Host library** (libimmerdink_host, CUDA-free, pure, thread-safe; `IMD_HOST_API_VERSION`
5):
```c
#define IMD_PACK_FORMAT 1
enum { IMD_PACK_ORDER_AUTO = 0, IMD_PACK_ORDER_GIVEN = 1 };

typedef struct {
  uint32_t            struct_size;     /* sizeof(imd_pack_desc_t) as the caller compiled it */
  int32_t             num_moduli;      /* N, 2..20 (no 0 = default: N is part of the key)  */
  int32_t             rounding;        /* IMD_RN | IMD_TRUNC                               */
  int32_t             order;           /* IMD_PACK_ORDER_AUTO | IMD_PACK_ORDER_GIVEN       */
  const imd_tensor_t* tensor;          /* X: extents, strides (elements), labels           */
  int32_t             num_contracted;  /* K: X's labels summed over (not in C)             */
  const int32_t*      contracted;      /*   GIVEN: in canonical order, first fastest       */
  int32_t             num_batch;       /* L: X's labels in C and in the partner            */
  const int32_t*      batch;           /*   GIVEN: in canonical order                      */
  int32_t             num_rows;        /* GIVEN: X's remaining labels in order; AUTO: 0    */
  const int32_t*      rows;            /*   (AUTO: NULL)                                   */
} imd_pack_desc_t;

typedef struct {
  uint32_t     struct_size;            /* in */
  int32_t      status, format, num_moduli, rounding, reserved;
  int64_t      b, r, k, k_hat;
  uint64_t     kappa;                  /* ⌈√k⌉ */
  imd_region_t sig, flag, res;         /* offsets and bytes in the packed buffer */
  uint64_t     pack_bytes;             /* §6b.4 total */
  uint64_t     pack_ws_bytes;          /* workspace imd_pack needs (may be 0) */
  uint64_t     key_digest;             /* FNV-1a of the key words: diagnostics only */
} imd_pack_layout_t;

typedef struct { const imd_pack_desc_t* A; const imd_pack_desc_t* B; } imd_packing_t;
                                       /* NULL member: that operand is packed in-line */

imd_status_t imd_pack_layout(const imd_pack_desc_t*, imd_pack_layout_t* out);
imd_status_t imd_pack_desc_for(const imd_tensor_t* A, const imd_tensor_t* B,
                               const imd_tensor_t* C, const imd_options_t*,
                               int32_t operand,          /* 0 = caller's A, 1 = caller's B */
                               imd_pack_desc_t* out, int32_t* labels, int32_t capacity);
imd_status_t imd_workspace_size_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t*,
                                       const imd_packing_t*, size_t* bytes);
imd_status_t imd_workspace_layout_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                         const imd_tensor_t* C, const imd_options_t*,
                                         const imd_packing_t*, imd_region_t* regions,
                                         int32_t capacity, int32_t* num_regions,
                                         size_t* total);
imd_status_t imd_lowering_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                 const imd_tensor_t* C, const imd_options_t*,
                                 const imd_packing_t*, imd_lowering_t* out);
```
- `imd_pack_layout`: statuses SUCCESS, INVALID_VALUE (N or rounding out of range; a listed
  label not in X, listed twice, or in two lists; GIVEN with a missing row label; AUTO with
  `rows`; §5 rows 1 and 5 applied to X alone: inconsistent repeated-label extents, span or
  extent-product overflow), NOT_SUPPORTED (§6b.3: κ ≥ T with RN; K empty). AUTO orders each
  class by (|s_X| after the diagonal merge, first position in X's labels).
- `imd_pack_desc_for`: fills `out` (GIVEN) with the classes and orders that v1.5's
  unpacked plan of (A, B, C, options) uses for one operand, its label lists written into
  the caller's `labels[capacity]` (`out`'s pointers point into it; `out->tensor` = the
  argument). A pack made from it reproduces that contraction bit for bit, including flagged
  outputs, with v1.5's C-write coalescing.
- `imd_workspace_size_packed` / `imd_workspace_layout_packed`: §4a with sX, fX, RX of each
  packed operand's **post-swap** role at 0 bytes (same region order, `bytes = 0` = absent).
  Same statuses as the in-line functions plus PACK_MISMATCH (they compute the slot keys:
  they are the plan-time compatibility check, without a GPU).
- `imd_lowering_packed`: as `imd_lowering`, for the packed plan; **version 5** of
  `imd_lowering_t` (184 bytes, after v1.6's version 4 at 176) appends `int32_t packed` (bit 0: caller's A packed, bit 1:
  caller's B) and `int32_t pack_mismatch` (0, or the first differing key part:
  `IMD_PACK_MISMATCH_FORMAT, _MODULI, _ROUNDING, _CLASSES` (incl. partner-only K labels),
  `_EXTENTS, _STRIDES, _ORDER` (two packs' K or L orders differ)). `kernel_launches`,
  `fused` and the plan bytes describe the packed plan (no K1 or pack launches for a packed
  operand; KF, if chosen, works on the in-line operand only).
- New status **`IMD_STATUS_PACK_MISMATCH` (8)**, appended; `imd_status_string` names it.

**Device library** (libimmerdink):
```c
typedef struct imd_pack_s* imd_pack_t;   /* immutable host object: the key and the layout */

imd_status_t imd_pack_create(imd_handle_t, const imd_pack_desc_t*, imd_pack_t*);
imd_status_t imd_pack_destroy(imd_pack_t);
imd_status_t imd_pack_bytes(imd_pack_t, size_t* pack_bytes, size_t* pack_ws_bytes);
imd_status_t imd_pack(imd_pack_t, const double* X, void* packed, size_t packed_bytes,
                      void* ws, size_t ws_bytes, cudaStream_t);

imd_status_t imd_plan_create_packed(imd_handle_t, const imd_tensor_t* A,
                                    const imd_tensor_t* B, const imd_tensor_t* C,
                                    const imd_options_t*, imd_pack_t packA, imd_pack_t packB,
                                    imd_plan_t*);        /* NULL pack: that operand in-line */

typedef struct {
  const double* data;          /* the FP64 operand (always: K5's native path, DIRECT)     */
  const void*   packed;        /* imd_pack's output, or NULL when the plan packs in-line   */
  size_t        packed_bytes;
  imd_pack_t    pack;          /* a handle whose key equals the plan's slot key, or NULL   */
} imd_operand_t;

imd_status_t imd_contract_packed(imd_plan_t, double alpha, const imd_operand_t* A,
                                 const imd_operand_t* B, double beta, double* C,
                                 void* ws, size_t ws_bytes, cudaStream_t);
```
- `imd_pack_create`: `imd_pack_layout`'s statuses, ALLOC_FAILED (the host object). The
  handle copies the descriptor (O(rank)); no device work.
- `imd_pack` checks, in order: null pointers (X and packed when `pack_bytes > 0`, ws when
  `pack_ws_bytes > 0`) → INVALID_VALUE; `pack_bytes = 0` → SUCCESS, nothing touched;
  ALIASING (packed ∩ X; ws ∩ X or packed); MISALIGNED (X 8, packed and ws 256 bytes);
  `packed_bytes < pack_bytes` or `ws_bytes < pack_ws_bytes` → INSUFFICIENT_WORKSPACE. Then it
  launches only (no allocation, no synchronization). Its kernel(s) are an implementation
  choice (the D4 work may change them); only the bytes of §6b.4 are specified.
- `imd_plan_create_packed`: the §6 plan-time checks, then the key of each packed slot
  against the plan's lowering with adopted orders (§6b.2) → PACK_MISMATCH, then §3.2 RN
  acceptance (last, as in §5). `options.num_moduli` and `rounding` must equal the packs'
  (else PACK_MISMATCH: `_MODULI` / `_ROUNDING`); `residue_group`, `vec_mode`, `k4_tune`,
  `mode` are free. The plan **copies the keys**; it keeps no reference to the handles.
  With both handles NULL it is `imd_plan_create`.
- `imd_contract_packed` checks, in order:
  1. A or B (the `imd_operand_t*`) NULL → INVALID_VALUE; for each operand: `packed` non-NULL
     iff the plan packs it, and then `pack` non-NULL with a key equal to the plan's slot
     key, else PACK_MISMATCH (host-only compare of O(rank) words);
  2. §6's null rule with `data` for A and B, plus `packed` when α ≠ 0, k ≠ 0, the dispatch
     is EMULATE and `pack_bytes > 0` → INVALID_VALUE;
  3. an empty output → SUCCESS, nothing touched;
  4. ALIASING: C ∩ any of A.data, B.data, A.packed, B.packed; ws ∩ any of those or C
     (byte intervals as §5 row 6; a packed buffer's interval is `[packed, packed +
     pack_bytes)`). Packed buffers may overlap each other and either FP64 operand (all are
     only read);
  5. MISALIGNED (packed buffers 256 bytes);
  6. `ws_bytes` < the packed workspace size, or `packed_bytes < pack_bytes` →
     INSUFFICIENT_WORKSPACE.
  `imd_contract` on a plan made with a pack returns PACK_MISMATCH (it has no packed
  pointers to read).
- **Dispatch.** EMULATE reads the packed σ, flags and residues and the FP64 data only for
  flagged outputs; DIRECT, K-empty, SCALE (k = 0, α = 0) and NOOP never read the packed
  buffer (DIRECT reads `data`).
- **`vec_mode`**: unchanged. The vec region (or the plan's resident copy) still holds X's
  vectors: K5's native path reads X through them.
- **KF / fused paths**: the plan-time choice is unchanged in form; KF (and D4's fused
  variants) scale and pack only the in-line operand. With both operands packed no scale or
  pack kernel runs: K0 (generate mode), K4, K5.
- **K4**: `k4_select` has the same inputs (m, n̂, k̂, b, N, g, SMs); a packed operand
  changes only K4's operand pointer and, in the B role, its batch stride (r·k̂). `k4_tune`
  times the plan's own parameters on its private temporary memory, as now.

**Versioning.** `IMD_HOST_API_VERSION` 5 (v1.6's KL took 4) adds the functions, `IMD_STATUS_PACK_MISMATCH`, the
`IMD_PACK_*` enums and `imd_lowering_t` v5; `imd_pack_desc_t` and `imd_pack_layout_t` carry
`struct_size` (trailing fields may be added; older callers get what they know). Nothing
existing changes. The **packed bytes are versioned by `IMD_PACK_FORMAT`**, which is part of
the key: a pack is valid for any library version with the same format.

### 6b.6 Workspace and plan bytes with packed operands

- §4a unchanged except: for a packed post-swap A, `sA = fA = RA = 0` bytes; for a packed B,
  `sB = fB = RB = 0`. vec and W unchanged. (Both packed at 8192³, N = 14: 2.625 GiB → W +
  vec only, ≈ 0.88 GiB.)
- Plan device bytes unchanged (§4a); plan host bytes grow by the copied keys (O(rank)),
  reported by `imd_lowering_packed`.
- The pack buffer: `pack_bytes` (§6b.4) per packed operand, owned by the caller; `imd_pack`'s
  workspace `pack_ws_bytes`, needed only during `imd_pack`. With the user tensors and the
  workspace, that is all the device memory (MEMORY_MODEL §3 to be updated).

### 6b.7 Threads, lifetime and ordering

- **`imd_pack_t`** is immutable after creation: any number of threads may pass one handle
  to `imd_pack`, `imd_plan_create_packed` and `imd_contract_packed` concurrently. It may be
  destroyed once every call that was passed it has **returned** (no GPU work reads it:
  `imd_pack`'s kernel arguments are copied at launch; plans copy the key). A plan outlives
  the handles it was created with; `imd_contract_packed` needs *a* handle with the same
  key (it may be a new one).
- **The packed buffer** is caller-owned device memory on the plan's device. `imd_pack`
  writes it on its stream; `imd_contract_packed` only reads it, so **any number of
  in-flight contracts may read one packed buffer concurrently**, on any streams
  (`pos_stream_events`, witness `neg_witness_concurrent`). The caller orders, as for any
  stream-ordered library: **RAW** — every contract that reads it after `imd_pack` on the
  same stream or behind an event (`neg_stream_no_raw`); **WAR** — a re-pack into it, a
  `cudaFree[Async]` of it, or any other write, after every contract reading it, by stream
  order or events (`neg_stream_no_war`, `neg_stream_free_early`).
- **The FP64 operand** must hold the values that were packed from the start of `imd_pack`
  until every contract using that pack has completed: K5's native path reads it for flagged
  outputs, and a result mixing two versions is neither contraction
  (`neg_stream_writex_early`). If X changes, re-pack.
- Two `imd_pack` calls into one buffer must not overlap (like a workspace, §6 rule 1);
  into distinct buffers, any concurrency.
- **Capture**: `imd_pack` and `imd_contract_packed` only launch kernels on the caller's
  stream (no synchronization, allocation or host↔device copy), so both may be captured
  into a CUDA graph; a replay keeps the captured pointers. Plan and pack creation are not
  capturable (as before; pack creation does no device work at all).
- **[spec: PackedOperand StreamSpec: NoStaleRead, PackReadsX1, CompleteAtEnd,
  Termination]**; host-side handle lifetime follows HostAPI's pattern (to be added there
  in implementation).

### 6b.8 Semantics summary (amends §1)

For a contraction with packed operands, C is a pure function of (input bits, α, β, N,
rounding, the canonical orders, library format), where the orders are the packs' (§6b.2).
For unflagged outputs the orders do not matter, so C equals the unpacked v1.5 result; for
flagged outputs it equals the unpacked result in the pack's K order. Results do not depend
on whether the pack was made by this or an earlier call, g, parts, chunking, K4
configuration, `vec_mode` or KF. **[spec: PackedOperand SigmaEqual, WEqual, CEqual,
EmulatedOrderFree over N ∈ {2, 3}, RN and TRUNC, both roles, g ∈ 1..N, one or two packed
operands, non-finite inputs]**

### 6b.9 Implementation-phase spec items (done in v1.7)

- **IndexMap** (constants `Packing`, `PackEnum`; 28 pos / 42 neg): lowering with adopted
  orders (every invariant, EinsumPreserved included, on the adopted lowering for any order
  within a class), the slot key, `PackKeySound` (equal keys ⇒ the pack's §6b.4 image is the
  plan's operand buffer, role B at batch stride r·k̂), `PackComplete` (a pack with the plan's
  classes is accepted in any order), `DescForReproduces`, `PackNoOOB`, PACK_MISMATCH after §5
  rows 1–5 and before κ ≥ T (`neg_pack_after_kappa`); negatives: fusion not redone,
  lowering not adopted, plan not adopting, key fused over all carriers, key without strides,
  v1 B-role stride, no tail.
- **Pipeline** (constant `Packed`; 39 pos / 47 neg): a packed slot — no K1 / KL / K2 / K3
  for it, KF on the in-line operand only, sX / fX / RX absent, K4 reads the pack (plane l at
  l·b·r·k̂, batch stride r·k̂, the tail), K5 its σ / flags; negatives: the full workspace,
  K1 or K3 still launched, K4 reading RX, K5 reading sB, the v1 B stride, no tail, the pack
  still in flight (the caller's RAW).
- **HostAPI** (constants `PackMode`, `NewHandle`, `ShareBuf`; 8 pos / 17 neg): the handle
  read by imd_pack / imd_plan_create_packed / imd_contract_packed from several threads,
  destroyed after return, a plan outliving it, a new handle with the same key;
  negatives: a lazily filled handle cache, a plan keeping a pointer to the handle, a GPU
  kernel reading it, two imd_pack into one buffer.
- `params.py cert_pack_layout`; oracle `pack()`, `pack_desc_for`, `lower_packed`,
  `contract_tensor(..., packs=...)`; the GPU test of every packed byte (tests/test_packed.py).
- 284 TLC models, all as expected.

### 6b.10 Design questions — resolved: every recommendation (the first option of each) adopted by Eric, 2026-10-03

1. **Entry point.** A new `imd_contract_packed` with `imd_operand_t` (recommended: the v1
   `imd_contract` signature and ABI stay untouched; a packed plan through `imd_contract`
   is PACK_MISMATCH) — or widen `imd_contract`.
2. **Plans take handles.** `imd_plan_create_packed(…, imd_pack_t, imd_pack_t)`
   (recommended: one key computation, the handle is what users hold) — or descriptors
   (`imd_packing_t`, as the host queries take).
3. **Contract-time key check through a handle** (recommended: O(rank) host compare, catches
   a buffer of the wrong pack *kind*; it cannot catch a wrong buffer of the right kind) —
   or no handle at contract time (plan-time check only, slightly leaner).
4. **FP64 data required at contract** (recommended: bit-exact, flagged outputs need it;
   costs nothing unless an output is flagged) — or optional with a documented deviation
   (`data = NULL` ⇒ flagged outputs are NaN), which lets a caller drop X after packing.
5. **Orders.** AUTO = X's stride order by default plus GIVEN and `imd_pack_desc_for`
   (recommended) — or GIVEN only, or AUTO only. With AUTO only, C writes in the B role and
   the partner's gather may be strided (§6b.2), and flagged outputs may differ from the
   unpacked call.
6. **Role-agnostic layout with a tail** (recommended; one pack serves both post-swap roles,
   ≤ 15·k̂ extra bytes) — or rows padded to ⌈r⌉₁₆ (simpler K4 strides, up to 16× bytes for
   r = 1), or per-role packs (two layouts, two keys).
7. **One handle for both roles** (recommended: follows from 6; the swap stays C's).
8. **TRUNC packs** allowed (recommended: rounding is just a key field) — or RN only.
9. **g-free layout** with all N planes (recommended; g stays an in-line workspace knob) —
   or g-dependent packs (no benefit: a group is a contiguous slice either way).
10. **Partner-only K labels → PACK_MISMATCH** (recommended) — or support them (needs a
    stride-0 read of the packed row along that label, which K4's dense TN operand cannot
    express).
11. **Statuses.** New PACK_MISMATCH (8) with a reason code in `imd_lowering_packed`
    (recommended); a short packed buffer reuses INSUFFICIENT_WORKSPACE (recommended) — or a
    new INSUFFICIENT_PACK.
12. **No header in the packed buffer** (recommended: pure data, the bytes are a function of
    key and values; `key_digest` exists for diagnostics only) — or a device-side header
    with the digest, checked by a debug build.
13. **`imd_pack` takes a workspace** sized by the host API (recommended: keeps the pack
    kernel free, e.g. D4's fused pack or a split long-row K1 may need scratch; 0 if not) —
    or no workspace parameter.
14. **K-empty packs → NOT_SUPPORTED, DIRECT plans accept and ignore packs** (recommended) —
    or reject packs in DIRECT plans.
15. **Interaction with C1 (adaptive precision)**: N is in the key, so a pack serves one N.
    An Ozaki-I slice format (C1(c)) would be a new `IMD_PACK_FORMAT` where one pack serves
    every N′ ≤ N; out of scope here (recommended: keep N in the key for format 1).

### 6b.11 Implementation notes (v1.7)

**Resolved (Eric, 2026-10-04):** the null-pointer reading below is accepted as implemented: a missing or mismatched pack handle is PACK_MISMATCH; a missing packed buffer is INVALID_VALUE only when the call would read it.

What §6b left to the implementation, as implemented (libimmerdink_host `src/host/pack.cpp`,
`analysis.cpp analyze_packed`; libimmerdink `src/api.cpp`, `src/contract.cu`; the oracle
`spec/oracle.py` transcribes the same choices and the tests compare every one).

- **Adoption by mode position.** The key is label-free, so a plan maps the pack's labels
  onto its own operand's by **mode position** in X's descriptor (pack label at position p ↔
  plan label at position p). The ranks must be equal and the map a bijection consistent at
  every position (a repeated label repeats in both); otherwise PACK_MISMATCH `_CLASSES`. The
  same tensor under renamed labels matches; a re-described tensor (another rank, extent-1
  modes added) is rejected (the key is not complete, §6b.2: safe).
- **Plan-time order** (imd_plan_create_packed, imd_lowering_packed): the argument checks and
  §5 rows 1–5 of (A, B, C, options); for the host queries then each pack **descriptor's** own
  status (imd_pack_layout's INVALID_VALUE / NOT_SUPPORTED: a handle is valid by
  construction); then per packed operand, the caller's A first: `_FORMAT`, `_MODULI`,
  `_ROUNDING`, `_CLASSES` (position map; a K label X does not carry is `_CLASSES` by the
  §6b.10 #10 policy, checked explicitly: a pack with an equal stride-0 mode would otherwise
  match, soundly); then, both packed, `_ORDER` (the mapped K or L orders differ); then the
  plan adopts the orders (K and L from the packs, each packed operand's row class from its
  pack; §5 step 9 fusion redone in them) and compares each slot key word for word:
  `_EXTENTS` (b, r, k, a mode count or extent) or `_STRIDES`, the first differing word;
  then §3.2 RN acceptance last. imd_workspace_size_packed / _layout_packed stop before the
  RN rule. A plan whose C has an extent-0 C-only label (lowered to I, §5 step 7) has that
  label in its I class: `_CLASSES` for a pack of the A side (empty output; conservative; the
  IndexMap model makes the same choice, witness `pk_conly0`).
- **Which checks are safety rules** (spec IndexMap): the key alone rejects every unsound
  pack (`PackKeySound`). The `_ORDER` check is implied by the key (a policy and a reason
  code: `pos_pack_order_policy`; it also rejects rare sound pairs whose K strides
  coincide). The partner-only `_CLASSES` check is implied by the key except at k = 0, where
  the slot key's (e, 0) mode fuses after an extent-0 mode and matches — harmless, a k = 0
  plan never reads the pack (`pos_pack_partner_policy`, witness `pk_po0`); so §6b.2's "a
  partner-only K label appears as a mode (e, 0) and mismatches" holds for k ≥ 1 only, and
  the explicit check keeps the §6b.10 #10 rule for every k. With both packed, K and L are
  adopted from either pack (equal after the `_ORDER` check).
- **imd_lowering_packed with PACK_MISMATCH**: `lowered = 1`, the summary fields filled as
  for NOT_SUPPORTED (the packed workspace of the packed roles, launches), plan bytes 0.
- **imd_pack_layout**: with INVALID_VALUE only `struct_size` and `status` are written; with
  NOT_SUPPORTED the sizes and the layout too (`key_digest` 0, as for any non-SUCCESS).
  `key_digest` = FNV-1a over the key words as little-endian int64.
- **imd_pack_desc_for**: `out->struct_size` is an input (≥ this version's size);
  `capacity` ≥ the number of distinct labels of X (≤ X's rank), else INVALID_VALUE; lists
  written K, then L, then R into `labels`. X's extent-1 labels (absent from the plan's
  orders) are appended to their class by membership (not in C: K; in C and the partner: L;
  else R). Statuses: the argument checks and §5 rows 1–5 (not the RN rule).
- **pack_ws_bytes** = align₂₅₆(8·(r + k + b)) (0 for an empty pack): imd_pack keeps X's
  row, k̄ and batch scatter vectors there. **imd_pack launches two kernels**: one writes the
  vectors and every byte of the packed buffer KL does not (the two alignment gaps, the
  tail and the last gap: no byte twice), then the v1.6 KL kernel on X as a one-operand view
  (rows m = r, g = N: every plane is "group 1", rows_pad = r) writes σ, the flags and all N
  planes with zero K padding (bit-identical to K1 + K2/K3 by v1.6). Reading along rows,
  rows shorter than 512 k̄ get taller KL tiles (up to 64 rows; a launch choice).
- **KF / KL with a packed operand**: KF's tiles and RB-row zeroing cover the in-line operand
  only, `fused` = 0 when both are packed; `fused_large` has no bit of a packed operand (also
  under the test hook). `kernel_launches` = (fused ? 1 : v + #in-line) + per group the
  in-line operands' packs (B: + the zero-rows launch) except group 1's after KF / KL +
  gemm_launches + 1; both packed: v + gemm_launches + 1.
- **K4 on a pack**: the operand pointer of group g₀ is `res + g₀·b·r·k̂` with batch stride
  `r·k̂` (both roles; for A it equals the workspace's m·k̂), passed per call.
- **imd_pack_t** holds a copy of X's descriptor (the position map; X's byte interval for
  imd_pack's ALIASING check), the pack lowering and key, and imd_pack's launch data; plans
  copy only the key (`plan_host_bytes` + 8 per key word).
- **Clarification of §6b.5 `imd_contract_packed` step 1 (to confirm).** Read literally,
  "`packed` non-NULL iff the plan packs it" makes a NULL `packed` of a packed slot a
  PACK_MISMATCH at step 1, so step 2's "plus `packed` when α ≠ 0, k ≠ 0, EMULATE and
  `pack_bytes > 0`" could never fire. Implemented so that every rule of §6b.5 is reachable:
  step 1 is "an **in-line** slot with `packed` or `pack` non-NULL, or a **packed** slot whose
  `pack` is NULL or has another key → PACK_MISMATCH"; a packed slot's NULL `packed` is step
  2's INVALID_VALUE when it would be read, and allowed when it is not (α = 0, k = 0, DIRECT,
  an empty pack). Step 6's `packed_bytes < pack_bytes` applies to a non-NULL `packed`.
  Tested (tests/test_packed.py suite api).

## 6d. Python front end (ROADMAP B1) [v1.9]

`python/immerdink` gains the device API: contractions on caller-owned CUDA arrays from
Python. It is a **binding**, not a second implementation: every number (sizes, legality,
statuses) and every byte of the result comes from libimmerdink / libimmerdink_host through
the §6 / §6a / §6b C ABI; Python adds no numerics, no sizing logic, no data movement.

**Decisions (Eric, 2026-10-04).** ctypes over the C ABI, as `host.py` (no compiled
extension, no nanobind, no new build dependency; the C API stays the single source of
truth). Any CUDA array is accepted, with no array library required to import or use
immerdink: `__cuda_array_interface__` (v2/v3: CuPy, Numba, PyTorch) or DLPack
(`__dlpack__` / `__dlpack_device__`: CuPy, PyTorch, JAX), plus raw device pointers with
explicit shape and strides. Kernel scope (§6 "Kernel-level scope"): no host↔device copy, no
operand or workspace allocation, the caller's stream; plans are cheap handles with explicit
lifetimes and exact footprints.

### 6d.1 Modules

| Module | Contents |
|---|---|
| `immerdink/_capi.py` | ctypes declarations of `immerdink.h` (every function, `imd_tensor_t`, `imd_options_t`, `imd_pack_desc_t`, `imd_packing_t`, `imd_operand_t`, with size asserts), library discovery and loading of libimmerdink (and its host library). No logic. |
| `immerdink/_arrays.py` | The array protocols: `__cuda_array_interface__`, DLPack capsules (read with `ctypes.pythonapi`), raw pointers (`DeviceArray`), stream objects. Pure Python, no CUDA call. |
| `immerdink/device.py` | `Handle`, `Plan`, `Pack`, `PlanCache`, `Layout`, `contract`, `workspace_size`, `pack_spec`, the status exceptions. |
| `immerdink/host.py` | unchanged (§6a); the device API reuses its public queries (`lowering`, `pack_desc_for`, `pack_layout`, `PackSpec`, `Desc`, `ImmerdinkError`). |
| `immerdink/planner.py` | [v1.11, §6e] the host queries for hot paths: `Query`, `Planner`, `Memo`, lazy result views over host.py's C structs (bit-identical results, ~1–2 µs per query). |

`import immerdink` loads no library and needs no GPU; libimmerdink is loaded on first use:
`lib=` / `immerdink.load_library(path)`, else `$IMMERDINK_LIB` (the .so), else
`$IMMERDINK_LIB_DIR/libimmerdink.so`, else `$IMMERDINK_BUILD_DIR` or `<repo>/build`, else
the system search path (the order of `host.py`). The host queries a device object makes go
to the `libimmerdink_host.so` next to that libimmerdink (the one it links), so both report
the same numbers.

### 6d.2 API

```python
import immerdink as imd

plan = imd.Plan("kjbc,kiac->abij", A, B, C, N=14, g=0, rounding="RN", mode="EMULATE",
                vec_mode="GENERATE", k4_tune=False, packs=None, handle=None)
plan.workspace_bytes, plan.device_bytes, plan.host_bytes   # exact (§4a, §6a)
plan.lowering                                              # host.Lowering of this plan
plan.contract(A, B, C, alpha=1.0, beta=0.0, workspace=ws, stream=s, packed=None)  # -> C
plan.contract_ptr(alpha, a_ptr, b_ptr, beta, c_ptr, ws_ptr, ws_bytes, stream=0, packed=None)
plan.close()                                               # also: with-block, finalizer

imd.contract("ik,kj->ij", A, B, out=C, alpha=1.0, beta=0.0, N=14, ...,
             workspace=ws, stream=s, plan_cache=None)      # einsum front end -> C
imd.workspace_size("ik,kj->ij", A, B, C, N=14, g=7)        # host library, no plan

spec = imd.pack_spec("ik,kj->ij", A, B, C, operand=0, N=14)   # imd_pack_desc_for (GIVEN)
pa = imd.Pack(spec)                                           # imd_pack_create
pa.pack_bytes, pa.workspace_bytes, pa.layout
pa.pack(A, out=bufA, workspace=pws, stream=s)                 # imd_pack
plan = imd.Plan("ik,kj->ij", A, B, C, N=14, packs=(pa, None)) # imd_plan_create_packed
plan.contract(A, B, C, workspace=ws, stream=s, packed=(bufA, None))   # imd_contract_packed

cache = imd.PlanCache()                  # imd.default_plan_cache is the default one
cache.get("ik,kj->ij", A, B, C, N=14)    # warm before a capture, then
cache.freeze()                           # a miss raises KeyError
cache.plans(), cache.device_bytes, cache.clear()
imd.set_cai_stream_policy("ignore")      # CAI v3 stream opt-out (default "check", §6d.3)
```

- **Operands** `A, B, C` of `Plan`, `contract`, `workspace_size` and `pack_spec`: CUDA
  arrays, or `imd.Layout(shape, strides=None)` (strides in **elements**, None = dense, last
  mode fastest) to plan before any array exists. `imd.DeviceArray(ptr, shape,
  strides=None, dtype="f8", readonly=False)` wraps a raw device pointer (element strides;
  it exports `__cuda_array_interface__` v3 itself). Raw descriptors with integer labels go
  through `host.Desc` (`Plan.from_descs(A, B, C, ...)`).
- **The spec** is `"A,B->C"` with single-character labels (as `host.contraction`): explicit
  output, no ellipsis; repeated labels follow §5 (diagonals in A/B allowed, in C
  NOT_SUPPORTED). Label ids are the code points.
- **Options** as `host.py`: `N` (required, no default: it is the accuracy choice), `g`,
  `rounding` "RN" | "TRUNC", `mode` "EMULATE" | "DIRECT", `vec_mode` "GENERATE" |
  "RESIDENT", `k4_tune` bool.
- **Buffers** (workspace, packed buffers, `imd_pack`'s workspace): any CUDA array of any
  dtype, contiguous (its bytes = size × itemsize), a `DeviceArray`, or a `(ptr, nbytes)`
  tuple; None only when the C function accepts NULL (0 bytes needed).
- `contract_ptr` / `Pack.pack_ptr` are the raw entry points for hot loops and other
  bindings: integers only, no Python-side layout check (the caller vouches that the
  pointers match the plan's descriptors); the C checks (null, aliasing, alignment, sizes)
  still run.

### 6d.3 Arrays → `imd_tensor_t`

| Protocol | pointer (element 0) | shape | strides → elements | dtype | readonly | device |
|---|---|---|---|---|---|---|
| CAI v2/v3 | `data[0]` | `shape` | `strides` bytes / 8 (None: C-contiguous) | `typestr` `<f8` | `data[1]` | CUDA by definition (not checked) |
| DLPack (0.x, 1.x) | `data + byte_offset` | `shape` | `strides` (NULL: row-major) | code 2 (float), bits 64, lanes 1 | 1.x flag `READ_ONLY` | `kDLCUDA` (2) or `kDLCUDAManaged` (13), else TypeError; a `kDLCUDA` device id other than the plan's device: ValueError |
| `DeviceArray` | `ptr` | `shape` | `strides` | `dtype` | `readonly` | caller's word |

- **dtype must be float64** for A, B, C and a pack's X (TypeError otherwise). A byte stride
  not divisible by 8 is TypeError (`imd_tensor_t` strides are elements); any sign and zero
  are passed through, so the §5 rules decide (negative strides fine; zero strides fine in A
  and B; in C the sorted-|stride| rule rejects them, INVALID_VALUE at plan creation).
  `mask` (CAI) must be absent.
- The pointer is that of element (0, …, 0), as `imd_contract` expects; with negative
  strides it is not the lowest address (both protocols define it this way).
- CAI is preferred when an object has both (a dict lookup, no capsule). DLPack is asked
  with `max_version=(1, 0)` (falling back to the 0.x call). In a call it is asked with
  `stream=` the call's stream (1 for NULL, the protocol's spelling of the legacy default
  stream): the **producer** orders its pending work before that stream (its rule). When
  only the layout is read (plan creation, `pack_spec`, `workspace_size`, `PlanCache.get`) it
  is asked with `stream=-1` ("the producer must not synchronize"). The capsule is read and
  dropped: immerdink takes no ownership (the caller keeps the array alive while its GPU work
  runs, as for any stream-ordered library).
- **CAI v3 `stream`** (reviewed 2026-10-05). immerdink never synchronizes streams on the
  caller's behalf (an event record / wait would be a hidden CUDA call and break capture), so
  it cannot honour the consumer synchronization CAI v3 asks for. Policy, process-wide:
  - **"check"** (default): in `Plan.contract`, `immerdink.contract` and `Pack.pack`
    (operands, workspaces and packed buffers alike), an array whose CAI `stream` is not
    None and not the call's stream raises ValueError, telling the caller to pass the
    producer's stream or to opt out. The call's NULL (None / 0) and a CAI 1 are the same
    stream (libimmerdink is not built with per-thread default streams). An integer compare:
    no CUDA call, capture-safe.
  - **"ignore"**: the opt-out the CAI v3 specification sanctions for consumers (as Numba's
    `NUMBA_CUDA_ARRAY_INTERFACE_SYNC=0` and CuPy's `CUPY_CUDA_ARRAY_INTERFACE_SYNC=0`): the
    field is not looked at and ordering across streams is the caller's, as in §6.
    `immerdink.set_cai_stream_policy("ignore")`, or `IMMERDINK_CAI_SYNC=0` in the
    environment at import. Not a per-call keyword.
  - No event-wait mode. `contract_ptr` / `pack_ptr` take integers and check nothing.
- **Validation in Python (what the C API cannot see).** A plan is created for fixed
  extents and strides, so `Plan.contract` checks that each array's shape equals the plan's
  and its strides equal the plan's on every mode with extent > 1 (strides of extent-1
  modes, and of an empty tensor, are ignored, as §5 step 4 drops those modes): ValueError
  otherwise. C must be writable (ValueError); `packed` buffers and workspaces too. Pack.pack
  checks X against the pack's descriptor the same way. Everything else is the C API's:
  null pointers, ALIASING, MISALIGNED (8-byte operands, 256-byte workspace and packed
  buffers), INSUFFICIENT_WORKSPACE, PACK_MISMATCH. Python does not query pointer
  attributes (the device of a CAI pointer is not checked; a DLPack device is).

### 6d.4 Statuses → exceptions

Every non-SUCCESS status raises a subclass of `host.ImmerdinkError` (a RuntimeError with
`.status`, the status name) that also derives from the matching builtin, so callers can
catch either:

| Status | Exception |
|---|---|
| INVALID_VALUE | `InvalidValueError(ImmerdinkError, ValueError)` |
| NOT_SUPPORTED | `NotSupportedError(ImmerdinkError)` |
| ALIASING | `AliasingError(ImmerdinkError, ValueError)` |
| INSUFFICIENT_WORKSPACE | `InsufficientWorkspaceError(ImmerdinkError, ValueError)` (the message names the bytes needed) |
| MISALIGNED | `MisalignedError(ImmerdinkError, ValueError)` |
| CUDA_ERROR | `CudaError(ImmerdinkError)` |
| ALLOC_FAILED | `AllocFailedError(ImmerdinkError, MemoryError)` |
| PACK_MISMATCH | `PackMismatchError(ImmerdinkError, ValueError)` (plan creation: `.reason`, the `imd_lowering_packed` pack_mismatch name) |

[v1.12] Exception: the host queries that return a plan-time status in `.status`
(`host.predict`, `host.min_moduli` and their planner views) do not raise. `min_moduli` returns
INVALID_VALUE (e.g. a request without a target) and the lowering's statuses in `.status`.

Python-side argument errors (not CUDA memory, wrong dtype, shape or stride mismatch with the
plan, readonly output, malformed spec) are plain TypeError / ValueError, raised before any C
call.

### 6d.5 Streams, capture, workspace

- `stream`: None or 0 → NULL (the legacy default stream, as `imd_contract` defines it;
  libimmerdink is not built with per-thread default streams); an int is the
  `cudaStream_t` value itself (1 = `cudaStreamLegacy`, 2 = `cudaStreamPerThread`); an
  object with `__cuda_stream__` (the CUDA Python protocol, `(version, handle)`),
  `.cuda_stream` (PyTorch), `.ptr` (CuPy) or a `ctypes.c_void_p`. Anything else is
  TypeError.
- **Capture-safe.** `Plan.contract`, `contract_ptr`, `Pack.pack` and `contract` with a
  cached plan make no CUDA runtime call of their own (no synchronization, allocation, event
  or copy): parsing is pure Python; the only CUDA work is the library's launches, which
  §6 / §6b.7 make capturable. A DLPack producer's own stream ordering is its business.
  Plan and pack creation are not capturable (§6), so a `contract` whose plan is not yet
  cached is not either: create the plan, or call once, before capture.
- **Workspace**: the caller's, exactly as §6: `plan.workspace_bytes` (from
  `imd_workspace_size[_packed]`, never recomputed in Python) bytes, 256-byte aligned, at
  most one in-flight call per workspace. immerdink allocates none: with
  `workspace_bytes > 0` and `workspace=None` the C null rule applies (InvalidValueError,
  message with the size). `imd.workspace_size(...)` answers before any plan exists.

### 6d.6 Lifetimes, threads, plan cache

- `Handle`: `imd_create` on the current device. A default handle per device
  (`imd.default_handle()`) is created on first use and lives for the process. Plans and
  packs hold a reference to their handle, so the §6 rule (the handle outlives every plan
  creation) holds by construction.
- `Plan` and `Pack` own their C object: `close()` (idempotent), `with`, or a
  `weakref.finalize` when the last reference goes. A plan is bound to the device current at
  its creation (`Plan.device`; `imd_plan_create` uses it); `Plan.lowering` uses that device's
  SM count and reports the plan-time rule's K4 configuration (a `k4_tune` plan may run
  another, bit-identical one).
- **Packs.** A packed plan passes its packs' handles to `imd_contract_packed` (§6b.7: a
  handle with the plan's key), so it holds them: `Pack.close()` closes the pack for new use
  (`pack()`, new plans: ValueError naming the pack) at once, but `imd_pack_destroy` waits
  until the last open plan made with it is closed or collected (a count kept by the plans).
- **GENERATE** plans may be closed as soon as the calls returned (§6 lifetime), which every
  Python reference pattern guarantees. **RESIDENT** plans must outlive the GPU work of their
  calls (§6), and **graph replays are GPU work of the plan**: a replay reads the resident
  vectors, and finalizing the plan `cudaFree`s them. So a RESIDENT plan, or the cache
  holding it, must stay alive while any CUDA graph that captured it may be replayed, and its
  streams must be synchronized before `close()` or dropping the last reference. Python
  cannot know (that would need a host synchronization); a RESIDENT plan finalized without
  `close()` emits a `ResourceWarning`.
- **Threads**: a `Plan` is immutable after creation; any number of threads may call
  `contract` on one plan (ctypes releases the GIL during the C call) under §6's rules: an own
  workspace per in-flight call, and no `close()` while another thread is in a call.
- **`PlanCache`** (used by `imd.contract`): a plan is a pure function of (descriptors,
  options, packs, device), so a cache keyed by (spec, shapes, element strides, options,
  current device) returns a plan equivalent to a fresh one. Thread-safe (a lock around the
  dict; plan creation outside the lock; a losing duplicate, never used, is closed).
  - It never evicts on its own. A bounded variant (deferred) would have to **drop
    references** rather than call `close()`, for the reason below.
  - `clear()` empties it. GENERATE plans are only **dropped**: `close()` on a plan another
    thread is using in `contract()` would be a host use-after-free; each is destroyed by its
    finalizer once no reference remains. RESIDENT plans are **closed**, which is valid only
    after the streams of their calls were synchronized and while no CUDA graph that
    captured them will be replayed.
  - `plans()` lists the distinct plans held; `device_bytes` is the sum of their
    `imd_plan_device_bytes` (the RESIDENT vectors).
  - `freeze()`: a lookup that would create a plan raises KeyError, so a caller (ehrenfest)
    can assert every plan was made before a capture (`freeze(False)` thaws it).
  - `imd.default_plan_cache` is the default; `plan_cache=False` makes a plan per call and
    closes it after the call returns (GENERATE only; RESIDENT → ValueError).
- `cudaGetDevice` (host-only, capture-safe) supplies the device of the default handle and
  of the cache key, and `cudaDeviceGetAttribute` the SM count `Plan.lowering` passes to
  `imd_lowering` (not on the call path); they are the only CUDA runtime functions the
  package calls, through the libcudart libimmerdink already loaded.

### 6d.7 Per-call overhead

A plan precomputes everything a call needs that does not change (expected shapes and byte
strides per operand, the ctypes function with its argtypes, the in-line operand structs'
static fields, the error context); a call extracts three pointers, compares a few tuples
and calls C. Measured (`bench/python_overhead.py`: "ik,kj->ij" 16³, N = 8, host time per
call, median of 200 interleaved rounds of 50 calls, RTX 4090 box, 2026-10-05):

| Entry point | µs per call | over the bare ctypes call |
|---|---|---|
| `lib.imd_contract(...)` via ctypes, prebuilt integer arguments | 3.9 | — |
| `Plan.contract_ptr` | 3.95 | +0.05 |
| `Plan.contract`, `__cuda_array_interface__` arrays (layout checks included) | 4.85 | +0.9–1.0 |
| `immerdink.contract`, cached plan, CAI arrays | 5.9 | +1.9–2.1 |
| `Plan.contract`, DLPack arrays (the tests' hand-built producer) | 19.3 | +15.4 |

The bare call is the library's own enqueue (fused KF path: three launches) plus ctypes.
DLPack costs per array about 2.7 µs in the producer (capsule construction; a real
library's is its own) plus about 2 µs in the consumer (`__dlpack_device__`, capsule name and
pointer, three reads), so CAI is preferred when both exist.

### 6d.8 Tests and verification

- **No TLA+ change.** The package adds no state the C API does not already model: plans,
  packs and handles are the C objects with the same lifetimes (`HostAPI`: shared plan,
  early destroy, resident plans; `PackedOperand` / `HostAPI` pack handle); the plan cache
  memoizes a pure function and only hands out immutable plans (`pos_shared_plan`); a cache
  never destroys a GENERATE plan, it only drops references; finalizers run only when no
  Python reference (hence no Python call) remains, which is the GENERATE lifetime rule; the
  RESIDENT rule (graph replays included) stays the caller's, as in C. The packs' use count
  only delays `imd_pack_destroy` past every plan that passes the handle, which is stronger
  than §6b.7 requires.
- `python_device` (ctest, GPU lock, allocation shim preloaded; pytest, no CuPy/PyTorch
  needed: a ctypes libcudart helper allocates and copies, a minimal CAI object and a
  hand-built DLPack capsule (0.x and 1.x) present the memory): results bit-exact against
  `spec/oracle.py` (GEMM, permuted and batched rank ≥ 4 tensors, non-contiguous and negative
  strides, diagonals, a scalar output, DIRECT, α = 0, β ≠ 0, packed A / B / both via
  `pack_spec`, every packed byte equal to `oracle.pack`), and equal to the direct C call;
  `contract_ptr`; plan reuse over many calls and many plans alive at once; the plan cache
  (threads, `clear()` dropping GENERATE and closing RESIDENT plans, `plans()`,
  `device_bytes`, `freeze()`); stream objects of every accepted kind; the CAI stream policy
  (mismatch raises, NULL ≡ 1, opt-out); DLPack layout reads with stream −1 and the device-id
  check; `Pack.close()` while plans use the pack; the ResourceWarning of an unclosed
  RESIDENT plan; CUDA-graph capture and replay of `Plan.contract`
  and `imd.contract`; no CUDA allocation in any call (shim); every error row above (wrong
  dtype, host memory, shape / stride mismatch, readonly C, missing / short workspace,
  misaligned and aliased pointers, NOT_SUPPORTED, INVALID_VALUE, PACK_MISMATCH); the ctypes
  declarations against `immerdink.h` (every function declared). CuPy / PyTorch interop
  tests run when the module is importable, else skip.

---

## 6c. Performance model (ROADMAP B2) [v1.8, v1.10]

A CUDA-free prediction of what one `imd_contract` (or `imd_pack`) costs on a given GPU, per
phase and in total, and a helper that ranks the configurations a caller may choose (N, g,
`vec_mode`, packed operands) by predicted time inside a device-memory budget, each with its
exact footprint. It is designed to be called directly by ehrenfest, which therefore needs no
model of its own. **Decided by Eric:** it lives in `libimmerdink_host` next to the lowering
(one implementation; Python wraps it); a calibration for the RTX 4090 (sm_89) ships with the
library and a recalibration tool writes a calibration file for another GPU, which the model
loads instead; output = predict + rank, the caller makes the final choice; kernel level only
(no streams, graphs or transfers); accuracy and the choice of N are the caller's, the model
only reports each candidate's plan-time status (which includes the §3.2 RN acceptance rule).

Nothing here changes what a plan does: the calibration is read only by the `imd_perf_*`
functions, never by plan creation (`k4_select` keeps its fixed A3 constants, §4), so a
calibration file cannot change a result, a launch or a footprint.

### 6c.1 Model form

`t_call = Σ_phases t_p + t_call-term`, each term **linear in calibrated parameters**:

```
t_p = Σ_j θ[class(p), j] · x[p, j]        (ns; θ ≥ 0 except the call term; j < 6)
K4:   t_p = max(Σ_{j≠4} θ_j x_j, θ_4 x_4)  (the A3 round model, or the DRAM term)
```

The features `x` are exact functions of the plan's lowering (the `ExecPlan` that
`imd_lowering` / `imd_lowering_packed` build: GEMM view, dispatch, KF / KL choice, packed
roles, groups, K-chunks, K4 configuration), computed by the host library only. The parameters
`θ` are per **kernel class** and physically meaningful: ns per launch (launch + fixed
latency + the bandwidth ramp of a small grid), ns per DRAM byte (1 / effective bandwidth),
ns per residue or CRT operation (1 / effective instruction throughput), ns per byte re-read
from L2, and for K4 the A3 round model (ns per round, ns per MAC at q concurrent CTAs) and ns
per DRAM byte beyond L2. A phase's time is a sum of terms (no overlap, made accurate by the
fit: each kernel's terms are fitted on that kernel's own measurements), except K4, which is
the roofline max of its compute (round) model and its DRAM term: K4 is either, and a sum
would charge every compute-bound GEMM for its reads. Linear parameters make each fit a small
non-negative least-squares problem (K4: one more scan over the DRAM rate), deterministic
and reproducible.

Notation (post-swap view, §1.1): `rows_A = b·m`, `rows_B = b·n`, padded rows `pad_A = b·m`,
`pad_B = b·n̂`, elements `E_x = rows_x·k`, `V` = `vec_entries`, `G = ⌈N/g⌉` groups of sizes
`g_1 = g, …, g_G = N − (G−1)·g`, `L32(N)` = CRT limbs (§2), β' = [β ≠ 0], `f` = the eps
fraction (below), `L2` = the calibration's `l2_bytes`. Only **in-line** operands x (not packed,
§6b) have scale and pack phases.

| phase (bench tag) | when | class | x₀ | x₁ | x₂ | x₃ | x₄ |
|---|---|---|---|---|---|---|---|
| `K0 vec` | `IMD_VEC_GENERATE`, no KF (any dispatch but NOOP) | K0 | 1 | `8V` written | | | |
| `KF fused` | `fused` (§4) | KF | 1 | `Σ_x (8E_x + g·pad_x·k̂ + 3rows_x) + 8V·[GENERATE]` | `g·Σ_x E_x` residues | `Σ_x rows_x` | `f·x₂`; x₅ = the busiest SM's chain (below) |
| `KL A`, `KL B` | `fused_large` bit of x | KL | 1 | `8E_x + g·pad_x·k̂ + 3rows_x` | `g·E_x` | `16·rows_x·max(0, k̂ − 2·tk_x)` L2 re-read | `f·x₂` |
| `K2 pack A`, `K3 pack B` | groups of x not packed by KF / KL | PACK | launches (B: + the zero-rows launch per group if n̂ > n) | `Σ_groups (8E_x + g_i·pad_x·k̂)` | `Σ_groups g_i·E_x` | `f·x₂` | |
| `K4 gemm` | EMULATE | K4[config] | rounds | MACs run at q = 1 | q = 2 | q = 3 | DRAM bytes beyond L2: `Σ_groups max(0, g_i·b·(m + n̂)·k̂ + g_i·b·m·n̂·(2·k_chunks − 1) − L2)`; x₅ = the busiest SM's MACs of K-chunks whose length is not a multiple of 64 (`mac_ktail`; [v1.10] format 2: of 32, below) |
| `K5 crt` | EMULATE | K5 | 1 | `N·b·m·n̂ + 3b(m+n) + 8bmn(1+β')` | `N·L32·b·m·n` limb operations | | |
| `DIRECT` | DIRECT dispatch | DIRECT | 1 | `8b(mk + nk) + 8bmn(1+β')` | `2bmnk` FP64 flops | | |
| `K5 beta-only` | SCALE dispatch or α = 0 | BETA | 1 | `8bmn(1+β')` | | | |
| `call` | every non-NOOP call | CALL | 1 | `kernel_launches` | | | |
| `pack prep` (imd_pack) | every non-empty pack | K0 | 1 | `8(r+k+b)` + the zeroed gaps and tail | | | |
| `pack KL` (imd_pack) | every non-empty pack | KL | as KL with g = N, rows = pad = b·r, x the pack's one-operand view (`pack_exec`) | | | | |

(x₅ is unused outside K4 and KF: room for one more feature per class in a later format.)

**[v1.10] `chain` and `wave` features** (format 2; the free slots): K0 x₂, PACK x₄, K5 x₃,
DIRECT x₃, BETA x₂ = `chain`, the busiest SM's per-thread compute chain of the phase's launches
(KF's x₅ is the same definition); KL x₅ and PACK x₅ = `wave`, the busiest SM's tile loads in
sequence (memory latency). Both are defined below ("Waves and occupancy"). The K4 classes
keep the round model (it already is a wave model). K5 x₄ = `dram`: the W plane bytes beyond the
effective L2, `max(0, N·b·m·n̂ − L2)` (K4 leaves at most the effective L2 of them resident; at
8192², N = 8 K5 reads 1.07 GB at 0.93 TB/s, which the byte term fitted on mostly resident planes
predicted 24 % low; refit on the v1.8 data: K5 phase median on ≥ 1 ms calls 8.7 → 4.4 %).

- **KF's per-thread chain** (`kf.chain`, added 2026-10-05 for N = 15–17 on mid-size shapes). KF
  launches B = tiles × parts blocks (the shared launch rules `kf_lg_tk`, `kf_operand_tiles`,
  `kf_small`, `pack_moduli_parts`: tiles of 4096 values with 16 per thread, or 1024 with 4 per
  thread when the 4096 grid has fewer than 16 blocks per SM even with every modulus split; parts
  when tiles < 2 per SM). Each thread emits the residues of its E values for ⌈g/parts⌉ moduli
  in sequence, so with few blocks per SM the busiest SM's chain, not the bytes, sets the time:
  `x₅ = ⌈B / (SMs · r)⌉ · E · ⌈g/parts⌉ · (1 + f)`, r = 2 resident blocks per SM for E = 16
  (47.5 KB of static shared memory), 4 for E = 4; an eps-path residue doubles the step. Measured:
  at N = 15, 576³ (E = 16, 288 blocks) takes 62 µs in KF and 448³ (E = 4) 20 µs; without the
  term KF at N = 15–17 on 576–960³ was predicted up to 45 % low.

- **[v1.10] Waves and occupancy** (generalizes KF's chain to every kernel). Every launch rule
  of the pipeline (grid, threads per CTA, values or outputs per thread, grid-stride) lives in
  `src/host/analysis.hpp` and is called by both `contract.cu` and the model, so the model sees
  the real launch shape: `launch_blocks` (the 256-thread grid of a grid-stride kernel, capped at
  2²⁰ blocks), `kf_*` (KF), `kl_launch` (KL: tile height, threads 512 / 1024, values per thread
  8 / 4, chunks per row; `pack_kl_lg_rows` for `imd_pack`), `pack_launch` (K2 / K3: tile
  height, tiles, moduli parts), `k5_launch` (K5: λ-fastest tile, or 1 / 2 / 4 outputs per
  thread). For a launch of B = Bx·By CTAs over `work` units with `r` CTAs resident per SM:
  `waves = ⌈B / (S·r)⌉`, `per_block = ⌈work / Bx⌉` (grid-stride iterations of one CTA) and
  **`chain = waves · per_block · u`**, `u` = the serial steps of one thread per work unit:

  | kernel | work unit | u (steps per unit) |
  |---|---|---|
  | K0 `k0_vec`, `kp_prep`, BETA, DIRECT | 256 outputs (one per thread) | 1; DIRECT: k FMAs |
  | KF | a tile | E · ⌈g/parts⌉ residues (× (1 + f)) |
  | KL (`wave`) | a tile of th rows | nc chunk loads (nc = chunks per row) |
  | K2 / K3 (`wave`, per group launch, summed) | a tile | 1 tile load |
  | K2 / K3 `k23_pack` (per group launch, summed) | a tile | 16 · ⌈gᵢ/parts⌉ residues (× (1 + f)) |
  | K5 j̄-fastest (E = 1 / 2 / 4) | 256 items of E outputs | E · N · L32 limb operations |
  | K5 λ-fastest | a 1024-output tile | 4 · N · L32 |

  `chain` counts compute steps (clock-bound), `wave` memory round trips (not clock-bound).
  **Adopted after the v1.8 data (refit, hold-out; NOTES-B2 §6):** a KL compute chain (nc · E ·
  g) fitted to 0 and the KL `wave` replaced it (KL B p90 22.7 → 19.2 %); the PACK `wave` takes
  K3 from 18.7 / 37.0 % to 12.2 / 32.4 % (median / p90 on ≥ 1 ms calls) and ≥ 1 ms calls from
  2.8 / 7.8 % to 2.1 / 5.8 %: a K2 / K3 launch of a few hundred tiles (B with n = 96 … 384) runs
  at ~0.35 TB/s, one ~6 µs tile round trip per wave, which the byte term at 0.9 TB/s did not see.
  A chain step is a thread's serial unit of work; for a grid much larger than one wave the
  chain is proportional to the work (collinear with the op features), for small grids it is the
  wave-quantized latency the byte and op features cannot see (one CTA too many is a second
  wave). `r` (CTAs resident per SM) is **measured** at calibration time for every kernel and
  variant (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the launch's threads and shared
  memory; with registers, static shared memory and threads from `cudaFuncGetAttributes`) and
  stored in the calibration (format 2, `kernel.<name>` lines); a format-1 calibration gets
  the built-in table of v1.8 (KF E = 16: 2, E = 4: 4, the others 2 by the shared memory of the
  tile kernels; their chain parameters are 0 there, so a format-1 file predicts exactly as in
  v1.8). K4's rounds use the configuration table's CTAs per SM (the plan-time rule's
  constants; `imd_bench --methods occupancy` reports the measured K4 occupancy next to it, so a
  difference is visible). The zero-rows launch of K3 counts as a launch only.
  **Adoption rule:** a chain term replaces an ad-hoc one (KF's fixed r = 2 / 4) only if the
  hold-out stays as good or better; otherwise the measured-r chain is fitted beside it and the
  difference is reported (NOTES-B2).
- **K4's partial K tile** (`mac_ktail`, found on the hold-out set). A K-chunk whose length is
  not a multiple of the 64-wide K tile (k̂ = 1008, 5008, 30000, 150000; every full chunk K_c' =
  131056) runs CUTLASS 2.x's predicated residue tile, and the whole chunk is measurably slower
  (sq5000, 3000 × 700 × 5000: K4 24–29 % above the round model without the feature; with it the
  K4 hold-out p90 on ≥ 1 ms calls drops from 19 % to 18 % and the 64×128 class from 28 % to
  15 %). Tested criteria: k̂ mod 64 (adopted), 128, 256.
  **[v1.10] Format 2 changes the criterion to k̂ mod 32 ≠ 0** (k̂ is a multiple of 16, so
  k̂ mod 32 = 16): the int8 residue rows (k̂ bytes) are then not 32-byte aligned, every row
  starts mid-sector and the K4 tile loads touch one more sector per row. It explains both the
  slow k̂ = 1008 / 5008 / 30000 / 150000 (mod 32 = 16) and the not-slow k̂ = 800 (mod 32 = 0, mod 64
  = 32, which the v1.8 term over-predicted by 26–35 %). Refit on the v1.8 data, hold-out:
  100 µs – 1 ms max error 29.1 → 14.8 %, N = 15–17 100 µs – 1 ms median / p90 8.0 / 15.1 →
  5.3 / 11.7 %; mod 128 was worse (≥ 1 ms median 3.9 %). The fine-k̂ sweep (NOTES-B2 §6)
  separates the criteria. A format-1 calibration keeps mod 64 (v6-identical predictions).
  **Per chunk (review 2026-10-05):** K4 reads chunk c of a row at `row·k̂ + c·K_c'` (lda = k̂,
  gemm_k4.cu), so the criterion is the row start of each chunk: slow iff `k̂ mod 32 ≠ 0` or
  `c·K_c' mod 32 ≠ 0`; K_c' = 131056 ≡ 16 (mod 32), so for an aligned k̂ the odd chunks are
  slow. This equals k̂ mod 32 for single-chunk calls (every measured case but two). **Evidence
  strength:** 128×256 (the rule's usual choice) has the full fine-k̂ sweep; 128×128 has the
  4864–5120 points, where k̂ = 4960 (mod 32 = 0, mod 64 = 32) is still +14 % (partly counter-
  evidence: that tile may also pay for the 64-wide residue tile); 64×128 and 64×64 are
  untested. The two-chunk check (512² × 262144, chunk 1 misaligned, vs 262160, all misaligned)
  is inconclusive: K4 there is DRAM-bound and 262144 (a power-of-two row stride) is the slower
  one (NOTES-B2 §6).
  **[v1.13] Correction (measured, NOTES-K4ALIGN; §4b).** The per-chunk rule above uses the
  chunk *start*. The measured mechanism depends on the chunk *end*. CUTLASS 2.x loads a
  chunk's residue tile (`kc mod 64` elements) first, so row r's main-loop loads sit at phase
  `(r·k̂ + e_c) mod 64`, where `e_c = k0 + kc` is the chunk end.
  - With K_c' = 131056 (≡ 48 mod 64), an aligned k̂ had its chunk 0 misaligned, not the odd
    chunks. The 262144 call was slower because of that, not because of a DRAM-channel
    effect of its power-of-two stride: 196608, also aligned, measured worst
    (2048² × 196608, N = 8: 23.4 TFLOPS-eq, 70.5 ms per call).
  - k̂ ≡ 16 mod 32 misaligns half of the rows (the phases cycle through all four 16-byte
    classes), not every row.
  - The slow-chunk count of format 2 happened to agree for every calibrated shape.
  - Since v1.13, K_c' = 131008 ≡ 0 mod 64. Every full chunk ends on a multiple of 64 and
    the last chunk ends at k̂, so the c·K_c' term is always 0 mod 32. The format-2 criterion
    becomes exactly "slow iff k̂ mod 32 ≠ 0" for the 16/48 phase classes. The class-32 cost
    (+3 % for 128×256 and +10 to +19 % for the smaller tiles, measured) has no term of its
    own.
  - The calibration format is unchanged (review 2026-10-05). The shipped θ was fitted with
    the old chunk length, and a later step refits it.

- **The eps path** (the `op_eps` features, reviewed 2026-10-05). The packs quantize an
  element to `μ·2^ε`; a residue costs one more modular multiply when ε > 0, i.e. |a'| ≥ 2⁵³
  (§3.4, the "rare" path of the pack emitter), and a warp pays it for its whole tile (32 lanes
  × up to 16 elements) when any element of the tile has ε > 0. This depends on the **data**,
  so it is an input: `eps_fraction` f ∈ [0, 1] in the requests (the fraction of pack work on
  the eps path), forced to 0 when T < 2⁵³ (N ≤ 13). The default (f < 0 in the request) is a
  plan-only **typical-data** estimate: rows of comparable entries are scaled to a norm of about
  T, so with the margin `D = log₂T − ½·log₂k − 53` (T exact, not ⌊log₂T⌋; log₂T = 54.58 /
  58.39 / 62.19 / 65.98 / 77.19 at N = 14 / 15 / 16 / 17 / 20) the fraction of 512-element tiles whose
  maximum after the row's power-of-two scaling reaches 2⁵³ is a function of D alone, k-free:
  simulated for the benchmark generator `(u − ½)·exp(g/2)` with random row scales (k = 1024 and
  8192 agree), 0.03 at D = −2.5, 0.23 at −2, 0.65 at −1.5, 0.96 at −1 (a tile's maximum is about
  1.6 bits above the RMS margin); the default is the ramp f = clamp((D + 2.5)/1.5, 0, 1). So at
  N = 14 only k < 286 is on the path (fully for k ≤ 36), at N = 15 fully for k ≤ 7 000 and not
  at all for k ≥ 56 000, at N ≥ 16 for every practical k. Rows with one dominant entry have f ≈ 512/k (one tile per row);
  callers pass their own f. Every prediction reports `eps_s`, the eps terms at f = 1, so the
  exact bracket for any data is `[time − f·eps_s, time + (1 − f)·eps_s]` ([v1.10] at a fixed clock:
BURST or an explicit `clock_mhz`; under SUSTAINED the eps work also changes the settled clock). The calibration
  identifies op_eps on the same plans with different data (`imd_bench --data dominant`), not
  only through N. Consequence: with the default f, doubling k can lower the predicted time
  (f falls from 1 to 0 as k grows); never below the time without the eps terms (tested).
- **K4's DRAM term.** The operands of a group were just written by the packs and stay in L2
  while the group's operands and W planes fit; beyond that K4 streams them from DRAM
  (measured memory-bound: r_mm-like n̂ = 64–128 with long k, skinny m ≫ n). The calibration
  holds an **effective** L2 (`l2_bytes`), fitted: the L2 is shared with W, the other operand
  and the packs' traffic, so the working set streams from DRAM well before the nominal size
  (on the 4090 the fit picks 0.4 × 72 MiB; with the nominal 72 MiB the memory-bound g < N
  points were up to 48 % low). The DRAM rate is per configuration: the three larger tiles
  agree (one rate), 64×64 is slower per byte, consistent with its narrower tiles re-reading
  the operands more often (NOTES-B2).

- **KL's tile** (`tk_x`): 4096 FP64 along k̄, 512 when the operand is read along rows
  (`pack_along_rows`, §4 KL "th · tk = 4096"); rows of more than two tiles re-read the middle
  chunks from L2 in passes 2 and 3 (the x₃ bytes). The tile rule moves into a host header that
  `contract.cu launch_kl` uses too (one implementation).
- **K4** (`class = K4[c]`, c = the plan's configuration: `k4_select` for the calibration's SM
  count S, or the request's override). For each group size g_i and each K-chunk length κ
  (`k_chunks − 1` chunks of K_c', then the last): tiles `T = ⌈m/tm⌉·⌈n̂/tn⌉·g_i·b`, the
  busiest SM's `j = ⌈T/S⌉` tiles run as `⌊j/o⌋` rounds of o CTAs and, if `ρ = j mod o > 0`,
  one round of ρ; x₀ += rounds, x_q += (rounds of q CTAs)·q·tm·tn·κ. With the A3 constants
  as parameters this is exactly `k4_estimate` (§4 "Selection", summed over groups and
  chunks): the same round decomposition, one function used by both. The batch (≤ 65535)
  and n̂ splits are not modeled (they only cut a very large tile count in parts).
- **K5 per N** (A5: one kernel per N): the N·L32 limb operations and the N·b·m·n̂ plane
  bytes carry the N dependence; [v1.10] the j̄-per-thread choice and the per-N occupancy enter
  through K5's chain (`k5_launch`, `kernel.k5_n<N>_e<E>`).
- **g**: the KF / KL group-1 terms use g_1 = g; groups 2..G are the PACK phase (one operand
  read per group); K4 is summed over groups (fewer, smaller launches as g shrinks).
- **`vec_mode`**: RESIDENT removes K0 and KF's vec bytes (and the call term's launch).
- **Packed operands** (§6b): a packed role has no KF / KL / PACK terms (KF covers the in-line
  operand only, `fused` = 0 with both packed); K4 and K5 are unchanged; `imd_pack` is
  predicted separately (`imd_perf_predict_pack`, phases `pack prep` + `pack KL` + call).
- **Batch dimensions and permuted (tensor) layouts** enter through the GEMM view (b, m, n,
  k), `pack_along_rows` and the dispatch; the model has no stride feature: permuted tensors
  run within a few % of the matrix case (ROADMAP "Where we are"); the hold-out set measures
  it (6c.5).
- **Small problems**: the affine x₀ + x₁ form absorbs a kernel's bandwidth ramp (an effective
  bandwidth `B·bytes/(bytes + b½)` is affine in bytes); K4's wave quantization is the round
  model. The **call term** (`θ₀ + θ₁·kernel_launches`, sign-free) maps the sum of the
  measured phases (event marks between kernels, §6c.2) onto the timed call: launch gaps the
  phases do not see, minus the marks' own overhead.
- **Bounds.** Each phase also reports its MEMORY_MODEL §4 lower bound with the actual g:
  `bytes` (the x₁ DRAM bytes above; K4: `N·b·(m + n̂)·k̂ + N·b·m·n̂·(2·k_chunks − 1)`), `ops`
  (K4: `2·N·b·m·n̂·k̂` int8; DIRECT: `2bmnk` FP64) and `bound = max(bytes / peak_bw, ops /
  peak)` with the datasheet peaks stored in the calibration. `imd_bench`'s per-phase "% of
  bound" takes them from here (it had its own copy of the formulas, at g = N).

#### [v1.10] Clock regimes: BURST and SUSTAINED

The GPU's SM clock is not a constant. Short or isolated calls run at the boost clock (the 4090:
2805 MHz in nearly every calibration record); long or back-to-back sequences of
tensor-core-heavy calls hit the board power limit (450 W) and the clock settles lower (2300–2450
MHz at 4096³, about 1950 MHz at 16384³). ehrenfest runs long repeated sequences. The model is
split into a time model at a given clock and a clock model per regime.

- **Time at clock f.** Each feature is either clock-bound (scales with the SM clock) or not:
  DRAM bytes (every `byte` feature, K4's DRAM term), `launch` and the call term are not; every
  other feature (op, op_eps, row, chain, l2byte, K4's round model) is. With the calibration's
  reference clock `f_ref`: `t_p(f) = Σ_{clock-bound j} θ_j x_j · f_ref / f + Σ_{others} θ_j x_j`
  (K4: `max(rounds · f_ref / f, DRAM term)`). **As calibrated (NOTES-B2 §6):** θ are the costs
  of the burst calibration state as measured, **not** normalized per record (normalizing by
  the NVML clock read after each timed loop, `perf_fit --clock-normalize`, made the post-BIOS
  hold-out worse: ≥ 1 ms median / p90 2.9 / 7.3 % vs 1.6 / 6.2 %; that clock is one sample after
  the loop, 2805 MHz for most records). The per-call probe (`imd_bench --clock-probe`, one warp
  timing `%globaltimer`) reads ~11 % below NVML with ~28 % call-to-call scatter and is only a
  relative indicator. `f_ref` (`power.clock_eff_mhz`) is then the **effective** clock of those
  θ: scanned from the boost clock down by the leave-one-case-out error of the sustained
  windows (the burst calibration's large calls already ran below boost: the limiter acts
  within a timed loop). On the 4090: 2530 MHz (boost 2805). `clock_mhz` in requests and
  results is on this scale.
- **Power at clock f** (an energy roofline): `P(f) = P_idle + [ (e_tc·O_tc + e_alu·O_alu) ·
  (f/f_ref)^γ + e_byte·D ] / t(f)`, with O_tc = K4's int8 ops (its phase `ops`), O_alu = the
  residue and CRT limb operations (the op features of KF, KL, PACK, K5) plus DIRECT's FP64
  flops, D = the compulsory DRAM bytes (Σ phase `bytes`), γ = 2 (dynamic energy per operation
  ∝ V², V ∝ f in the DVFS range; the fitter scans γ ∈ {1, 2, 3} and keeps the best: γ = 1 on
  the 4090, the windows span 2300–2800 MHz only). P(f) rises monotonically with f (numerator
  up, t(f) down). The coefficients are effective: e_byte (335 pJ/B on the 4090) carries the SM
  power of memory-bound work too, e_alu fitted to 0.
- **Regimes.** `BURST`: f = f_ref. `SUSTAINED`: f = f_ref if P(f_ref) ≤ P_cap, else the f in
  [f_min, f_ref] with P(f) = P_cap (bisection, deterministic), f_min if even that is above the
  cap. An explicit `clock_mhz` in the request overrides both (e.g. a locked clock, or the
  clock measured with a record, on the `f_ref` scale). The prediction
  reports the clock used and P(f). The power is the **call's own** average: in a sequence of
  mixed calls the card settles on the mix's average power, so a heavy call between light ones
  runs faster than its SUSTAINED estimate (a sequence-level solve is the caller's, from the
  reported `power_w`; not modeled here).
- **Calibration** (format 2 `power.*`): `f_ref` = the median SM clock of the uncapped
  calibration records (then the effective-clock scan above); `P_cap` = the enforced power limit
  (NVML), `f_min`; `P_idle` fixed at the awake idle (`perf_fit --idle-w`, 60 W on the 4090: with
  most windows pinned at the cap a free intercept took 366 W), e_tc, e_alu, e_byte by
  non-negative least squares on **sustained windows** (15 s pre-heat each: thermal and boost
  history move the settled clock by up to 10 %): `imd_bench --methods
  sustained` runs one plan back-to-back for `--sustain-s` seconds (calls queued ahead, as a
  sequence, after the warm-up) and records the mean time per call, the mean SM clock and the
  mean board power (NVML instant power and clock, sampled every 20 ms; the energy counter is
  never read, so nothing perturbs the window). Each window gives one linear equation in
  (P_idle, e_tc, e_alu, e_byte) at its measured clock. A calibration without a power model
  (format 1, or `power.cap_w = 0`) predicts SUSTAINED = BURST.
- **Default: SUSTAINED** (accepted, review 2026-10-05: ehrenfest runs long sequences; for calls
  below the cap both regimes coincide, so the default only changes the large, tensor-core-heavy
  calls, which in a sequence do run at the capped clock). Two known biases: **pessimistic** for a
  heavy call between light ones (the card's clock follows the sequence's average power, the
  model the call's own), and **optimistic** for long hot runs (16384³ measured at ~1950 MHz after
  minutes at the cap, the model 2385 MHz: ~20 % on the clock; the calibration's 15 s pre-heat
  does not reach that thermal state). A caller that has measured its settled clock passes it as
  `clock_mhz` (NVML scale).
- **`clock_mhz`** (request) is an NVML-scale SM clock for capped / sustained operation; it is
  clamped to the effective clock (`min(clock_mhz, clock_eff)`, the value reported): values at or
  above it mean BURST.
- The power model and the solved clock use the cache-free (COLD) time in every cache state
  (documented approximation: the REPEAT saving is large only on calls that fit the L2, which
  are far below the cap). The bisection stops at 0.05 MHz (≤ 40 steps).

#### [v1.10] Cache state: COLD and WARM

**As calibrated (NOTES-B2 §6):** COLD (an evicted L2) is slower than the repeat state by +4 …
+16 % for calls whose footprint fits the L2 (256³ … 1024²) and ±2 % above; WARM (the previous
call wrote this call's A) is within −7 … +3 % of REPEAT, so **WARM is modeled as REPEAT**. The
parameters θ are **COLD costs**: the fitter adds each repeat record's resident-input saving
`Δ·res_repeat` back to its phases before the fit, and a REPEAT / WARM prediction subtracts it.

The calibration records are repeat calls of one plan on the same operands: at the start of a
call, the L2 holds the end of the previous call's traffic (C, the W planes, the last residues),
so the operands A and B are still resident only when the whole call footprint fits the L2
(small calls). With the effective residency size `L_w` (format 2 `warm.l2_bytes`), the call's
inputs `I` = 8·E of each in-line operand (+ 8bmn of C when β ≠ 0) and its footprint `F = I +
workspace_bytes + 8bmn`:

- `res_repeat = clamp(L_w − (F − I), 0, I)`: input bytes resident in the calibration's repeat
  state (the most recent L_w bytes of traffic; A and B are the oldest);
- **COLD** (`IMD_PERF_CACHE_COLD`, the default): `res = 0`, the inputs come from DRAM;
- **WARM** (`IMD_PERF_CACHE_WARM`): the inputs were just written or read (e.g. C of the
  previous call is A of this one); `res = res_repeat` (as measured; the design's `min(I, L_w)`
  predicted 10–20 % savings on large calls where 1–7 % were measured). **Provisional: one
  method (square GEMM pairs, N = 14 / 16).** Callers whose input is the previous call's output
  should pass WARM (COLD would charge up to +18 % on small calls).
- **REPEAT** (`IMD_PERF_CACHE_REPEAT`): `res = res_repeat`, the state the calibration and the
  hold-out are measured in (the same call just ran); the validation predicts in it (method
  packed's interleaved calls and `--l2 flush` records: COLD). (Design question: COLD is the
  default as asked; the v6 model, fitted on repeat calls, effectively predicted REPEAT.)

Each reading phase (KF; KL A / KL B; K5 for C) gets `−res·Δ` added in proportion to
its share of I (never below 0), with Δ = `warm.l2byte` the calibrated saving per byte served from L2 instead of
DRAM (ns per byte; ≥ 0). The phase's `x₁` stays the compulsory DRAM bytes (no new feature: the
phase layout is frozen); the prediction reports `warm_bytes` = res. Calibration: `imd_bench
--l2 flush` (a 2 × L2 buffer written before each timed call, outside the timed window: COLD),
the repeat calls (`--l2 repeat`, the default) and `--methods pair` (a square GEMM whose second
call reads the first call's C as its A: WARM); Δ and L_w are fitted on the differences
(a scan over L_w, least squares for Δ on COLD − REPEAT; 4090: L_w = 0.8 × 72 MiB, Δ = 1.46 ps
per byte). **If the measured COLD / WARM difference is below a few
% of the call on every measured shape, the calibration keeps Δ = 0 (COLD = WARM, the option
stays as a reported no-op) and NOTES-B2 says so.** A format-1 calibration has Δ = 0.

#### [v1.10] Prediction intervals and ties

Per call and per phase the prediction carries an interval `[time_lo_s, time_hi_s]` (`phase_lo_s`,
`phase_hi_s`; calibration `band_*` = `{lo, hi}` factors), aimed at 80 % coverage (2-fold CV on the
hold-out; BURST / REPEAT residuals, applied unchanged in every regime / cache state): the time times two quantiles of measured / predicted on the hold-out set. The
quantiles are the narrowest of 10 / 90, 7.5 / 92.5, 5 / 95, 2.5 / 97.5, 1 / 99 whose
cross-validated coverage (below) reaches 80 %: the in-sample 10 / 90 percentiles of ~140 calls
cover only ~71 % of new cases (a case's residuals are correlated; measured on the v1.8 hold-out),
5 / 95 covers ~81 %. The ratios are taken per size class of the
**predicted** call time (< 100 µs, 100 µs – 1 ms, ≥ 1 ms) for the call, per kernel class for a
phase (the CALL phase: zero width). `bench/perf_validate.py --bands` computes them from the
hold-out residuals of the fitted calibration and writes them into it (format 2 `band.*`); it
reports the coverage by **2-fold cross-validation over the hold-out cases**, repeated over 20
seeded random halvings (bands from half the cases, coverage on the other half, both ways), so
the coverage it prints is not in-sample.
A calibration without bands (format 1) has zero-width bands. The bands describe the model's
error on shapes like the hold-out's (BURST, the bench's repeat calls); the same ratios are
applied in every regime and cache state (documented approximation).

`imd_perf_rank` (v7): the best candidate is the feasible one with the smallest `time_s`; in
**every** order, every feasible candidate with `time_s ≤ best.time_s × tie_ratio` is a **tie**
(`tie = 1`, the best included; `1 ≤ tie_ratio ≤ 1.25`). The default order is
`IMD_PERF_ORDER_TIME = 0` (review 2026-10-05: a zero-initialized request gets the pure time
order); `IMD_PERF_ORDER_TIES_BY_FOOTPRINT = 1` is opt-in and puts the ties first. **Changed
from band overlap after the data (design question):** overlapping 80 % call bands (±6–17 %)
tied candidates up to 1.36× slower (236 of 360 candidates on a CC tensor) and the footprint
tie-break then picked slow ones. Candidates of one contraction share most of the model error, so
the tie window is the model's error of a **time ratio**: `tie_ratio` (format 2 `rank.tie_ratio`)
= exp of the 80th percentile of |log(meas_i/pred_i) − log(meas_j/pred_j)| over pairs of hold-out
calls of the same case (`perf_validate --bands`; 1.075 on the 4090; 28 of 360 ties). With
TIES_BY_FOOTPRINT the ties come first, ordered by `total_bytes` (the smaller footprint wins), then `time_s`, N, g, vec mode,
packed; then the remaining feasible candidates by time, then the infeasible SUCCESS ones, then
the others (as v6). `raw_rank` = the candidate's position in the v6 time order, so the raw order
is always available; `order = IMD_PERF_ORDER_TIME` (the default; a v6-sized request always
gets it) is the v6 order. The accuracy-aware ranking (a later item) extends the request and the
candidate by appended fields (`struct_size`, `candidate_size`), not by changing these.

### 6c.2 Calibration: data, fit, file, shipped table

- **Data.** `tools/calibrate.sh` runs `bench/sweep_calib.txt` through `bench/run_sweep.sh`
  (quiet gate, GPU lock, warm-up) with the profiling build of `imd_bench`: per case and
  variant the timed median of `imd_contract` and the median per-phase times of
  `--phase-reps` profiled calls (`phases`), packed cases (`imd_pack` phases) and DIRECT. The
  fit set covers every class: KF and KL shapes, every K4 configuration (forced with `--k4`),
  N from 4 to 20, g < N (PACK), RESIDENT, packed operands, K-empty and DIRECT. ≈ 12 min of
  GPU time on the 4090. Its shapes are disjoint from the hold-out set (6c.5).
- **Fit** (`bench/perf_fit.py`, numpy only). The fitter gets every feature vector from
  `imd_perf_predict` through `host.py` (no feature formula in Python). Per class, θ minimizes
  the relative error `Σ ((θ·x − t)/t)²` over that class's measured phases, subject to θ ≥ 0
  (exact NNLS by enumerating the active sets of ≤ 5 parameters); for K4, a scan over the DRAM
  rate (bandwidths 0.1–2 TB/s, or none): the round model is fitted on the phases the DRAM term
  does not explain, and the rate with the smallest relative error of the max wins; around it a
  scan of the effective L2 (0–1 × the recorded L2 size) keeps the fraction with the smallest K4
  error. Features are computed under a seed calibration of the measured device (the records'
  `sm_count` and `l2_bytes`), never the shipped one. Then the call term (with the C library
  doing the sum / max of the fitted phases):
  ordinary least squares of `(t_timed − Σ_p t̂_p) / t_timed` on (1, kernel_launches). Classes
  the sweep cannot exercise get documented stand-ins: BETA takes K5's launch and byte
  parameters. The fitter prints per-class residuals and writes the file.
- **File format 1** (`*.calib`): UTF-8 text, one `key = value` per line, `#` comments and
  blank lines ignored, keys unique, values to the end of the line (trimmed). Required:
  `format = 1`, `gpu`, `sm_count`, the three peaks, `l2_bytes` and **every** parameter
  (`<class>.<feature>`, ns per feature unit, e.g. `kl.byte`, `kl.op_eps`, `k4_64x64.mac2`,
  `k4_64x64.byte`); optional
  metadata `date`, `version` (immerdink commit / host API), `clocks` (clock and power state
  of the calibration run), `notes`. Unknown keys, duplicates, a missing key, a malformed or
  negative-where-forbidden number, or another `format` → INVALID_VALUE with the line number.
  Numbers are parsed and printed locale-independently (`std::from_chars` / `std::to_chars`,
  shortest round-trip form), so format → parse is bit-exact. No dependency, parseable by any
  C program.
- **[v1.10] Data, continued.** The sweep's first line runs `imd_bench --methods occupancy`
  (no timing: the measured CTAs per SM, registers, static shared memory and threads of every
  kernel and variant, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` /
  `cudaFuncGetAttributes`, plus the K4 configurations' measured occupancy); every timed record
  carries `sm_mhz_median`, the SM clock right after each timed call (the one-warp
  `%globaltimer` / `clock64` probe, outside the timed window; ~11 % below NVML, a relative
  indicator) besides NVML's clock after the loop (`gpu_after`). `bench/sweep_sustained.txt`
  (sustained windows with 15 s pre-heat, ≈ 7 min) feeds the power fit, `bench/sweep_cache.txt` (flush / repeat /
  pair, ≈ 2 min) the cache fit; `tools/calibrate.sh` runs all three.
- **File format 2** [v1.10] (written by v7; format 1 is still read). Format 1 plus, all
  required:
  - every v1.10 parameter (`k0.chain`, `kl.wave`, `pack.chain`, `pack.wave`, `k5.chain`, `k5.dram`, `direct.chain`,
    `beta.chain`);
  - `kernel.<name> = resident regs smem threads` for every kernel of `imd_perf_kernel_name`
    (four integers; resident ≥ 1, the others ≥ 0);
  - `power.clock_eff_mhz`, `power.clock_min_mhz`, `power.cap_w`, `power.idle_w`,
    `power.tc_pj`, `power.alu_pj`, `power.byte_pj`, `power.gamma` (≥ 0; `power.cap_w = 0`:
    no power model, `power.clock_eff_mhz = 0`: no clock scaling);
  - `warm.l2_bytes`, `warm.l2byte` (≥ 0); `rank.tie_ratio` (1 … 1.25);
  - `band.call_small`, `band.call_mid`, `band.call_large` and `band.<class>` for every class
    = `lo hi` (two factors, 0 < lo ≤ hi; `1 1`: no band).
  A value of several numbers is separated by blanks. **Reading format 1** migrates: the new
  parameters 0, the built-in occupancy table, no power model, Δ = 0, bands `1 1`; such a
  calibration predicts exactly as host API 6 did (tested). `imd_perf_calib_format` writes the
  calibration's own format (a migrated format-1 struct keeps `format = 1` and is written as
  format 1), so format → parse stays bit-exact for both. **A version-6-sized struct** (a caller
  compiled against host API 6) cannot hold the v7 fields: `imd_perf_calib_default`,
  `imd_perf_calib_parse` and the predictors give it a **true format-1 view** of a format-2
  calibration: format 1, the format-2 parameters zeroed (`k0.chain`, `kl.wave`, `pack.chain`,
  `pack.wave`, `k5.chain`, `k5.dram`, …), built-in occupancy, mod-64 ktail, no power / cache /
  bands. It equals the parse of its own format-1 text (tested), but it is a **degraded,
  unvalidated model** (the format-2 θ were fitted with those terms): such callers should pass
  NULL (the full shipped calibration).
- **Shipped table.** `src/host/calib/rtx4090_sm89.calib` (the 4090 at stock clocks, as
  shipped: no clock lock; its `clocks` line records the state) is embedded at configure time
  (CMake reads the file into a raw string literal: the file is the one source; editing it
  re-runs the configure step). `imd_perf_calib_default` parses that text; a NULL calibration
  pointer anywhere means the default. [v1.10] The text is parsed **once** (a function-local static, immutable after its
  thread-safe initialization: no mutable state) and a NULL pointer uses it in place, with no
  parse or copy per call (it cost ≈ 2 µs per query before). **NULL is the recommended way to use
  the shipped calibration**: `imd_perf_calib_default` copies `struct_size` bytes of it, so a
  caller compiled against version 6 gets its format-1 view (the v7 fields do not fit its
  struct: no occupancy table, power model, cache term or bands), while NULL always means the
  full one.

### 6c.3 API (host API version 6; [v1.10] version 7)

All functions are pure and thread-safe like §6a; a calibration is a **caller-owned plain
struct** (no handle, no allocation, no lifetime rules), so nothing new needs a spec (6c.6).

```c
#define IMD_PERF_CALIB_FORMAT   1
#define IMD_PERF_MAX_PARAMS     6    /* parameters per class (5 used) */
#define IMD_PERF_PARAM_CAPACITY 96   /* param slots: class·6 + j (12 classes used) */
#define IMD_PERF_MAX_PHASES     16
enum { IMD_PERF_CLASS_K0, IMD_PERF_CLASS_KF, IMD_PERF_CLASS_KL, IMD_PERF_CLASS_PACK,
       IMD_PERF_CLASS_K4_128x256, IMD_PERF_CLASS_K4_128x128, IMD_PERF_CLASS_K4_64x128,
       IMD_PERF_CLASS_K4_64x64, IMD_PERF_CLASS_K5, IMD_PERF_CLASS_DIRECT, IMD_PERF_CLASS_BETA,
       IMD_PERF_CLASS_CALL, IMD_PERF_CLASS_COUNT };
enum { IMD_PERF_PHASE_K0, IMD_PERF_PHASE_KF, IMD_PERF_PHASE_KL_A, IMD_PERF_PHASE_KL_B,
       IMD_PERF_PHASE_PACK_A, IMD_PERF_PHASE_PACK_B, IMD_PERF_PHASE_K4, IMD_PERF_PHASE_K5,
       IMD_PERF_PHASE_DIRECT, IMD_PERF_PHASE_BETA, IMD_PERF_PHASE_CALL,
       IMD_PERF_PHASE_PACK_PREP, IMD_PERF_PHASE_PACK_KL, IMD_PERF_PHASE_COUNT };
enum { IMD_PERF_BETA_NONZERO = 1, IMD_PERF_ALPHA_ZERO = 2 };          /* request flags */

typedef struct {
  uint32_t struct_size;  int32_t format, sm_count, reserved;
  double   peak_dram_bytes_per_s, peak_int8_ops_per_s, peak_fp64_flops_per_s, l2_bytes;
  char     gpu[64], date[32], version[64], clocks[128], notes[128];   /* NUL-terminated */
  double   param[IMD_PERF_PARAM_CAPACITY];                            /* ns per unit */
} imd_perf_calib_t;

imd_status_t imd_perf_calib_default(imd_perf_calib_t* out);
imd_status_t imd_perf_calib_parse(const char* text, size_t length, imd_perf_calib_t* out,
                                  int32_t* error_line);            /* error_line may be NULL */
imd_status_t imd_perf_calib_load(const char* path, imd_perf_calib_t* out, int32_t* error_line);
imd_status_t imd_perf_calib_format(const imd_perf_calib_t*, char* buf, size_t capacity,
                                   size_t* length);
const char*  imd_perf_class_name(int32_t cls);          /* "k0", "kf", ..., "call"         */
const char*  imd_perf_param_name(int32_t cls, int32_t j); /* "launch", "byte", ...; NULL: unused */
const char*  imd_perf_phase_name(int32_t phase);        /* the bench tags: "K0 vec", ...   */

typedef struct { uint32_t struct_size; int32_t flags, k4_config /* -1: rule */, reserved;
                 double eps_fraction; /* < 0: the typical-data default; else 0..1 */ }
        imd_perf_request_t;
typedef struct {
  int32_t phase, cls;  int64_t launches;
  double  time_s, bytes, ops, bound_s;  double x[IMD_PERF_MAX_PARAMS];
} imd_perf_phase_t;
typedef struct {
  uint32_t struct_size;
  int32_t  status, dispatch, k4_config, sm_count, num_phases;
  double   time_s, bound_s, flops_eq, tflops_eq;
  double   eps_s, eps_fraction;     /* eps terms at f = 1; the f used */
  int64_t  kernel_launches;
  uint64_t workspace_bytes, plan_device_bytes, pack_bytes, pack_ws_bytes;
  imd_perf_phase_t phase[IMD_PERF_MAX_PHASES];
} imd_perf_t;

imd_status_t imd_perf_predict(const imd_tensor_t* A, const imd_tensor_t* B,
                              const imd_tensor_t* C, const imd_options_t*,
                              const imd_packing_t* /* NULL: none */,
                              const imd_perf_request_t* /* NULL: defaults */,
                              const imd_perf_calib_t* /* NULL: shipped */, imd_perf_t* out);
imd_status_t imd_perf_predict_pack(const imd_pack_desc_t*, const imd_perf_request_t*,
                                   const imd_perf_calib_t*, imd_perf_t* out);
```
- **predict**: returns the plan-time status of `imd_lowering(_packed)` (also in
  `out->status`); the prediction is filled only on SUCCESS (otherwise every field but
  `struct_size` and `status` is 0: a plan that cannot be created has no time). `time_s` =
  Σ phases (the call term is the `call` phase), `tflops_eq = 2bmnk / time_s / 10¹²` (DESIGN's
  TFLOPS-equivalent), `bound_s` = Σ phase bounds, `kernel_launches`, `workspace_bytes`,
  `plan_device_bytes` = the lowering's (exact). `k4_config` / `sm_count`: the configuration
  predicted and the calibration's SMs. With `IMD_PERF_ALPHA_ZERO` the call is K0 + `K5
  beta-only`.
- **predict_pack**: `imd_pack_layout`'s status; on SUCCESS the phases `pack prep`, `pack KL`,
  `call`, `pack_bytes` and `pack_ws_bytes` (exact); 0 time for an empty pack. Of the request
  only `eps_fraction` is used.
- **Calibration functions**: parse / load fill `out` (with `out->struct_size` as the
  caller's size; fields past it are not written) or return INVALID_VALUE (with
  `*error_line`, 0 when not tied to a line: a missing key, an unreadable file);
  `imd_perf_calib_format` writes the canonical text (every key, numbers in the shortest round-trip form) and its
  length (without the NUL), INSUFFICIENT_WORKSPACE if `capacity` ≤ length (the length is
  still reported, as `snprintf`). A calibration whose `format` is not 1 or whose `sm_count`
  < 1 is INVALID_VALUE in every function that takes one.

**Rank:**
```c
enum { IMD_PERF_VEC_GENERATE = 1, IMD_PERF_VEC_RESIDENT = 2 };             /* vec_modes */
enum { IMD_PERF_PACK_NONE = 1, IMD_PERF_PACK_A = 2, IMD_PERF_PACK_B = 4,
       IMD_PERF_PACK_AB = 8 };                                              /* pack_modes */
typedef struct {
  uint32_t struct_size, candidate_size;    /* sizeof as compiled by the caller            */
  int32_t num_moduli_count, residue_group_count;  /* ≥ 1; 0: every g in 1..N               */
  const int32_t* num_moduli;               /* N values (2..20)                             */
  const int32_t* residue_groups;           /* g values ≥ 1 (> N: skipped for that N)       */
  int32_t vec_modes, pack_modes;   /* masks; 0 = GENERATE / NONE only                    */
  int32_t reuse;                   /* contractions per pack, ≥ 1                        */
  int32_t flags;                   /* IMD_PERF_BETA_NONZERO                             */
  uint64_t budget_bytes;           /* 0: no budget                                      */
  double   eps_fraction;           /* as imd_perf_request_t                             */
} imd_perf_rank_request_t;
typedef struct {
  int32_t  num_moduli, residue_group, vec_mode, packed;  /* packed: bit 0 / 1 = caller's A / B */
  int32_t  status, feasible, k4_config, dispatch;
  double   time_s, contract_s, pack_s, tflops_eq, eps_s, eps_fraction;
  uint64_t workspace_bytes, plan_device_bytes, pack_bytes, pack_ws_bytes, total_bytes;
  int64_t  kernel_launches;
} imd_perf_candidate_t;
imd_status_t imd_perf_rank(const imd_tensor_t* A, const imd_tensor_t* B,
                           const imd_tensor_t* C, const imd_options_t* base,
                           const imd_perf_rank_request_t*, const imd_perf_calib_t*,
                           imd_perf_candidate_t* out, int32_t capacity,
                           int32_t* num_candidates);
```
- Candidates: the product of the N list, g (the list, values > N skipped; or 1..N), the
  vec modes and the pack modes; `base` gives rounding, mode and `k4_tune` (N, g, `vec_mode`
  are overwritten). A packed operand uses `imd_pack_desc_for`'s descriptor (the unpacked
  plan's orders: bit-identical results, §6b.2).
- Per candidate: the plan-time status (`imd_lowering(_packed)`: §5 rows 1–5, PACK_MISMATCH,
  the §3.2 RN acceptance), `contract_s` = `imd_perf_predict`, `pack_s` = Σ
  `imd_perf_predict_pack` of its packed operands, **`time_s = contract_s + pack_s / reuse`**
  (the rank key), and the exact footprint from the existing functions: `workspace_bytes`
  (`imd_workspace_size(_packed)`), `plan_device_bytes` (`imd_plan_bytes`), `pack_bytes` and
  `pack_ws_bytes` (Σ `imd_pack_layout`), `total_bytes` = their sum (separate buffers: a
  caller that lends the contract workspace to `imd_pack` needs less). `feasible` = SUCCESS
  and `total_bytes ≤ budget` (or no budget).
- Order: feasible by `time_s`, then infeasible SUCCESS candidates by `time_s`, then the
  others in enumeration order; ties by `total_bytes`, then N, g, `vec_mode`, `packed`. The
  first `min(capacity, n)` are written (`candidate_size` bytes each), `*num_candidates = n`.
- Statuses: INVALID_VALUE for the request (NULL pointers, empty or out-of-range N, bad masks,
  `reuse < 1`, `capacity < 0`, a struct or candidate size below version 1's); otherwise
  SUCCESS, with each candidate's own status.
- The model reports, the caller decides: no accuracy model (N comes from the caller's list),
  no K4 configuration candidates (a plan runs the rule's, or `k4_tune`).

**Python** (`host.py`, thin wrappers): `Calibration` (`.default()`, `.load(path)`,
`.parse(text)`, `.format()`, `.params` by name), `predict(spec, ext, N=, g=, vec_mode=,
packing=, beta=, alpha_zero=, k4_config=, calib=)` → per-phase records with names,
`predict_pack(pack)`, `rank(spec, ext, N=[…], g=None, vec_modes=, packs=, reuse=, budget=,
calib=)` → ranked candidates. `python3 -m immerdink.host … --rank` prints the table.

**Versioning.** `IMD_HOST_API_VERSION` 6 adds the functions, the enums and the structs (each
with `struct_size` or a caller-given element size; fields are only appended). The exception is
`imd_perf_phase_t`, which has no size field and is a fixed `[IMD_PERF_MAX_PHASES]` array inside
`imd_perf_t`: its 96-byte layout and the array length are **frozen** for version 6 (a change is
a new struct and new function names);
`imd_lowering_t` and everything existing are unchanged. The calibration file is versioned
by `format` (a new feature set is a new format and a refit).

**[v1.10] Host API version 7** (appended fields only; every v6 layout above is unchanged and a
v6-sized struct gets v6 behavior except where noted):

```c
#define IMD_PERF_CALIB_FORMAT    2    /* written; format 1 is read too (migrated)          */
#define IMD_PERF_KERNEL_CAPACITY 128
enum { IMD_PERF_KERNEL_K0, IMD_PERF_KERNEL_KF16, IMD_PERF_KERNEL_KF4, IMD_PERF_KERNEL_KL8,
       IMD_PERF_KERNEL_KL4, IMD_PERF_KERNEL_PACK, IMD_PERF_KERNEL_ZERO_ROWS,
       IMD_PERF_KERNEL_DIRECT, IMD_PERF_KERNEL_BETA, IMD_PERF_KERNEL_PACK_PREP,
       IMD_PERF_KERNEL_K4 = 10,             /* + IMD_K4_*                                   */
       IMD_PERF_KERNEL_K5 = 14,             /* + 4 (N - 2) + v, v: 0 E = 1, 1 E = 2, 2 E = 4,
                                               3 lambda-fastest                             */
       IMD_PERF_KERNEL_COUNT = 90 };
typedef struct { int32_t resident, regs, smem_bytes, threads; } imd_perf_kernel_t;
const char* imd_perf_kernel_name(int32_t kernel);   /* "k0", "kf16", ..., "k5_n14_e4", "k5b_n14" */
enum { IMD_PERF_BAND_SMALL, IMD_PERF_BAND_MID, IMD_PERF_BAND_LARGE };  /* < 100 us, < 1 ms, >= */
enum { IMD_PERF_REGIME_DEFAULT = 0, IMD_PERF_REGIME_BURST = 1, IMD_PERF_REGIME_SUSTAINED = 2 };
enum { IMD_PERF_CACHE_COLD = 0, IMD_PERF_CACHE_WARM = 1, IMD_PERF_CACHE_REPEAT = 2 };
enum { IMD_PERF_ORDER_TIME = 0, IMD_PERF_ORDER_TIES_BY_FOOTPRINT = 1 };

/* imd_perf_calib_t, appended (format 2):                                                    */
  imd_perf_kernel_t kernel[IMD_PERF_KERNEL_CAPACITY];       /* measured occupancy           */
  double clock_eff_mhz, clock_min_mhz, power_cap_w, power_idle_w;
  double energy_tc_pj, energy_alu_pj, energy_byte_pj, energy_gamma;
  double warm_l2_bytes, warm_l2byte_ns;
  double band_call[3][2];                                    /* {lo, hi} per size class      */
  double band_class[16][2];                                  /* per IMD_PERF_CLASS_*         */
  double tie_ratio;                                          /* rank tie window, 1 ... 1.25  */
/* imd_perf_request_t, appended (v6: 24 bytes, v7: 40):                                      */
  int32_t regime, cache;          /* IMD_PERF_REGIME_*, IMD_PERF_CACHE_*                    */
  double  clock_mhz;              /* > 0: NVML-scale clock, clamped to clock_eff (overrides
                                     the regime; >= clock_eff: BURST)                       */
/* imd_perf_t, appended:                                                                      */
  int32_t regime, cache;          /* the ones used (DEFAULT resolved)                        */
  double  clock_mhz, power_w;     /* the SM clock assumed; the call's average power there (0:
                                     no power model)                                         */
  double  warm_bytes;             /* input bytes served from L2 (WARM)                       */
  double  time_lo_s, time_hi_s;
  double  phase_lo_s[IMD_PERF_MAX_PHASES], phase_hi_s[IMD_PERF_MAX_PHASES];
/* imd_perf_rank_request_t, appended:                                                         */
  int32_t regime, cache, order, reserved;   double clock_mhz;
/* imd_perf_candidate_t, appended:                                                            */
  double  time_lo_s, time_hi_s, clock_mhz, power_w;
  int32_t tie, raw_rank;
```
- A request shorter than v7 (24 / 64 bytes) means regime DEFAULT, COLD, no clock override and,
  for rank, the time order. DEFAULT = SUSTAINED when the calibration has a power model,
  else BURST. **So a v6-sized request with the shipped (format-2) calibration gets COLD +
  SUSTAINED: its numbers change** (small calls slower by the cold-input cost, capped calls by the
  settled clock); this is the one exception to "v6-sized means v6 behavior". Invalid: an unknown regime, cache or order, `clock_mhz` < 0 or not finite.
- A calibration struct shorter than v7 (passed by a v6 caller) is read as a true format 1
  (the format-2 parameters zeroed, built-in occupancy, no power model, Δ = 0, no bands; §6c.2),
  whatever its `format` field says. `calib_valid` for format 2: resident ≥ 1, every power / warm value ≥ 0
  and finite, 0 < lo ≤ hi in every band, 1 ≤ tie_ratio ≤ 1.25.
- `imd_perf_predict_pack` uses the request's regime, cache (its input is X: WARM serves X from
  L2 up to L_w) and clock as `imd_perf_predict`.
- Python (`host.py`): `predict(..., regime=, cache=, clock_mhz=)`, `Prediction.regime`,
  `.cache`, `.clock_mhz`, `.power_w`, `.warm_bytes`, `.time_lo_s`, `.time_hi_s`, phases'
  `lo_s` / `hi_s`; `rank(..., order=, regime=, cache=, clock_mhz=)`, `Candidate.tie`,
  `.raw_rank`, `.time_lo_s`, `.time_hi_s`, `.clock_mhz`, `.power_w`; `Calibration.kernels`,
  `.power`, `.warm`, `.bands`; `kernel_names()`.
- `IMD_HOST_API_VERSION` 7. No change to plans, launches, results or footprints.

### 6c.4 Tests

`host_c_api` (C): default calibration, format → parse round trip (bit-exact), every parse
error with its line, predict / rank smoke and footprint equality. `host_python`: the same
from Python plus an **independent Python transcription of the features** on random
descriptors (as for `k4_select`), K4 features × A3 constants = `k4_estimate`, monotonicity
against the shipped (fitted) calibration and a synthetic one with a large eps-path cost
(doubling m, n or k never lowers the predicted time at fixed configuration and fixed eps
fraction; with the default fraction never below the time without the eps terms), the eps
bracket and the default ramp (exact log₂T), rank: every
footprint equals `imd_workspace_size(_packed)` / `imd_plan_bytes` / `imd_pack_layout` exactly,
the budget is respected, the order is as specified; `perf_calib_shipped`: the shipped file
parses and its sm_count is 128. `host_cuda_free` unchanged (the model is plain C++).

[v1.10] `host_perf` adds: the Python transcription of every chain feature (it re-implements the
shared launch rules independently: KL, K2 / K3, K5 variants, grid caps) against the C features
on the random descriptors; a format-1 calibration (the v6 shipped file, kept as a test vector)
predicts bit-identically to the v6 formulas (chain parameters 0, no clock scaling) and round-
trips as format 1; format-2 parse errors (a missing `kernel.` line, a band with p10 > p90,
resident 0); BURST at `clock_mhz = f_ref` equals BURST; SUSTAINED ≤ BURST in speed (never
faster) and = BURST when P(f_ref) ≤ cap; P(f_s) = cap within 1e-9 relative when capped; a
larger clock never predicts a longer time; COLD ≥ WARM; bands contain the time and scale with
it; rank: ties = the feasible candidates within tie_ratio of the best's time, ordered by total_bytes,
`raw_rank` is a permutation reproducing the v6 order, `order = TIME` and a v6-sized request give
the v6 order. `host_c_api`: the v7 sizes, v6-sized structs still accepted. The kernel-side launch
refactor (rules moved to `analysis.hpp`) is covered by the existing bit-exact GPU tests.

### 6c.5 Validation (hold-out)

`bench/sweep_perf_validate.txt` (shapes disjoint from the fit set: the regression sweep's
cases, tensor cases, odd sizes, many small shapes; N = 8 / 14 plus 10 / 16, g < N,
RESIDENT, packed) run like the calibration; `bench/perf_validate.py` compares the predicted
total with the measured timed median (and per phase with the phase medians) and prints the
error distribution per size class. **Targets** (relative error of the total):

| measured call | median \|err\| | 90th percentile |
|---|---|---|
| ≥ 1 ms | ≤ 5 % | ≤ 15 % |
| 100 µs – 1 ms | ≤ 10 % | ≤ 20 % |
| < 100 µs | ≤ 15 % | ≤ 30 % |

**[v1.10] Further targets.** (1) The table above holds in the BURST regime and the records' cache state
(**REPEAT**; method packed and `--l2 flush`: COLD). The default COLD on small calls rests on the
6-shape cache sweep only (`bench/sweep_cache.txt`) and stays met after the chain features (a
clock-conditioned evaluation, `perf_validate --record-clock`, was worse and is not used); per-class phase errors are reported before / after (K3, K4 at
k̂ mod 64 ≠ 0 named). (2) **SUSTAINED**: the predicted time of each sustained window vs its
measured mean time per call, with the power model fitted leave-one-case-out (the window's own
case excluded): the table's size-class targets, and the predicted settled clock within 5 %
(median) of the measured (the effective-clock scan used the same 14 windows, so the reported
leave-one-out median is mildly optimistic). (3) **Intervals**: 80 % of hold-out calls inside
their [lo, hi]
band, between 70 % and 90 % per size class, by the 2-fold cross-validation above. (4) **Cache**:
the measured COLD / repeat / WARM differences and the model's, per case (no target: an effect
size; Δ = 0 if it is small).

Tiny calls drift by 10–20 % between performance states (BENCHMARKING "Tiny shapes"), so
their target is loose. If a target is missed: per-class residuals first; then the smallest
physically motivated feature (e.g. tile-quantized bytes for grids below one wave, the K5
λ-fastest kernel), never per-shape factors; a feature change is a new file format.

### 6c.6 Spec

**No TLA+ change.** The model is a pure function of (descriptors, options, request,
calibration) over caller-owned plain data: no handle, no shared or global state, no
lifetime, no GPU, no effect on plans (the calibration is never read by plan creation).
`imd_perf_rank`'s footprints are the existing exact functions (§4a, §6a, §6b.4, already
checked: `Pipeline LayoutExact / PlanBytesExact`, `cert_workspace`, `cert_pack_layout`) and
its feasibility test only compares them with a number; it creates no plan and changes no
legality. HostAPI's rules (pure, thread-safe host queries) cover it as they cover §6a.
[v1.10] Unchanged: the clock, power, cache and band terms are more pure arithmetic over the
calibration; the launch rules that moved to `analysis.hpp` produce the same launches (a
refactor of launch-shape arithmetic, covered by the bit-exact GPU tests), so no Pipeline action
changes.

---

## 6e. Hot-path planner queries from Python [v1.11]

Eric (2026-10-05): **"Planner calls will be very many one at a time calls on a hot path"**
(ehrenfest, a Python code). Contraction calls stay as §6d (ctypes, about 1 µs over C). This
section is about the host queries (§6a, §6b, §6c): `workspace_size`, `plan_bytes`,
`lowering`, `pack_desc_for`, `predict`, `predict_pack`, `rank`. **No C change:** the host API
stays version 6 and `libimmerdink_host` stays the single source of truth; the new code is the
Python module `immerdink.planner` and a C timing tool. (v1.10 is the parallel
performance-model revision; this section does not depend on it.)

### 6e.1 Where the time went (before)

`bench/planner_overhead.py`: six ehrenfest-like contractions (GEMM `ik,kj->ij` 16³ and 512³;
CC ring `kjbc,kiac->abij` v = 64, o = 32; CC ladder `abef,efij->abij` v = 48, o = 16; DF
`Qia,Qjb->iajb` Q = 256, o = 16, v = 64; DF transform `Qma,mi->Qia` Q = 256, m = 80) × N ∈
{8, 14, 17}; µs per call, median of 11 interleaved rounds of 60 calls, one P-core
(`taskset`), quiet machine, i9-13900K, Python 3.14.7, 2026-10-05. The **C floor** is the C
function alone in a C loop (`imd_planner_floor`, `bench/planner_floor.c`: median of 7 batches
of ~4 ms).

| Entry point | C floor | `host.py` |
|---|---|---|
| `workspace_size` | 0.25–0.42 | 7.8–9.5 |
| `lowering` | 0.52–0.71 | 11.7–13.4 |
| `plan_bytes` | 0.37–0.58 | 8.1–9.7 |
| `pack_desc_for` | 0.28–0.50 | 12.2–14.5 |
| `predict` (calibration passed; NULL: 2.68–2.91) | 0.64–0.86 | 16.3–19.8 (calib passed: 13.9–17.4) |
| `predict_pack` | 0.36–0.46 | 8.7–9.4 |
| `rank` (one N, every g) | 5.1–14.5 | 32–63 |
| `immerdink.workspace_size` (Layouts) | 0.25–0.42 | 7.5–8.8 |

`host.predict` = parse the spec into `Desc`s 2.2–2.7 + build the ctypes structs (three
tensors of three arrays each, options, request, the 1648-byte output) 4.3–5.3 + the ctypes
call with prebuilt arguments 1.4–1.6 (of which C 0.6–0.9) + decode into dataclasses 4.7–6.6.
With `calib=None` the C library also parses the embedded calibration text on every call
(`resolve_calib(NULL)`: +2.1 µs; [v1.10 merge] no longer: the shipped calibration is parsed once
in C, §6c.2). `host.rank` calls C twice (count, then fill) and builds a
dataclass per candidate. Plan creation's host part (descriptors, structs, `imd_workspace_size`)
is 7.0–8.4 µs, small next to `imd_plan_create` and paid once per `PlanCache` entry.

**Target.** The C floor plus one ctypes foreign call (0.6–0.9 µs: the call row minus the
floor): about 1–2 µs for a single query (≤ 2 µs for `predict`, inside the 2–3 µs asked for),
and the C floor plus about 1–2 µs for `rank`, whose C cost is one prediction per candidate.

### 6e.2 Decision

In order of least new machinery:

- **(a) Cheaper marshalling in pure ctypes — chosen.** A `Query` builds the three descriptor
  structs once per contraction; option structs, packings, requests and the default
  calibration are built once and reused; a call allocates only its output struct; results are
  **lazy views** reading the C struct on attribute access; `rank` is one C call. Reaches the
  target (§6e.6).
- **(c) Memoization — chosen, explicit and opt-in.** `Memo(maxsize)`: bounded, keyed by the
  full query, for callers that repeat identical queries (a hit: 0.6–1.0 µs).
- (b) A handle-based C API (`imd_perf_query_create` caching the lowering) or a batch entry
  point — **not needed.** A single query's C floor is 0.25–0.86 µs in total, so caching the
  lowering would save well under 0.5 µs per call, while adding a C object with a lifetime
  (a HostAPI spec change; §6c.6's "no handle, no state" property lost) and a host API version.
  ehrenfest's pattern is one query at a time, so a batch entry point does not apply.
  Reconsider only if a profile of ehrenfest shows `rank` dominating (its C cost scales with
  the number of candidates).
- (d) A nanobind module — **not needed.** After (a) the Python overhead over the C floor is
  0.6–1.0 µs per single query (about one ctypes call plus one small object); a compiled
  module could save at most that, and a memo hit is already below 1 µs. Not worth a compiled
  extension and build dependency (§6d's decision stands).

### 6e.3 API (`python/immerdink/planner.py`)

```python
from immerdink import planner

q = planner.Query(spec, extents, strides=None, calib=None, memo=None, lib=None)
q = planner.Query((A, B, C))                       # raw descriptors, as host.py
q = planner.Query.from_layouts(spec, A, B, C)      # CUDA arrays or immerdink.Layout
q.workspace_size(N, g=0, vec_mode="GENERATE", *, rounding="RN", mode="EMULATE", packing=None)
q.plan_bytes(N, g=0, vec_mode="GENERATE", *, rounding, mode)            # (device, host)
q.lowering(N, g=0, vec_mode="GENERATE", *, rounding, mode, sm_count=0, packing=None)
q.pack_desc_for(operand, N, g=0, vec_mode="GENERATE", *, rounding, mode)   # host.PackSpec
q.predict(N, g=0, vec_mode="GENERATE", *, rounding, mode, packing=None, request=None,
          beta=False, alpha_zero=False, k4_config=None, eps_fraction=None, calib=None)
q.predict_pack(pack, *, eps_fraction=None, request=None, calib=None)
q.rank(N, g=None, vec_modes=("GENERATE",), packs=("NONE",), reuse=1, budget=0, beta=False,
       *, rounding, mode, eps_fraction=None, calib=None, **fields)       # -> Ranking

req = planner.Request(beta=True, eps_fraction=0.5)  # prebuilt imd_perf_request_t (+ **fields)
r = q.rank([12, 14], budget=2**31); r[0], len(r), r[1:4], r.to_list()  # lazy Candidates
p = q.predict(14); p.time_s, p.phase("K4 gemm"), p.to_prediction()   # lazy Prediction

pl = planner.Planner(calib=None, memo=planner.Memo(4096, max_bytes=64 * 2**20),
                     max_queries=1024, lib=None)
pl.query(spec, extents, strides=None); pl.query_arrays(spec, A, B, C)   # cached Query
pl.predict(spec, extents, N=14, ...)   # host.py's signatures: workspace_size, plan_bytes,
                                       # lowering, pack_desc_for, predict, rank
planner.default_calibration(lib=None)  # the shipped calibration, loaded once per library
```

`Prediction`, `Lowering` and `Candidate` have the attribute names and values of
`host.Prediction` / `host.Lowering` / `host.Candidate`; `.to_prediction()`, `.to_lowering()`,
`.to_candidate()` (and `Ranking.to_list()`) return exactly host.py's objects.

### 6e.4 Semantics

- **Bit-identical to host.py.** Each method calls the C function host.py calls with the same
  struct contents (descriptors, options, request, packing, calibration); views decode with
  host.py's code (`to_prediction` is `host._prediction`). Nothing is computed in Python.
- **`calib=None`** passes NULL ([v1.10 merge]: the C library parses its shipped calibration
  once, §6c.2, so NULL costs nothing extra; before that merge planner.py passed a cached
  `imd_perf_calib_default`). `planner.default_calibration()` still returns it, loaded once.
- **Requests**: the default is the struct host.py builds (flags 0, `k4_config` −1,
  `eps_fraction` −1). `Request(**fields)` and `rank(**fields)` set any later field of
  `imd_perf_request_t` / `imd_perf_rank_request_t` by name, and a view's attribute that is a
  field of the C struct is read by name, so later request and result fields flow through
  without changing this module (a `Prediction` attribute of a later host.py dataclass that
  is not a C field falls back to host.py's decoder; a `Lowering` / `Candidate` one raises
  AttributeError naming it, so a needed converter is never silently missing).
- **`rank`** passes the candidate count as the capacity (Σ over the N list of the g values
  ≤ N, or N, × vec modes × pack modes), bounded by 4096 (so an absurd N, which C rejects,
  allocates nothing large): one C call, a second, exact one only when the library reports
  more. Feasibility, order and ties are C's alone (no ranking logic in Python): a caller
  wanting the best configuration under a budget calls `rank(budget=b)[0]` (memoizable; the
  budget is part of the key).
- **Views** are immutable (`__slots__`, read-only properties) and own their output struct.
- **Errors**: the same statuses raise `ImmerdinkError` (plan-time statuses are returned in
  `.status`, as host.py); an unknown option name is ValueError (host.py: KeyError).
- The fast calls use the library's own function objects without `argtypes` (host.py's typed
  declarations are untouched) and pass only pointers, None and int32 values.

### 6e.5 Caches and threads

- **Query**: the descriptor structs are immutable after construction. Its other caches hold
  argument structs only, never results: option structs (≤ 512, cleared when full), packing
  descriptors (≤ 64), rank N / g arrays (≤ 64). Dict operations are atomic, output structs are
  per call: any number of threads may share a Query, with no lock on the call path.
- **Memo** (opt-in, never implicit): key = (the Query object, the method, every argument);
  calibrations by identity (treated as immutable). **Two bounds, both always on**: `maxsize`
  entries (default 4096) and `max_bytes` of stored C result structs (default 64 MiB).
  `nbytes` is exact: the `ctypes.sizeof` of each stored output struct, counted at insert.
  Worst cases per entry: a `Prediction` 1648 B (~1.7 KB); a `Ranking` 128 B per candidate,
  up to ~214 KB for every N in 2..20 × every g × both vec modes × four pack modes (1672
  candidates; a list repeating N values can be larger, up to the bound); `workspace_size`,
  `plan_bytes`, `pack_desc_for` results hold no C struct (0 counted). So 4096 rankings alone
  could reach ~0.9 GB without the byte bound; with it the memo stays ≤ 64 MiB of structs.
  Oldest entries are evicted until both bounds hold; a single result larger than `max_bytes`
  is returned but not stored (a bypass, not an error). An entry keeps its Query alive (the
  key holds it), also after the Planner evicted that Query. Lock-free lookups, inserts under a
  lock; an unhashable argument (e.g. a list of N) bypasses it. A hit returns the stored,
  immutable view. Valid because every query is a pure function of its arguments (§6a, §6c.3).
- **Planner**: Queries keyed by (spec, the spec's extents in label order, strides), by the
  normalized descriptors, or by (spec, Layouts) for `query_arrays`; bounded FIFO
  (`max_queries` ≥ 1, default 1024); an evicted Query stays valid for whoever holds it. A
  Query with its cache key takes 4.6–5.7 KB (measured with tracemalloc, fresh and after a few
  queries; three operand descriptors, its argument-struct caches), so 1024 queries are ≤ 6 MB
  and a count bound suffices. The spec → labels table is bounded by `max_queries` as well
  and emptied by `clear()`.
- Process-wide: one entry (function objects) per loaded `HostLibrary`,
  as host.py's library cache.

### 6e.6 Measured (after), tests, spec

Same method as §6e.1 (µs per call, min–max over the 18 points):

| Entry point | C floor | `host.py` | `planner.Query` | `Planner` (lookup) | Memo hit (`Planner`) |
|---|---|---|---|---|---|
| `workspace_size` | 0.25–0.42 | 7.8–9.5 | **0.84–1.00** | 1.17–1.41 | 0.57–0.60 |
| `lowering` | 0.52–0.71 | 11.7–13.4 | **1.27–1.45** | | |
| `plan_bytes` | 0.37–0.58 | 8.1–9.7 | **1.09–1.28** | | |
| `pack_desc_for` | 0.28–0.50 | 12.2–14.5 | **2.42–2.71** | | |
| `predict` | 0.64–0.86 | 16.3–19.8 | **1.60–1.82** | 2.06–2.30 | 0.74–0.79 |
| `predict_pack` | 0.36–0.46 | 8.7–9.4 | **1.27–1.37** | | |
| `rank` (one N, every g) | 5.1–14.5 | 32–63 | **7.2–16.7** | 7.7–17.2 | 0.95–1.00 |
| workspace of arrays / Layouts | 0.25–0.42 | 7.5–8.8 (`immerdink.workspace_size`) | | 1.55–1.76 (`query_arrays`) | |

Reading `.time_s` of a prediction adds < 0.02 µs; a memo hit through a held Query is 0.38 µs.
`pack_desc_for` builds a `host.PackSpec` dataclass per call (memoizable).

- **Tests** `host_planner` (CPU, ctest; `tests/test_planner.py`): 3000 random single queries
  (legal and illegal descriptors, every vec mode, mode, rounding and g, packed operands incl.
  PACK_MISMATCH, random requests, two synthetic calibrations and the default): every result
  bit-identical to host.py (floats compared by bits), every lazy attribute equal to the
  dataclass's, the same error statuses; 600 random `rank` requests (N and g lists, vec / pack
  masks, reuse, budgets, invalid values): identical lists, exactly one C call each (two
  beyond the 4096 capacity bound: 4320 candidates), `rank(N=10**7)` rejected without a large
  allocation, numpy integers as N and g; Planner keying (spec, extents, strides, raw
  descriptors, Layouts), its query and label bounds, `max_queries` ≥ 1; Memo hits, the entry
  and byte bounds (rank entries within `max_bytes`, exact `nbytes`, an oversized entry not
  stored), bypass; read-only `Query.calib`; C array fields as tuples; 8 threads sharing a Planner with a small Memo
  (evicting) and one without: results identical to the serial host.py ones, the bound held;
  argument errors.
- **No TLA+ change.** No C code changes; the Python caches memoize pure functions of
  immutable keys and hand out immutable values (the §6d.8 argument for `PlanCache`), and a
  Query holds no C object with a lifetime.

### 6e.7 Design questions (reviewed 2026-10-05)

Ruling: 1, 2, 5, 6, 8, 9 accepted; 3 kept until C parses the shipped calibration once (then
`calib=None` passes NULL); 4 with a byte bound on the memo; 7 accepted without
`Ranking.best` (removed: it re-derived C's feasibility and ordering in Python).

1. **A new module** `immerdink.planner` beside an unchanged `host.py` (rather than routing
   host.py's functions through the fast path): additive, no change for existing callers, no
   overlap with the v1.10 performance-model work in host.py.
2. **Lazy views** instead of dataclasses for fast results (`.to_*()` gives the dataclasses;
   eager decoding costs 4.7–6.6 µs per prediction).
3. **`calib=None` → the cached default passed explicitly.** Relies on NULL ≡
   `imd_perf_calib_default` (true; tested on every random query). Suggested to the
   performance-model owner: parse the shipped calibration once in C (a C++11 function-local
   static: thread-safe initialization, no mutable global state), which takes 2.1 µs off every
   `host.predict(calib=None)` and every C caller passing NULL. Not done here (perf.cpp is the
   v1.10 work's). **Done in v1.10** (§6c.2); planner.py now passes NULL.
4. **FIFO eviction** for Memo and Planner (not LRU): lock-free hits; the expected working set
   is small and stable. Default `Memo` size 4096 entries and 64 MiB (both bounds always on),
   `max_queries` 1024.
5. **Memo keys use the Query object's identity** (hashing descriptor dataclasses would cost
   ~1 µs per hit): two Query objects of one contraction do not share entries; a Planner gives
   one Query per contraction.
6. **Unknown option names raise ValueError** in the new API (host.py raises KeyError).
7. **`rank` returns a `Ranking`** (an immutable sequence of views) rather than a list.
8. **Untyped ctypes function objects** on the fast path (no argtypes conversion); host.py's
   typed declarations are unchanged.
9. **No change to the device API** (`Plan`, `PlanCache`, `immerdink.workspace_size`): plan
   creation is paid once per cache entry; workspace queries on a hot path use
   `Planner.query_arrays` (1.6–1.8 µs vs 7.5–8.8 µs).

---

## 6f. Accuracy-aware planning (ROADMAP B2) [v1.12]

Eric (2026-10-05): `rank` should answer **"the cheapest configuration that meets this error
target"**, combining the §3.8 bounds with operand statistics the caller supplies. Until now the
caller picked N itself, from `params.py --table` or docs/ACCURACY.md §4. The consumer is
ehrenfest, which makes many single planner calls on a hot path (§6e). Host API **version 8**.
No semantic, layout, launch or plan change.

### 6f.1 What the caller supplies

immerdink never reads the data here (kernel scope). The caller passes an `imd_err_request_t`:

| field | meaning | needed for |
|---|---|---|
| `metric` | one of:<br>`IMD_ERR_NORMWISE`, `\|D−x\| / (‖a_i‖₂‖b_j‖₂)` (the §3.8 metric);<br>`IMD_ERR_ELEMENTWISE`, `\|D−x\| / \|x\|`;<br>`IMD_ERR_ABSOLUTE`, `\|D−x\|` | always (default NORMWISE) |
| `target` | the error the configuration must meet; 0 = report only | `rank`, `imd_err_min_moduli` |
| `cond` | ρ ≥ 1, an upper bound on `max ‖a_i‖₂‖b_j‖₂ / \|x_ij\|` over the outputs that must meet the target | ELEMENTWISE |
| `norm_product` | an upper bound on `max ‖a_i‖₂‖b_j‖₂` (e.g. `max_i ‖a_i‖ · max_j ‖b_j‖`) | ABSOLUTE |
| `l1_ratio_a`, `l1_ratio_b` | optional: upper bounds on `max_i ‖a_i‖₁ / ‖a_i‖₂` (and on the same ratio for the columns of B), in [1, √k]; 0 = unknown = √k | a tighter bound for decaying or sparse rows |
| `basis` | which estimate must meet the target:<br>`IMD_ERR_BASIS_BOUND` (the default: the rigorous bound);<br>`IMD_ERR_BASIS_TYPICAL` (opt-in: the typical estimate) | `rank`, `imd_err_min_moduli` |
| `fail_prob` | p, the typical estimate's failure probability per call; 0 = 2⁻⁴⁰ (for a probability P per run of C calls, pass P/C) | TYPICAL |

**From the lowering:**
- k, the GEMM view's contracted length;
- M = b·m·n, the number of outputs;
- the dispatch;
- N and the rounding, from the options.

Rows and columns are those of the GEMM view (§1.1). For a tensor, a_i is A's fiber over the
contracted labels at one value of its free and batch labels, and b_j is B's. A role swap (§5)
changes nothing, because every formula is symmetric in A and B.

**No request (NULL)** reports the NORMWISE values with no target. NORMWISE needs **no
statistics at all**: the §3.8 bound relative to ‖a_i‖‖b_j‖ depends only on (N, k, rounding).

How the caller gets the statistics is up to the caller:
- row and column 2-norms and 1-norms cost one pass over the operands;
- ρ needs |x|, so it comes from a previous iteration, from physics, or from a cheap low-N
  pass (docs/ACCURACY.md §4).

### 6f.2 The two estimates

Notation:
- δ = ½ (RN) or 1 (TRUNC);
- d = T − κ/2 (RN) or T (TRUNC);
- s = √(1 + k·2⁻⁶⁰);
- u = 2⁻⁵³;
- ν_a and ν_b are `l1_ratio_a` and `l1_ratio_b`, clamped to √k (0 means √k).

**Rigorous bound (`err_bound`), EMULATE.** PROOFS L3c: L3a (the ℓ1 form of §3.8) with
‖a_i‖₁ ≤ ν_a‖a_i‖₂ for every row (so the scalar maximum the caller passes is valid) and L1d's
2⁻σ < 2‖a‖s/d:

```
E   = 2δ·(ν_a + ν_b)·s/d  +  4δ²·k·s²/d²            (relative to ‖a_i‖‖b_j‖)
```

- With ν = √k this is exactly §3.8's term1 + term2 (`params.error_guidance`).
- The final rounding satisfies |D − y| ≤ u|y|, where y = z·2^{−σ−τ}, |y − x| ≤ E‖a‖‖b‖ and
  |x| ≤ ‖a‖‖b‖. Hence:

```
NORMWISE      E·(1+u) + u
ELEMENTWISE   E·ρ·(1+u) + u
ABSOLUTE      (E·(1+u) + u) · norm_product  +  2⁻¹⁰⁷⁵
```

**[review] Underflow.** An ABSOLUTE bound can be subnormal (a tiny `norm_product`). In that
case the relative inflation of the C value (below) does nothing, so ABSOLUTE carries an
absolute term:
- in the specification, 2⁻¹⁰⁷⁵: RN to a subnormal satisfies |fl(y) − y| ≤ u|y| + 2⁻¹⁰⁷⁵, so
  this ABSOLUTE bound also holds when D underflows gradually;
- in C, 2⁻¹⁰⁷¹, which also covers the subnormal roundings of the computation.

**Guaranteed under the §3.8 hypotheses:**
- finite inputs;
- no overflow;
- NORMWISE and ELEMENTWISE: no underflow of D;
- the caller's statistics are true bounds.

**[review] The bound is on the product D** (α = 1, β = 0). imd_contract writes
C ← βC + RN(αD) (§3.7):
- α ≠ 1 adds one rounding: a bound n becomes n·(1+u) + u in every metric. For ABSOLUTE, pass
  |α|·norm_product.
- β ≠ 0 adds the fma's u·|βC + αD|, which no statistic here expresses.

It does not depend on the ε path (§6c), which is exact. It needs RN acceptance (κ < T);
otherwise there is no bound (+∞), and plan creation rejects that configuration anyway.

**DIRECT dispatch** (DIRECT mode, K-empty) is the native FMA chain (§3.7).
- Its bound is `γ_k = k·u / (1 − k·u)` relative to Σ|a_h b_h| ≤ ‖a‖‖b‖. The final rounding is
  already included.
- NORMWISE γ_k, ELEMENTWISE γ_k·ρ, ABSOLUTE γ_k·norm_product + 2k·2⁻¹⁰⁷⁵ (each fma may
  underflow; in C + (k + 16)·2⁻¹⁰⁷⁴). Proof: PROOFS L3c, for the chain of `contract.cu`'s
  native path.
- +∞ if k·u ≥ ½ (γ_k ≥ 1 there).

NOOP and SCALE (no product, k = 0) give 0.

**Typical estimate (`err_typical`), EMULATE with RN only.** This is a probabilistic bound
under a stated model, not a guarantee.

The model is the standard one of probabilistic rounding-error analysis: the quantization errors
are independent and zero-mean, with |e_h| ≤ ½.
- Per output, the first-order error Σ_h (a_h f_h 2^{−τ} + b_h e_h 2^{−σ}) is a sum of
  independent zero-mean terms. Each lies in an interval of width
  w_h = |a_h|2^{−τ} + |b_h|2^{−σ}, and Σ w_h² < 16(‖a‖‖b‖s/d)².
- Hoeffding's inequality plus a union bound over the M outputs then give, with probability
  ≥ 1 − p:

```
E_typ = min(E,  s/d·√(8·ln(2M/p))  +  k·s²/d²)        (relative to ‖a_i‖‖b_j‖)
```

It uses the same metric composition as E.

For TRUNC and DIRECT, `err_typical` = `err_bound`. TRUNC's errors are not zero-mean: their sign
follows the data (§3.3's same-sign bias).

**Where the model fails:** rows with low-entropy mantissas, e.g. many equal entries that are not
representable at the row's scale, such as uniform superpositions (entries 1/√n) or repeated
hopping constants. Their quantization errors are equal and add coherently, up to the rigorous
bound. With ν = √k that is E/E_typ ≈ √(k / (2 ln(2M/p))) above E_typ. This is measured in
NOTES-B2 §7.

**No fitted constant.** The estimate was checked against the accuracy data instead:
- it covers all 520 M5 runs (max observed / estimate 0.21 at p = 2⁻⁴⁰ where the emulation error dominates,
  N ≤ 13);
- held-out coverage (φ = 0.5…4, scaled inputs, k = 64…10⁵, structured rows) is in NOTES-B2 §7.

**In moduli:** at k = 10⁴ the rigorous bound is 2√k/T = 200/T, while E_typ ≈ 18.4/T at
M = 10⁶ and p = 2⁻⁴⁰ (15.1/T at 2⁻²⁰).
- That is about 0.9 modulus cheaper.
- Observed errors are another ~5× below E_typ, about half a modulus.

**[review] p** is the failure probability per call. For a probability P over a run of C calls,
pass p = P/C (the cost grows as √ln).

### 6f.3 API (host API version 8)

```c
enum { IMD_ERR_NORMWISE = 0, IMD_ERR_ELEMENTWISE = 1, IMD_ERR_ABSOLUTE = 2 };
enum { IMD_ERR_BASIS_BOUND = 0, IMD_ERR_BASIS_TYPICAL = 1 };
enum { IMD_ERR_UNCHECKED = 0, IMD_ERR_MET = 1, IMD_ERR_MISSED = 2 };   /* err_status        */
typedef struct {                 /* 64 bytes                                                  */
  uint32_t struct_size;  int32_t metric;
  double target, cond, norm_product, l1_ratio_a, l1_ratio_b;
  int32_t basis, reserved;  double fail_prob;
} imd_err_request_t;
typedef struct {                 /* 72 bytes                                                  */
  uint32_t struct_size;  int32_t status;
  int32_t num_moduli, rounding, dispatch, err_status;
  int64_t k, outputs;
  double bound, typical;         /* in the request's metric                                  */
  double unit_bound, unit_typical;  /* E and E_typ (EMULATE), γ_k (DIRECT): the per-(i, j)
                                       coefficient of ‖a_i‖‖b_j‖ before the metric           */
} imd_err_t;

/* The estimates of one configuration, from scalars. num_moduli 2..20, rounding IMD_RN/TRUNC,
   dispatch IMD_DISPATCH_*, k >= 0, outputs >= 0; req NULL = NORMWISE, no target. */
imd_status_t imd_err_estimate(int32_t num_moduli, int32_t rounding, int32_t dispatch, int64_t k,
                              int64_t outputs, const imd_err_request_t* req, imd_err_t* out);
/* The smallest N in [n_lo, n_hi] (0, 0 = 2..20) whose estimate (req->basis) meets
   req->target > 0, for this contraction and options (rounding, mode; N, g, vec_mode ignored):
   one lowering (not the workspace of any N), then the estimate per N. SUCCESS: out at that
   N. NOT_SUPPORTED with err_status MISSED: no N in range meets the target (out at n_hi,
   num_moduli 0). The lowering's own status (INVALID_VALUE, or NOT_SUPPORTED for e.g. a
   repeated or broadcast C label, with out zeroed: err_status UNCHECKED) is returned as such. */
imd_status_t imd_err_min_moduli(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                                const imd_options_t* options, int32_t n_lo, int32_t n_hi,
                                const imd_err_request_t* req, imd_err_t* out);

/* appended (v8):                                                                             */
imd_perf_request_t       { ...; const imd_err_request_t* accuracy; }          /* 40 -> 48 B  */
imd_perf_t               { ...; double err_bound, err_typical; int32_t err_status, err_metric; }
imd_perf_rank_request_t  { ...; const imd_err_request_t* accuracy; }          /* 88 -> 96 B  */
imd_perf_candidate_t     { ...; double err_bound, err_typical; int32_t err_status, infeasible; }
enum { IMD_PERF_INFEASIBLE_STATUS = 1, IMD_PERF_INFEASIBLE_BUDGET = 2,
       IMD_PERF_INFEASIBLE_ACCURACY = 4 };                                    /* infeasible bits */
```

**One implementation.** The formulas live in `src/host/accuracy.cpp`. `imd_err_estimate`,
`imd_err_min_moduli`, predict and rank all call the same internal function; Python only
marshals.

**Rigor in floating point.** E is computed in double and multiplied by (1 + 2⁻⁴⁰).
- Each of its ~20 roundings, T → double included, is below 2⁻⁵³ relative, so the result is
  never below the exact real bound.
- The test checks this against `params.error_estimate`'s exact rational enclosure, and checks
  that the result is at most 2⁻³⁹ above it.

**predict.** A v8 `imd_perf_t` reports `err_bound` and `err_typical` of the predicted
configuration, in the request's metric.
- With `accuracy` NULL (or a v7-sized request): NORMWISE, no target, `err_status` UNCHECKED.
- With a target: MET or MISSED, by `basis`.
- A statistic missing for the metric is INVALID_VALUE.
- `imd_perf_predict_pack` (no product) leaves the err fields 0.

**rank.** With `accuracy` and a target, a candidate whose estimate (by basis) exceeds the
target is **infeasible**: `feasible` = 0 and `infeasible` has `IMD_PERF_INFEASIBLE_ACCURACY`.
- A budget miss sets `_BUDGET`; a plan-time status sets `_STATUS`. A candidate is feasible
  iff `infeasible` = 0.
- The order is v6's:
  1. feasible candidates, by time;
  2. infeasible SUCCESS candidates (budget or accuracy), by time;
  3. the rest.
- Ties are among the feasible candidates only.
- So `rank(...)[0]` is **the cheapest configuration that meets the target only if it is
  `.feasible`**. Otherwise no candidate meets target and budget, and `[0]` is the fastest
  infeasible SUCCESS candidate (or the first non-SUCCESS one).
  Every candidate also carries its estimates, so a caller can relax the target without
  re-ranking.
- A v8 candidate reports the NORMWISE estimates even without `accuracy`. A candidate whose
  plan-time status is not SUCCESS has no plan: +∞, UNCHECKED, `infeasible` = `_STATUS`.
- Requests and candidates shorter than v8 behave as v7. A v8 request with v7-sized
  candidates (`candidate_size` 168) still marks accuracy misses infeasible (`feasible` = 0, and
  the order is the same); only the err fields are not written.

**Minimal N on the hot path.** `imd_err_min_moduli` is one lowering (0.3–0.7 µs) plus at most
19 closed-form evaluations, with no performance model.
- Time grows with N at fixed (g, vec mode), so the recommended pattern is
  `n = min_moduli(acc)`, then `rank([n, n+1], ...)`.
- Ranking from N = 2 upward predicts candidates the target rules out.

**Python.**
- `host.py`:
  - `Accuracy(metric=, target=, cond=, norm_product=, l1_ratio_a=, l1_ratio_b=, basis=,
    fail_prob=)`, a frozen dataclass marshalled to the C struct;
  - `err_estimate(N, k, rounding=, dispatch=, outputs=, accuracy=)` and
    `min_moduli(spec, extents, accuracy, rounding=, mode=, n_lo=, n_hi=)`, which return
    `ErrEstimate`;
  - `predict(..., accuracy=)`, `rank(..., accuracy=)`;
  - `Prediction.err_*`, `Candidate.err_*` / `.infeasible`.
- `planner.py`:
  - `planner.Accuracy(...)` builds the C struct once (immutable, hashable);
  - `Request(accuracy=acc)` and `q.rank(..., accuracy=acc)` take it;
  - `q.min_moduli(acc, rounding=, mode=, n_lo=, n_hi=)` and
    `Planner.min_moduli(spec, extents, acc, ...)` return a lazy `ErrEstimate` view
    (memoizable).
- Results are bit-identical to host.py (`host_planner`).

### 6f.4 Tests

**`host_accuracy` (CPU):**
- `imd_err_estimate` vs `params.error_estimate` on 20 000 random (N, k, rounding, dispatch,
  M, metric, statistics) cases: the rigorous bound lies in [exact, exact·(1 + 2⁻³⁹)], and the
  typical estimate agrees to 1e−11 relative (it may take the min with the inflated E); ABSOLUTE bounds
  also within 2⁻¹⁰⁶⁷ absolute; 2000 subnormal-`norm_product` cases.
- Invalid requests are rejected.
- min_moduli equals a brute-force scan of `imd_err_estimate` (and of params), for both bases.
- predict and rank:
  - the estimates equal `imd_err_estimate` at the candidate's (N, k, dispatch, M);
  - feasibility = (status, budget, estimate ≤ target);
  - the order is as specified;
  - the best feasible candidate is the fastest among those meeting the target;
  - v7-sized structs behave as v7.

**`params.py --check`:** `cert_error_estimate` checks the composed bound (with ν, ρ,
norm_product, and DIRECT's γ_k) against `emulate_element` / exact rationals on random rows, in
both roundings. [review] The rows include constant, non-representable rows and k up to 1200
at N = 3–5.

**`host_planner`:** Accuracy, min_moduli, and predict / rank with accuracy, vs host.py.

**`accuracy_err_api_quick` (GPU):** `bench/accuracy/err_api.py` checks that the measured error
is ≤ `err_bound` in every metric on random cases. The full run is a few hundred cases; it also
reports the typical estimate's coverage.

### 6f.5 Spec

**No TLA+ change.** The new functions are pure functions of scalars, descriptors and
caller-owned plain structs: no handle, no state, no lifetime, no GPU (the §6c.6 argument).
Rank's feasibility only compares one more number with a threshold. The bound itself is a
[cert] (params.py) with its proof in PROOFS L3; it is not a protocol property.

### 6f.6 Design questions

These are for the reviewer: NOTES-B2 §7.4.

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

## Changes v1.4 → v1.5 (shape-dependent K4 configurations, docs/NOTES-A3.md)

- **§4 K4 configurations.** Four K4Gemm instantiations (128×256, 128×128, 64×128, 64×64
  threadblock tiles); a plan runs one, chosen at plan time by `k4_select` (libimmerdink_host)
  from the GEMM view, N, g and the SM count. Same §3.5 arithmetic, bit-identical W, same
  launches, workspace and kernel sequence: no cert, spec or Apalache change.
- **§6 `k4_tune`** (optional, default off): plan-time autotuning over the configurations,
  with temporary device memory of `k4_tune_bytes`. `imd_options_t` grows to 24 bytes
  (rebuild, as in v1.2). Plan creation already runs device work with `IMD_VEC_RESIDENT`
  (K0 into plan memory, then a wait); tuning adds launches into private temporary memory
  that no call reads, so the HostAPI and Pipeline models are unchanged.
- **§6a.** `IMD_HOST_API_VERSION` 3: `imd_lowering_t` grows to 168 bytes with `sm_count`
  (input and output), `k4_config` and `k4_tune_bytes` (appended; version-1/2 callers
  unaffected). libimmerdink_host stays the one implementation of the choice.
- **§4 KF for k̂ ≤ 1024** (was 512): plans with 512 < k̂ ≤ 1024 take the fused launch
  (`fused` = 1, `kernel_launches` accordingly). Same outputs and workspace; Pipeline's
  `Fused` is k̂-independent, so no spec change.

## Changes v1.5 → v1.6 (fused K1 + pack for large operands, docs/NOTES-D4.md)

- **§4 KL.** For k̂ > 1024 (no KF), each operand whose `fused_large` bit is set is scaled and
  packed for group 1 by one launch KL instead of K1 + K2/K3 (+ the zero-rows launch for B):
  one CTA per row tile owns whole rows, reads them once from DRAM (chunk 0 in registers, the
  last chunk in shared memory, middle chunks re-read from L2 within a 64 MiB budget of rows
  in flight), computes σ with the same exact integers as K1 and stores σ, the flags and
  group 1's residues exactly as the separate kernels did. K0 stays a separate launch (KL
  reads the vectors). Results and the §4a workspace are unchanged; for g < N later groups
  still re-read the operand (holding all N residues would be the g = N workspace).
  **[spec: Pipeline FusedLarge; pos_large_g3, pos_large_g1, pos_large_a_g2,
  pos_large_b_resident, pos_large_batch2; neg_large_with_k0, neg_large_row_split,
  neg_large_no_sigma, neg_large_all_groups]**
- **§6a.** `IMD_HOST_API_VERSION` 4: `imd_lowering_t` grows to 176 bytes with `fused_large`
  and `reserved4` (appended; version-1/2/3 callers unaffected); `kernel_launches` counts KL.
  libimmerdink_host stays the one implementation of the choice (`fused_large`) and the count
  (`plan_launches`).
- **§6 test hooks.** `imd_test_set_fused` can also force KL for both operands of every emulated
  plan (and KL with 64-row tiles, so short rows span several chunks), for the bit-exactness
  tests.

## v1.7 (D1 packed operands; design approved 2026-10-03, implemented 2026-10-03)

- **Implementation** (docs/NOTES-D1.md): libimmerdink_host `pack.cpp` / `analyze_packed`
  (host API version 5), libimmerdink `imd_pack` (the KL kernel with g = N, one prep
  launch), `imd_plan_create_packed`, `imd_contract_packed`; §6b.11 lists the
  implementation choices and the clarification of §6b.5 step 1 to confirm. §6 points to
  §6b; §6a gains version 5 of `imd_lowering_t`. **[spec: IndexMap adopted orders (+10 pos /
  +8 neg), Pipeline packed slot (+9 / +8), HostAPI pack handle (+3 / +7); params
  cert_pack_layout; oracle pack / lower_packed; 284 TLC models]**
- **§6b (new).** An explicit packed-operand handle (`imd_pack_create` / `imd_pack` /
  `imd_plan_create_packed` / `imd_contract_packed`) and the CUDA-free sizing and
  compatibility queries (`imd_pack_layout`, `imd_pack_desc_for`, `imd_*_packed`,
  `imd_lowering_t` v5, `IMD_STATUS_PACK_MISMATCH`). The compatibility analysis: the packed
  bytes depend on X's values over its K cells, k, N, rounding and the R/L/K enumerations,
  not on the partner, C, g, parts, chunking, K4 configuration, `vec_mode` or KF; v1.5's
  canonical orders and the swap couple X to the partner and C, so the pack fixes its own
  orders and plans adopt them (bit-identical for unflagged outputs, and for all outputs in
  the adopted order). The key (format, N, rounding, b·r·k, the label-free FP64 element map in
  canonical order) is compared word for word. Byte layout format 1: σ, flags, all N planes
  `[l][λ][ρ][k̂]` with zero K padding and a zeroed tail, role-agnostic and g-free.
  Ordering (RAW/WAR on the packed buffer and the FP64 operand) is the caller's. **[spec:
  PackedOperand (new): 11 pos / 17 neg]**. §1, §4, §4a, §5, §6, §6a are unchanged until
  approval; §6 "Reserved" points here. Open questions: §6b.10.

## A5 (kernel-internal; no semantic, layout, API or spec change)

Small-shape polish (docs/NOTES-A5.md). Same kernel sequence, launches, workspace, plan bytes
and host API; every value the same integer (bit-identical, tested on every contract suite).
- **§4 K5.** One kernel per N in [2, 20] (instead of one per limb count with N as an
  argument): the §3.6 plane loop and constants are compile-time. K5 j̄-fastest takes 1, 2 or 4
  j̄ per thread by the output count (b·m·n ≤ 1024·SMs: 1; ≤ 16384·SMs: 2; else 4); the
  λ-fastest kernel is per N too.
- **§3.4 packs (KF, KL, K2/K3).** The byte-weighted residue sum s < 7·255² is reduced with the
  32-bit magic ⌈2³²/p⌉ (2²⁴ for p = 256, exact: s·e < 2³² for e < p ≤ 256; exhaustive CPU test)
  instead of the 64-bit one, and the sign is applied after SMod (SMod(−r mod p) = −SMod(r) for
  odd p; for p = 256 both give the byte (−r) mod 256). Same int8 residues.
- **§3.2 step 3 (K1, KF, KL).** The term ⌈m²·2ᵗ⌉ is computed with the significand normalized to
  53 bits (m′ = m·2ᶻ, the same real number m′²/2ˢ with s = 2(γ − ilogb x) + 104 − F ≥ 44) in two
  64-bit words instead of generic 128-bit shifts (`square_term_n`; equal to `square_term` on
  2²⁸ sampled (x, γ) pairs including every shift branch and subnormals, GPU test).

## Changes v1.7 → v1.8 (B2 performance model)

- **§6c (new).** A CUDA-free per-phase performance model in libimmerdink_host (host API
  version 6): `imd_perf_predict` / `imd_perf_predict_pack` (per-phase and total time,
  TFLOPS-eq, MEMORY_MODEL §4 bounds), `imd_perf_rank` (N × g × `vec_mode` × packed
  candidates ranked by predicted time inside a memory budget, with exact footprints and
  plan-time statuses), and the calibration (`imd_perf_calib_*`: a versioned `key = value`
  text file; the RTX 4090 calibration embedded; `tools/calibrate.sh` fits a new GPU). Each
  phase time is linear in calibrated per-kernel parameters over exact features of the
  lowering. No kernel, layout, launch or plan changes; the calibration is never read by plan
  creation. No spec change (pure functions over caller-owned data, §6c.6).
- **§4 KL**: the tile rule (`tk`) moves to a host header used by `launch_kl` and the model.

## v1.9 (B1 Python front end; no semantic, layout, C API or spec change)

- **§6d (new).** `python/immerdink` device API over the §6 / §6b C ABI with ctypes:
  `Plan`, `Pack`, `Handle`, `PlanCache`, the einsum front end `immerdink.contract`, raw
  entry points (`contract_ptr`, `pack_ptr`). Arrays through `__cuda_array_interface__`,
  DLPack or raw pointers (`DeviceArray`), no array library required; float64 only; Python
  checks only what the C API cannot see (shapes and strides against the plan, dtype,
  device, writable outputs, the CAI v3 stream with a spec-sanctioned opt-out); statuses map
  to exception classes. No CUDA call of its own besides `cudaGetDevice` (and
  `cudaDeviceGetAttribute` for `Plan.lowering`); capture-safe calls; workspace always the
  caller's; RESIDENT plans live as long as any graph that captured them. No TLA+ change
  (§6d.8 says why). Tests: `python_device`. Reviewed 2026-10-05 (CAI stream "check" by
  default, `PlanCache.clear()` drops GENERATE plans, `freeze()`, pack use counts).

## v1.10 (§6c fidelity follow-up: host API 7, calibration format 2; no semantic, layout, launch or spec change)

- **§6c.1** Wave / occupancy `chain` features for K0, KL, PACK, K5, DIRECT and BETA (KF's chain
  generalized) from the kernels' launch rules, now shared with `contract.cu` in
  `src/host/analysis.hpp` (`launch_blocks`, `kl_launch`, `pack_kl_lg_rows`, `pack_launch`,
  `k5_launch`); CTAs per SM measured at calibration time. Clock regimes (BURST, SUSTAINED with
  a power-cap model), cache state (COLD, WARM), prediction intervals and tie-aware ranking.
- **§6c.2** Calibration format 2 (occupancy table, power, warm and band keys); format 1 still
  read, migrated to v6-identical predictions. New sweeps for sustained windows and cache state;
  `imd_bench --methods occupancy | sustained | pair`, `--l2 flush | repeat`, per-call clock in
  every record.
- **§6c.3** Host API version 7: appended fields in the calibration, request, prediction, rank
  request and candidate structs; `imd_perf_kernel_name`.
- **§6c.4 / §6c.5** Tests and validation targets for the new features.

## v1.11 (hot-path planner queries; no semantic, layout, C API or spec change)

- **§6e (new).** `python/immerdink/planner.py`: `Query` (a contraction's C descriptor structs
  built once; option, packing, request and default-calibration structs reused; lazy result
  views; `rank` in one C call), `Planner` (bounded Query cache, host.py's signatures),
  `Memo` (explicit, thread-safe memo of identical queries, bounded by entries and by exact C
  struct bytes), `Request`. Results bit-identical to host.py; per query 0.8–1.8 µs instead of
  8–20 µs (C floor 0.25–0.86 µs), `rank` 7–17 µs instead of 32–63 µs, a memo hit 0.4–1.0 µs.
  Pure ctypes: host API unchanged (version 6), no nanobind (§6e.2). `bench/planner_floor.c`
  (`imd_planner_floor`) times the C functions alone; `bench/planner_overhead.py` prints
  both. No TLA+ change (§6e.6). Tests: `host_planner`. Design questions: §6e.7.

## v1.12 (accuracy-aware planning: host API 8; no semantic, layout, launch or spec change)

- **§6f (new).** `imd_err_request_t` gives the metric (normwise, elementwise or absolute), the
  target, and the caller's operand statistics (ρ, a norm-product bound, optional ℓ1/ℓ2 ratios).
  Two estimates come back:
  - a rigorous bound (PROOFS L3a composed with the final rounding; DIRECT: γ_k);
  - a typical estimate (Hoeffding under independent zero-mean RN quantization errors, RN
    EMULATE only).

  The API additions:
  - `imd_err_estimate` (scalars) and `imd_err_min_moduli` (one lowering, smallest N meeting
    the target);
  - predict reports `err_bound` / `err_typical`;
  - rank flags candidates that miss the target as infeasible (`infeasible` bits) and keeps
    v6's order.

  Python: `host.Accuracy`, `host.err_estimate`, `host.min_moduli`, and `planner.Accuracy` /
  `Query.min_moduli`. Reference and certificate: `params.error_estimate`,
  `cert_error_estimate`. No TLA+ change (§6f.5).
- **§3.8.** A pointer to §6f.

## v1.13 stage 1 (K4 load alignment: K_c' = 131008; no semantic, layout, pack format, API or calibration-format change)

- **§2, §3.5.** `K_c' = ⌊K_c/64⌋·64 = 131008` replaces 131056 (`⌊K_c/16⌋·16`). It is
  derived in `spec/params.py` (`K_CHUNK_ALIGN = 64`), not written as a literal. cert_kc
  checks `K_c' % 64 = 0` and `K_c − K_c' < 64`. The epilogue range for p = 256 is
  `[−2129666048, 2146435327]`; the regenerated `epi_vmin` / `epi_vmax` are in
  `constants.hpp`, which also gains `kKChunkAlign`.
- **§4b (new).** Why: the measured K4 load-phase rule `φ = (r·k̂ + chunk end) mod 64`, with
  CUTLASS loading the residue tile first. With 131056 (≡ 48 mod 64), an aligned k̂ had a
  fully misaligned chunk 0 (2048² × 196608, N = 8, 128×256: 23.4 TFLOPS-eq, 70.5 ms per
  call, K4 45.7 ms; measured before the change, NOTES-K4ALIGN). With 131008 the phase class
  depends on k̂ alone, and no chunk gets worse. Measured after the change (interleaved A/B):
  that call runs at 32.0 TFLOPS-eq (51.5 ms, K4 25.9 ms), and 196640 goes from 25.8 to
  31.8 TFLOPS-eq. Odd multiples of 16 and the window cases are unchanged within 2 %.
- **Results are bit-identical**, since the chunking does not change W or C.
  - The calls with `k̂ ∈ (j·131008, j·131056]` gain one K-chunk: one more add-mod launch per
    group and part, one more W pass, and possibly one more add-mod shape key.
  - Every other host query value is unchanged.
  - A pre-check before the change ran 11 long-k cases with `imd_test_set_limits(k_chunk =
    131008)` on the old code. All were bit-exact against the oracle at chunk 131008 and
    identical to the 131056 run.
- **Stage 0 (refactor, no behaviour change).** `k_hat_of(k, A)` (`src/host/k_hat.hpp`) is the
  one rounding of k to k̂ in libimmerdink_host, with A = 16 for every plan. It is mirrored
  by `params.k_hat_of`.
- **§6c.** A correction to the v1.10 per-chunk text: the criterion is the chunk end, not
  the start, and the 262144 effect was the residue-first phase, not a DRAM-channel effect.
  The format-2 criterion is now exactly "k̂ mod 32 ≠ 0". There is no format change, and θ is
  to be refit.
- **Spec.** `spec/apalache/Epilogue.tla`: `Kp = 131008`, with the tight max 2146435327
  (new counterexample value). The comments in Pipeline, MC_Pipeline, OzakiCRT and
  Int32Chunk change; no TLC model changes. `params.cert_row_phase` (new) covers the boundary
  k list and the 131008 / 131056 windows. Lean `Chunk.lean` (≤ 131056) covers 131008
  a fortiori and was not re-instantiated.
