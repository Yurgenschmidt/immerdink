# immerdink — design (v1.1)

This is the single source of truth for *what* v1 computes and *how*.
- TLA+ specs in `spec/` check the claims marked **[spec: X]** (see `spec/README.md`).
- `spec/params.py` checks the real-width arithmetic facts, marked **[cert]**.
- `spec/oracle.py` is the exact, bit-level executable reference of §3. GPU results must
  match it bit for bit.
- Hand proofs of the parametric lemmas are in `docs/PROOFS.md`; they are the blueprint for
  the Lean formalization.
- **[deviation]** marks a departure from published Ozaki-II, with the reason.

v1.1 folds in the corrections found by the spec phase: adversarial review plus mutation
testing of every artifact. The changes from v1 are listed at the end.

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

The canonical orders are a fixed function of the descriptors (§5 step 8).

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
  k").
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
plan time:  IndexMap lowering → scatter vectors, canonical orders, chunking, launch split
K0 vec      cudaMemcpyAsync: plan's pinned scatter vectors → ws.vec        (same stream)
K1 scale    A rows, B cols: flags + σ, τ (int16)       reads A, B once
for each residue group G (g moduli; ⌈N/g⌉ groups):
  K2 pack   A → A'_l[λ][ī][k̂]  for l ∈ G   (gather via scatter vectors, smem transpose,
  K3 pack   B → B'_l[λ][j̄][k̂]  for l ∈ G    K-contiguous int8, zero padding)  re-read A, B
  K4 gemm   for each K-chunk c: CUTLASS batched int8 TN over z = s·b + λ (s = slot in G),
            epilogue store-mod (c = 0) / add-mod (c > 0) → u8 planes W_l[λ][ī][n̂]
K5 crt      per (λ,ī,j̄): flagged ? native dot (reads A, B) : exact CRT; unscale; α/β; scatter
```
If `α = 0` or `k = 0`, only K0 and K5 (β-only) run.

**Per-batch modulus in K4.**
- One batched launch covers several moduli. A plain `VisitorScalarBroadcast` cannot
  express `slot = z div b`, because that is not linear in z.
- The epilogue's compute node therefore reads `p` and the Barrett constant from a
  `__constant__` table indexed by `g0 + blockIdx.z / b`.
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
  - No buffer is read before it is written for the current group or chunk.
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
  - Safe two-stream operation needs three event edges: K1 → pack (RAW), pack(G) → gemm(G)
    (RAW), and last gemm(G) → pack(G+1) (WAR). With single-buffered RA/RB those edges
    serialize everything anyway.
  - So v1 is single-stream.

**Launch limits.**
- The CUTLASS batch count `|G|·b` must be ≤ 65535 (gridDim.z), so plan time splits it.
- `m, n̂ ≤ 2³¹−1` (`GemmCoord` is int), and the per-launch K is ≤ K_c'.
- `lda = ldb = k̂` and `ldc = n̂`. The batch strides are int64, so k̂ itself may exceed 2³¹.

**GEMM configuration [CUT]:**
- CUTLASS 2.x `gemm::device::GemmUniversal`, mode `kBatched`, `arch::Sm80` on sm_89.
- int8 `mma.sync.m16n8k32`, `OpMultiplyAddSaturate`.
- TN layout: A RowMajor, B ColumnMajor, both K-contiguous. Alignment is 16, hence
  `k̂ = ⌈k⌉₁₆` and `n̂ = ⌈n⌉₁₆`.
- `Sm80EVT` epilogue:
  - a compute node implementing §3.5's exact reduction;
  - `VisitorAuxLoad<u8>` for chunk continuation;
  - `VisitorAuxStore<u8>`.
- Tile menu (99 KB of smem): 256×128×64 s3 or 128×256×64 s3; 128×128×64 s4–5;
  64×128×64 s6; 64×64×64 s6–10. Chosen by wave quantization over 128 SMs.
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
                                        C: row,col,batch)
RA          g·b·m·k̂                     int8 residues of A for one group
RB          g·b·n̂·k̂                     int8 residues of B for one group
W           N·b·m·n̂                     u8 modular products, all N planes

workspace_size = Σ_regions align(bytes), regions in this order at 256-aligned offsets
               = 0                if b·m·n = 0 (empty output: no-op)
               = align(vec)       if mode = DIRECT, K-empty, k = 0
```
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
8. **Canonical orders**, a pure function of the descriptors:
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
| 5 | overflow: extent products or spans exceed int64; workspace total exceeds uint64; in EMULATE mode with a non-empty output, m or n̂ > 2³¹−1 (CUTLASS limit; not applied to DIRECT) | INVALID_VALUE |
| 6 | C ∩ A or C ∩ B (byte intervals; checked even when k = 0, conservatively); workspace ∩ A, B or C, using the bytes actually used, `[ws, ws + workspace_size)` | ALIASING (execute time) |
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
                 int32_t mode; } imd_options_t;  /* IMD_EMULATE | IMD_DIRECT  */

imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                const imd_tensor_t* C, const imd_options_t*, size_t* bytes);
imd_status_t imd_plan_create(imd_handle_t, const imd_tensor_t* A, const imd_tensor_t* B,
                             const imd_tensor_t* C, const imd_options_t*, imd_plan_t*);
imd_status_t imd_contract(imd_plan_t, double alpha, const double* A, const double* B,
                          double beta, double* C, void* ws, size_t ws_bytes, cudaStream_t);
imd_status_t imd_create(imd_handle_t*);           /* lifecycle */
imd_status_t imd_destroy(imd_handle_t);
imd_status_t imd_plan_destroy(imd_plan_t);
const char*  imd_status_string(imd_status_t);
```

- `imd_workspace_size` is a **pure host function**: no GPU, no allocation, and exactly
  §4a.
- Statuses: SUCCESS, INVALID_VALUE, NOT_SUPPORTED, ALIASING, INSUFFICIENT_WORKSPACE,
  MISALIGNED, ALLOC_FAILED (host allocation in plan creation), CUDA_ERROR.
- `imd_plan_create` reports the §5 statuses and the RN acceptance rule (§3.2: κ ≥ T →
  NOT_SUPPORTED). That rule applies only when the emulated GEMM would run;
  `imd_workspace_size` is exactly §4a and never applies it.
- `imd_contract` checks, in order:
  1. null pointers → INVALID_VALUE;
  2. an empty output → SUCCESS with nothing touched;
  3. ALIASING (§5 row 6);
  4. MISALIGNED;
  5. `ws_bytes < workspace_size` → INSUFFICIENT_WORKSPACE.
- `imd_contract` does no host synchronization and no allocation, and has no hidden mode
  switches.
- A workspace belongs to at most **one** in-flight `imd_contract` at a time.
- Callers tile I/J/L themselves, since sub-views are valid inputs. Tiling over I/J/L leaves
  results bitwise unchanged; tiling over K by the caller does change them. The library
  handles K chunking and the batch-launch split internally.
- **Reserved, not v1** (the API and specs must not preclude these):
  - a packed-operand API `imd_pack` / `imd_contract_packed`. Fast-mode σ depends only on
    the operand itself, so packs are reusable across contractions with the same K labels,
    N and rounding;
  - complex values via the 2M method (2 GEMMs per modulus, moduli with √−1);
  - accurate-mode scaling, which is incompatible with independent packing.

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
