# immerdink — design (v1)

This is the single source of truth for *what* v1 computes and *how*, narrowed as far as the
research allows. TLA+ specs in `spec/` check the claims marked **[spec: X]**. The
real-width arithmetic facts are checked by `spec/params.py` **[cert]**. Where a choice
departs from the published Ozaki-II algorithm, it is marked **[deviation]** with a reason.

Status legend: **Decided** (D-numbers in SCOPE.md), **Open (user)** (see PLAN.md §Questions).

Sources: Ozaki–Uchino–Imamura arXiv:2504.08009 v4 [OZ2]; Uchino et al. arXiv:2508.03984
[SC25]; error analysis arXiv:2602.02549 [ERR]; complex arXiv:2512.08321 [CPLX] and
arXiv:2609.05419 [2M]; GEMMul8 v3.5.2 (commit 6a9eff8) [G8]; CUTLASS v4.8.0 [CUT].

---

## 1. What is computed

Inputs:
- FP64 tensors A, B, C with arbitrary int64 strides;
- integer labels per mode;
- FP64 scalars α, β;
- options: `N` (number of moduli, 2…20), `rounding` (RN default, TRUNC), `g` (moduli per
  residue group, 1…N, default N).

Output:

```
C[I,J,L] ← α · Σ_K A[I,K,L] · B[K,J,L]  +  β · C[I,J,L]
```

The sum is computed by the integer pipeline below. Results are **bitwise deterministic**: a
pure function of (input bits, α, β, N, rounding, library version). They do not depend on
launch configuration, tile shapes, K-chunking, residue grouping `g`, or stream. **[spec:
Pipeline, OzakiScaling]**

### 1.1 The GEMM view (IndexMap, §5)

Label analysis lowers every legal contraction to `b` independent GEMMs over canonical index
spaces. Here `ī ∈ [0,m)`, `k̄ ∈ [0,k)`, `j̄ ∈ [0,n)` and `λ ∈ [0,b)`:

```
C_λ[ī,j̄] ← α Σ_k̄ A_λ[ī,k̄] B_λ[k̄,j̄] + β C_λ[ī,j̄]
```

The **canonical k̄ order** is a fixed function of the labels and strides, so it is the
same for every call with the same descriptors. It fixes the K-chunk boundaries and the
native-path summation order. Neither affects emulated results (they are exact), but the
native path's result does depend on it (§3.7).

---

## 2. Constants (offline, exact; [cert])

| Symbol | Definition |
|---|---|
| `p_1..p_20` | greedy pairwise-coprime moduli from 256 down: **256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173**. Identical to [G8] `table.hpp`. [OZ2] eq. int8_law8 omits 241; that is an erratum, so don't "fix" the table to match it. |
| `P` | `∏_{l≤N} p_l` (even, since 256 is in every table prefix). `log2 P` = 63.58 / 71.41 / 79.24 / 110.16 / 117.78 / 125.38 at N = 8 / 9 / 10 / 14 / 15 / 16. |
| `H` | `⌊(P − 1)/2⌋`, the magnitude the CRT can represent uniquely (centered range). |
| `M_l, q_l` | `M_l = P/p_l`, `q_l = M_l⁻¹ mod p_l ∈ [1, p_l)`. |
| `T` | `⌊√H⌋`, the per-side norm budget: `‖a'‖₂ ≤ T ∧ ‖b'‖₂ ≤ T ⇒ Σ|a'||b'| ≤ T² ≤ H`. |
| `K_c` | `131071 = ⌊(2³¹−1)/128²⌋`, the max K-chunk length. All residues satisfy `|r| ≤ 128`. CUTLASS 2.x int8 MMA is `.satfinite` ([CUT] `mma_sm80.h`), so overflow would **saturate silently**. The p = 256 "wraparound" exemption used by [G8] on cuBLAS **does not apply** to us. |
| `L_w` | CRT word width in bits. It must satisfy `2P ≤ 2^{L_w}`: 128 for N ≤ 16, 160 for N ≤ 20. |

All constants are generated and checked by `spec/params.py` in exact integer arithmetic; it
is the only producer of the device constant tables. **[cert]** [G8]'s N=2 `log2P` entry is
mis-rounded. We use no floating-point constants at all.

---

## 3. Numerics, step by step (per batch λ; rows of A shown, columns of B are symmetric)

### 3.1 Non-finite scan (fused into §3.2's pass)
- `fA[ī] = 1` iff row ī of A_λ contains Inf or NaN. `fB[j̄]` is defined the same way for columns of B.
- Non-finite entries are treated as 0 by every later emulated step.
- Flagged outputs are produced by the native path (§3.7).

### 3.2 Scaling: exact-integer fast mode (Cauchy–Schwarz) — **[deviation]**
For row ī with finite entries `a_h`:
1. `γ = ilogb(max_h |a_h|)` (if the row is all zero: `σ := 0`, go to 4).
2. Normalize `x_h = a_h · 2^{−γ}` (exact; `|x_h| < 2`).
3. Fixed-point sum of squares: `S = Σ_h ⌈x_h² · 2^F⌉` with `F = 60`.
   - Each term is computed exactly (53×53-bit product, ceiling shift), so `S` is an exact
     integer.
   - The sum is order-independent and fits in uint128 for `k < 2⁶⁶`.
   - `S ≥ ‖x‖²·2^F`.
4. Shift: `σ = u − γ`, where `u` is the **largest integer** with
   - RN: `4 · S · 2^{2u} ≤ (2T − κ)² · 2^F`, where `κ = ⌈√k⌉`;
   - TRUNC: `S · 2^{2u} ≤ T² · 2^F`.
   - If `S = 0`, then `σ := 0`.
   - Plan time rejects `κ ≥ T` in RN mode (NOT_SUPPORTED: "N too small for this k").
5. Store `σ` as int16. The range is about [−1100, 1160], checked by **[cert]**.

**Guarantee:** `‖a'‖₂ ≤ T`, and therefore `Σ_h|a'_h||b'_h| ≤ H` for every (ī, j̄). That
gives unique CRT reconstruction. **[spec: OzakiScaling]**

Why we deviate:
- Both [OZ2]/[SC25]/[ERR] and [G8] choose the shift with floating-point `log2`/`__log2f`.
  [G8] also squares raw entries, which overflows for `|a| ≥ 2⁵¹²` and silently corrupts the
  row.
- The integer rule is overflow-free and order-independent, so it is reproducible and
  model-checkable.
- It is at least as tight: it finds the maximal `u` exactly instead of rounding a log down.

### 3.3 Quantization — **[deviation: RN by default]**
- `a'_h = RN_even(a_h · 2^σ)`; in TRUNC mode, `trunc(a_h · 2^σ)`.
- The value is held exactly as `μ · 2^ε` with `|μ| ≤ 2⁵³` and `ε ≥ 0`. It is never forced
  into int64, since `|a'| ≤ T` reaches 2⁷⁷ at N = 20.
- Why RN rather than [OZ2]'s trunc:
  - RN is unbiased and halves the per-entry error.
  - Trunc biases every product toward zero. That bias is systematic in CC energy
    accumulations (CC study, unverified magnitude).
- The price is the `κ/2` slack in §3.2, which costs `< 2⁻²⁰` of the budget for N ≥ 8.
- TRUNC stays available for paper-faithful comparisons.

### 3.4 Residues
- `r_l(a') = SMod(a', p_l) ∈ [−⌊p_l/2⌋, ⌈p_l/2⌉ − 1]`, which fits int8 (`[−128, 127]` for
  p = 256).
- It is computed exactly as `SMod((μ mod p_l) · (2^ε mod p_l), p_l)`, using a table of
  `2^ε mod p_l`.
- Only the congruence class matters, so any exact reduction (Barrett, dp4a byte weights) is
  allowed. **[spec: OzakiCRT, OzakiScaling]**

### 3.5 Per-modulus INT8 GEMM with modular epilogue
- Split canonical k̄ into chunks of length `K_c' = 131056 = ⌊K_c/16⌋·16`; the last chunk
  may be shorter. Chunk starts are multiples of 16 bytes, so every operand pointer stays
  16-byte aligned for CUTLASS.
- For chunk c: `acc = Σ_{h∈chunk} r_l(a'_īh) · r_l(b'_hj̄)` in int32. This is exact in any
  accumulation order, since `128²·K_c ≤ 2³¹−1`.
- The epilogue stores u8:
  - first chunk: `W ← acc mod p_l ∈ [0, p_l)`;
  - later chunks: `W ← (W + acc) mod p_l`.
- Final `W_l ≡ (A'B')_īj̄ (mod p_l)`. **[spec: OzakiCRT (KSplit)]**

### 3.6 CRT reconstruction (exact, multi-word) — **[deviation from D10 as first stated]**
```
x ← 0
for l = 1..N:  y ← (q_l · W_l) mod p_l          (8-bit × 8-bit)
               x ← x + M_l · y                  (L_w-bit; M_l·y < P)
               if x ≥ P: x ← x − P              (invariant x < P; peak < 2P ≤ 2^{L_w})
z ← (x ≤ H) ? x : x − P                         (centered; z = (A'B')_īj̄ exactly)
```
- No division, no floating point, and no `M·256` headroom. That lifts the int128 cap from
  N ≤ 15 to N ≤ 16; 160 bits covers N ≤ 20. **[spec: CRTRealization]**
- [G8]'s Garner tree is an equivalent exact alternative.
- [OZ2]'s double-double FP reconstruction is superseded: its error term `R_64` disappears.

### 3.7 Output
- `D = RN_double(z · 2^{−(σ_ī + τ_j̄)})`: **one rounding**, directly at the final exponent,
  including the subnormal range and overflow to ±Inf.
  - [G8] rounds twice for subnormal results.
- Flagged elements (`fA[ī] ∨ fB[j̄]`) instead use `D = native FP64 Σ_k̄ a·b`, computed in
  canonical k̄ order with `fma` (round to nearest). This propagates IEEE Inf/NaN like a
  plain dot product.
- Then:
  - `E = (α = 1) ? D : RN(α·D)`;
  - `C ← (β = 0) ? E : fma(β, C, E)`.
  - C is **not read** when β = 0.
  - With `α = 0` or `k = 0`: `C ← β·C`, and A, B are not read (BLAS semantics).

### 3.8 Error contract (RN mode, finite inputs, no over/underflow)
```
|D − Σ_h a_h b_h| ≤ 2√k · ‖a_ī‖₂ ‖b_j̄‖₂ / (T − κ/2)  +  k·2^{−σ−τ−2}  +  u·|D|
```
The `2^{−σ}` term is `≤ 2‖a‖/(T−κ/2)`, so the second term is second-order. This is a
**normwise per (row, column)** bound, as for every Ozaki scheme. Entries far below their
row/column norm lose relative accuracy. **[spec: OzakiScaling, exact rational check at toy
precision]**

Worst-case guidance, `2√k/T` relative to `‖a_ī‖‖b_j̄‖`:

| N | k = 10⁴ | k = 2¹⁷ |
|---|---|---|
| 6 | 1.8e−5 | 6.5e−5 |
| 8 | 7.6e−8 | 2.8e−7 |
| 9 | 5.0e−9 | 1.8e−8 |
| 10 | 3.3e−10 | 1.2e−9 |
| 12 | 1.5e−12 | 5.5e−12 |
| 14 | 7.4e−15 | 2.7e−14 |
| 16 | below FP64 rounding | below FP64 rounding |

Typical errors are smaller (the rounding errors are random). The CC study suggests **N ≈ 8–9
for CCSD** and **N ≈ 6–7 for (T)** at a 1e−8 Eh energy target (to be validated, PLAN M5).
For comparison, ehrenfest's current slice lane (15 int8 GEMMs) quotes rel. err ≈ 5e−10;
immerdink reaches that around N = 10.

---

## 4. Execution pipeline (one stream, in order)

```
plan time:  IndexMap lowering → scatter vectors, canonical orders, chunking, launch split
K1 scale    A rows, B cols : non-finite flags + σ, τ (int16)            reads A, B once
for each residue group G (g moduli; ⌈N/g⌉ groups):
  K2 pack   A → A'_l[λ][ī][k̂]  for l ∈ G   (gather via scatter vectors, smem transpose,
  K3 pack   B → B'_l[λ][j̄][k̂]  for l ∈ G    write K-contiguous int8; K padded with 0)
  K4 gemm   for each K-chunk c: CUTLASS batched int8 TN over (l ∈ G) × λ,
            epilogue store-mod (c = 0) / add-mod (c > 0) → u8 planes W_l[λ][ī][n̂]
K5 crt      per (λ,ī,j̄): flagged ? native dot : exact CRT;  unscale; α/β; scatter to C
```

Pipeline properties checked by **[spec: Pipeline]**:
- **Exactly once.** Every C element is written exactly once, by K5, after all N planes for
  it are final.
- **No stale reads.** No buffer is read before it is written for the current group/chunk.
  Residue buffers are reused across groups only after their GEMMs retire.
- **Bounded workspace.** All workspace regions are disjoint and fit `workspace_size`
  (MEMORY_MODEL §3).
- **β = 0 / α = 0 / k = 0** follow the read rules of §3.7.
- **C may not alias A or B.** That is rejected at execute time. K1–K3 finish reading A and
  B before K5 writes C, but K5's native path for flagged elements reads A and B while other
  K5 threads write C. Rejecting aliasing keeps K5 race-free.
- Two streams without events produce a hazard (negative model), so v1 is single-stream.

**Launch limits:**
- The CUTLASS batch count is `gridDim.z ≤ 65535`, so plan time splits `|G|·b` into
  multiple launches.
- `m̂`, `n̂` must be ≤ 2³¹−1 (CUTLASS `GemmCoord` is int).

**GEMM configuration [CUT]:**
- CUTLASS 2.x `gemm::device::GemmUniversal`, mode `kBatched`, `arch::Sm80` on sm_89.
- int8 `mma.sync.m16n8k32` with `OpMultiplyAddSaturate`.
- A RowMajor, B ColumnMajor (TN, both K-contiguous); alignment 16, so `k̂ = ⌈k⌉₁₆` and
  `n̂ = ⌈n⌉₁₆`.
- The epilogue is an `Sm80EVT`:
  - `VisitorScalarBroadcast` for the per-batch modulus;
  - a compute node for mod-reduction (mask for p = 256, magic-multiply Barrett otherwise,
    exhaustively verified over the reachable int32 range);
  - `VisitorAuxLoad<u8>` for K-chunk continuation;
  - `VisitorAuxStore<u8>` to store.
- Tile menu for 99 KB of smem: 256×128×64 s3 / 128×256×64 s3 (large), 128×128×64 s4–5
  (medium), 64×128×64 s6 and 64×64×64 s6–10 (small/batched). They are chosen by wave
  quantization over 128 SMs.
- cuBLASLt (V0, int32 out) is kept only as a test oracle and baseline. Its IMMA path can't
  run a custom epilogue.

**Fallbacks:**
- **DIRECT mode** (native FP64, our own small kernel): for K-empty contractions
  (outer/Hadamard), `k = 1`, or when the caller passes `mode = DIRECT`.
- No cuTENSOR/cuBLAS runtime dependency. The library never silently changes mode.

---

## 4a. Workspace layout (exact; MEMORY_MODEL §3; [cert], [spec: Pipeline])

```
align(x) = ⌈x⌉₂₅₆      k̂ = ⌈k⌉₁₆      n̂ = ⌈n⌉₁₆      g ∈ [1, N]

region      bytes                       contents
sA          2·b·m                       int16 shifts σ  (A rows)
sB          2·b·n                       int16 shifts τ  (B cols)
fA          1·b·m                       u8 non-finite flags (A rows)
fB          1·b·n                       u8 non-finite flags (B cols)
vec         8·(2m + 2n + 2k + 3b)       int64 scatter vectors (A: row,k,batch; B: k,col,batch;
                                        C: row,col,batch), copied from the plan's pinned buffer
RA          g·b·m·k̂                     int8 residues of A for one group
RB          g·b·n̂·k̂                     int8 residues of B for one group (rows n..n̂ zero)
W           N·b·m·n̂                     u8 modular products, all N planes

workspace_size = Σ_regions align(bytes)        (0 if b·m·n = 0;
                                                only align(vec) in DIRECT mode or when k = 0)
```

The regions are disjoint and laid out in the order listed, at 256-byte-aligned offsets.

## 5. IndexMap: label semantics and legality (v1)

Lowering order:
1. Check extents per label.
2. **Repeated label within A or B** → diagonal view (sum its strides).
3. **Repeated label within C** → NOT_SUPPORTED.
4. Drop extent-1 modes.
5. Classify every label by its membership set:
   - in {A,C} only → I;
   - {B,C} → J;
   - {A,B} → K;
   - {A,B,C} → L.
6. **A label only in A** (or only in B) → K, with stride 0 in the other operand. This is
   the summed axis; k grows by its extent.
7. A label only in C with extent > 1 → NOT_SUPPORTED.
8. Fuse modes that are adjacent in every tensor where they appear.
9. Choose canonical orders:
   - ī, j̄, λ by C's |stride|, so CRT scatter writes coalesce;
   - k̄ by the larger operand's |stride|.
10. Build int64 additive scatter vectors (TBLIS style): `off_X = vL[λ] + vRow[·] + vCol[·]`.

Legality, always checked (no debug-only asserts, unlike TBLIS):

| Case | Rule |
|---|---|
| strides | any sign, int64; zero strides allowed in A and B |
| A vs B | may alias each other (e.g. `W = BᵀB`) |
| C self-overlap | rejected unless the **sorted-\|stride\| rule** holds: over C's modes with extent > 1, sorted by \|stride\|, each \|s₍t₎\| > Σ_{u<t} \|s₍u₎\|(e₍u₎−1). This is sufficient for injectivity. It rejected none of 20,000 random slice/permute views of dense arrays; it does reject some exotic non-overlapping layouts. |
| C vs A, C vs B | byte-interval intersection → ALIASING (execute time) |
| zero extents | BLAS rules: empty I/J/L → no-op with 0 workspace; `k = 0` → `C ← β·C` |
| overflow guards | all extent products and spans in int64; `m̂, n̂ ≤ 2³¹−1` |
| alignment | A, B, C pointers aligned to 8 bytes; workspace aligned to 256 bytes |

**[spec: IndexMap]** checks, on tiny integer tensors:
- classification is a partition;
- lowering preserves the einsum;
- gather is a bijection onto the canonical buffer (padding is inert);
- the sorted-stride rule ⇒ scatter is injective;
- `Scatter ∘ GEMM ∘ Gather = einsum` for every legal small case.

Negative models: drop the C rule; drop extent consistency; drop the stride-0 lowering.

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
```

- `imd_workspace_size` is a **pure host function** with no GPU and no allocation. It
  returns exactly MEMORY_MODEL §3, so a planner (ehrenfest's fit law) can price steps
  without a device. **[cert]** checks the formula, and Pipeline checks region bounds.
- Statuses: SUCCESS, INVALID_VALUE, NOT_SUPPORTED, ALIASING, INSUFFICIENT_WORKSPACE,
  MISALIGNED, CUDA_ERROR.
- No host synchronization, no allocation, and no hidden mode switches inside
  `imd_contract`.
- Callers tile I/J/L themselves; ehrenfest already does. The library handles K chunking and
  the batch-launch split internally.
- **Reserved, not v1** (the API and specs must not preclude these):
  - a packed-operand API `imd_pack` / `imd_contract_packed`. Fast-mode σ depends only on the
    operand itself, so packs are reusable across contractions with the same K labels, N and
    rounding. This is a key CC win: B^Q and t2 are reused, and (ia|jb) is rebuilt about 6×
    per iteration;
  - complex via the 2M method (2 GEMMs per modulus, moduli with √−1);
  - accurate-mode scaling, which is incompatible with independent packing.

---

## 7. What was considered and rejected

| Option | Why not |
|---|---|
| GETT-style code generation | Residues must be rewritten anyway, so packing makes the layout free (TBLIS-style). GETT adds an autotuner for nothing. |
| Optimized CPU kernels | No AVX-512/AMX on the dev box; emulation loses to native DGEMM there. CPU = exact Python oracle. |
| CUTLASS 3.x / CuTe collectives | No sm80/sm89 builders and no SM80 EVT in 3.x ([CUT]). |
| V0 int32 planes | 4× the bytes per plane. Ehrenfest's lane paged 209 GB through this kind of buffer. |
| Register-fused CRT (V3) | Exceeds 255 regs/thread at useful tiles (MEMORY_MODEL §7). |
| FP CRT ([OZ2] double-double) | Exact integer is simpler, verifiable and division-free. |
| Accurate mode in v1 | Needs the other operand (Ā·B̄), which breaks operand packing. Fast mode gains about 1 modulus on CC data vs k·max·max. Revisit for wide-range inputs. |
| FP8 / INT4 paths | FP8 can't hold residues exactly at speed on Ada; INT4 forces p ≤ 16. |
