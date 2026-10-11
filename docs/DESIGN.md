# immerdink — design (v1.19)

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
§4b also holds **stage 2** (implemented 2026-10-06, host API 9; marked
**[stage 2]**): `k̂ = ⌈k⌉_A` as row stride and GEMM K extent, with `A = 64` iff the logical
k > 96 and 16 otherwise (a pure function of k), pack format 2 for A = 64, the resolved
alignment in `imd_lowering_t` (host API 9) and a test hook to force A. Results stay
bit-identical.
v1.14 makes the packs' residue step branch-free (§3.4.1): every pack emitter holds |a'| as
W(N) u32 words (W = 2 for N ≤ 16, 3 for N = 17–20, a per-plan template parameter) and
reduces it with byte weights, so the ε > 0 multiply, its per-thread branch and its
lane-divergent table load are gone. The int8 bytes are identical: no semantic, layout,
pack-format, API, calibration-format or TLA+ change; changed text is marked **[v1.14]**.
v1.15 (§6c.7): measurements of long hot runs and a clock-scaled byte energy in the power
model, `power.byte_f_pj` (host API 11, optional calibration key: absent = 0 predicts exactly
as before). Implemented 2026-10-06; its design questions were decided by the independent
reviewer. v1.15 also retires the model's eps data dependence (§6c.1 "The eps path —
retired"; the v1.14 kernels have none): `eps_fraction` is validated and ignored, `eps_s` /
`eps_fraction` out are 0, the `op_eps` features are 0 and their keys optional (format 2
kept). The shipped calibration is the combined refit of both (NOTES-B2 §10).
v1.16 (§3.4.1) finishes the pack emitter: PTX shifts (no undefined shift outside the
contract), the exact maxima of the reduction sum, a W-aware occupancy query for the
performance model, and W = 1 for N ≤ 8 evaluated and not adopted (pack phases −2.7 %, below
the 3 % bar, for +11 % library size). Same int8 bytes: no semantic, layout, launch-shape,
pack-format, API, calibration-format or TLA+ change; changed text is marked **[v1.16]**.
v1.17 (§6c) adds two features to the performance model in the slots the eps path freed: KF's
per-element chain (`kf.elem`) and the PACK bytes that are not L2 hits (`pack.dram`, with the
residency size `warm.pack_l2_bytes`), all optional keys (absent = 0 predicts exactly as
before; calibration format 2 kept), host API 12 (one appended calibration field), a less
overfit choice of the effective clock f_ref, and a full recalibration (NOTES-B2 §11). No
semantic, layout, launch, pack-format or TLA+ change; changed text is marked **[v1.17]**.
v1.18 (§6c.8) extends the performance model after two offline analyses: a cache term with a
per-call REPEAT saving and a clean-L2 state (`IMD_PERF_CACHE_CLEAN`), the PACK group working
set with K4's W planes, and a power model with busy-GPU power, DRAM energy and the boost clock
(γ = 2.5 as a prior); every key optional (absent = 0 predicts exactly as before; calibration
format 2 kept), host API 13 (seven appended calibration fields, one cache state, the reported
clock up to the boost clock), the sustained-DIRECT bench variant fix and a full recalibration
(NOTES-B2 §12). No semantic, layout, launch, pack-format or TLA+ change; changed text is
marked **[v1.18]**.
v1.19 (§6g, stage 0 of docs/NOTES-C1.md, ROADMAP C1(a)) adds adaptive precision as
**precision phases** on host API 13: a plan and a pack per N, switched between calls. It
publishes the footprint guarantees a caller needs to size one workspace and one packed buffer
for every phase (§6a F1–F3: workspace and pack bytes nondecreasing in N and g, `imd_pack`'s
write extent), the rule for re-packing one buffer at another N (§6b.7), certificates of the
internal layout facts (I1–I3, not ABI), and a host-side Python guard (`PackedBuffer`). Its
stage 1 (§6g.10–§6g.12, ROADMAP C1(b)) chooses N per phase on the host: the Python
`PrecisionPlan` composes the §6f queries per phase (per-role pack cost, pack-mode masks and a
BUDGET status, share groups ranked jointly, an optional DIRECT level), and `rho_from_output`
gives the rigorous ρ of an ELEMENTWISE target from a previous output (`cert_rho_recipe`). Its
stage 3 (NOTES-C1 §4.4, C2) adds the K-segmentation caller recipe to §3.8 with its error bound
(PROOFS L7, `cert_kseg_composition`, `oracle.contract_segments`) and rewrites the §7 rows that
the C1/C2 study decided. That study's CPU evidence (NOTES-C1 §4.3) decided the stage-4 and
stage-6 gates: caller-side K-equilibration (stage 4) is closed and library-internal weights
(stage 6) deferred. Device statistics (stage 5) were deferred earlier, on read-only consumer
evidence (NOTES-C1 IR26). Each has its trigger in §7, and the convergence study's caller
guidance is in §6g.12. The performance model's small-N hold-out (M-c) failed, so levels
outside N ∈ {8, 10, 14, 16} stay [M, model-only, outside hold-out] (§6g.9; the one MC-3a attempt
of 2026-10-09 declared no pass, §6c.9), and the GPU timing of the schedules (M-d2) ran on 2026-10-09 and passes (§6g.9, NOTES-C1 §5.12). No semantic, launch, pack-format, API,
calibration-format or TLA+ change; changed text is marked **[v1.19]**. The one ABI-adjacent
point: `imd_perf_calib_t` keeps its layout and size, but `param[95]` (the last slot, unused
before) is now the optional key `warm.kl_drop` (§6c.9), so a hand-built `imd_perf_calib_t` must
zero `param`.
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
| `K_c'` | **[v1.13]** `131008 = ⌊K_c/64⌋·64 = 2047·64` is the chunk length used (it was `131056 = ⌊K_c/16⌋·16`). Numerically any chunk length ≤ K_c is exact. A multiple of 16 keeps CUTLASS pointers 16-byte aligned (correctness). A multiple of 64, K4's threadblock K tile, keeps every full chunk free of a residue tile, so a row's K4 load phase depends on k̂ alone (§4b, performance). 131008 is the largest such multiple ≤ K_c **[cert: cert_kc, cert_row_phase; proof: PROOFS R6 `chunk_constants`]**. |
| `L_w` | Width of the **unsigned** CRT word, `32·L32` with `L32` minimal such that `2P ≤ 2^{L_w}`. `L32` = 1 for N ≤ 3, 2 for N ≤ 7, 3 for N ≤ 12, 4 for N ≤ 16, 5 for N ≤ 20. The bound is exact: the reachable peak needs every bit **[cert: cert_limbs, cert_crt_peak; proof: PROOFS R6 `L32_exact`, `peak_needs_every_limb`]**. |

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
   - Terms are < 2⁶³. S is exact, order-independent, and fits uint128 for `k < 2⁶⁶`
     **[proof: PROOFS R3]**.
   - `‖x‖²·2^F ≤ S < ‖x‖²·2^F + k`.
4. **Shift.** `σ = u − γ`, where `u` is the **largest integer** satisfying
   - RN: `4·S·4^u ≤ R·2^F`, with `R = (2T − κ)²`;
   - TRUNC: `S·4^u ≤ R·2^F`, with `R = T²`.

   Here `κ = ⌈√k⌉` is computed exactly in integers (`c = isqrt(k); κ = c + (c² < k)`).
   Using `⌊√k⌋` is **unsound** **[spec: OzakiScaling neg_kappa_floor; cert; proof: PROOFS R1]**.

   Computing u:
   - R is a per-plan constant of at most 157 bits.
   - Estimate `c = (bitlen(R·2^F) − bitlen(LhsCoef·S) + 1) >> 1`. This must be an
     **arithmetic (floor) shift**, because the difference can be negative at real widths.
   - Then do one exact multi-word compare (about 220 bits at N = 20): `u = Rule(c) ? c : c−1`.
   - **[spec: OzakiScaling ShiftIsMaximal, UOf; cert; proof: PROOFS R2]**
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
  p = 256) **[proof: PROOFS R4]**.
- It is computed exactly as `SMod((μ mod p_l)·(2^ε mod p_l), p_l)` **[proof: PROOFS R4]**.
  - `mod` here is the **non-negative** mathematical mod. C `%` on a negative μ is wrong.
  - It uses a `2^ε mod p_l` table.
- Any exact reduction is allowed (Barrett, dp4a byte weights); only the congruence class
  matters.
- The int8 store is correct only because `p_l ≤ 256` **[spec: OzakiCRT neg_resbits_p9,
  neg_resbits_p9_store]**.
- **[v1.14]** The packs (K2/K3, KF, KL, `imd_pack`) compute the same class without the
  `2^ε mod p_l` table: `SMod(Σ_i b_i·(2^(8i) mod p_l) mod p_l, p_l)` over the bytes `b_i`
  of `|a'|`, then the sign (§3.4.1).

### 3.4.1 [v1.14] The pack emitter's residue: |a'| as W words (branch-free ε) **[cert: cert_pack_words]**

The pack emitter (`PackEmitterT<E, W>`, src/kernels/pack_emitter.cuh) converts E ∈ {16, 8, 4}
consecutive k̄ of one row and stores, per modulus, their E int8 residues. It is the only
residue code of K2/K3 (`k23_pack`), KF (`kf_scale_pack`), KL (`kl_scale_pack`) and so of
`imd_pack` (KL). Up to v1.13 it reduced μ by byte weights and then, when any of the thread's
E elements had ε > 0, multiplied by `pow2_mod[l][ε]` and reduced again. That made a warp
diverge whenever one of its elements had ε > 0, and the table load was serialized across
the lanes' distinct ε. The cost depended on the data (§6c "The eps path").

**Rule.** The emitter holds `X = |a'| = μ·2^ε` exactly, as `W(N)` little-endian u32 words:
- `W(N) = 2` for N ≤ 16 and `W(N) = 3` for N = 17–20. **[cert: cert_pack_words]**
  `T(N) < 2^(32·W(N))` for every N: `⌊log₂T⌋ + 1` = 63 bits at N = 16 and 78 at N = 20
  (`kPackXBits2 = 63`, `kPackXBits3 = 78`, `kPackW2MaxN = 16`, generated into constants.hpp).
  W = 1 would do for N ≤ 8 (T < 2³²; N = 8 is the largest such N, T(8) has 32 bits and T(9)
  36, **[cert: cert_pack_words]**). **[v1.16]** It was implemented (`PackEmitterT<E, 1>`:
  X = u32(μ) << ε, one `dp4a`; a third instantiation of every pack kernel), measured and
  **not adopted**: pack phases −2.7 % (geometric mean of 26 per-phase medians at N = 4–8,
  three interleaved A/B rounds; −0.4 to −1.4 % on the KL phases of 4096³ and the DF
  tensor, −1.5 to −3.5 % on 1024² × 16384, −4 to −7 % on `imd_pack`, −4 to −10 % on KF
  at 1024³ with ±10 % process noise), calls −1.3 % (no regression on any stable shape),
  against +11 % contract.cu compile time (16.3 → 18.1 s) and +11 % `libimmerdink.so`
  (6.58 → 7.33 MB). The bar was a ≥ 3 % pack-phase gain; NOTES-BFEPS §4 has the data.
- **Precondition `|a'| ≤ T(N)`**, for RN and TRUNC: each entry is bounded by the row's
  norm, `|a'_h| ≤ ‖a'‖₂ ≤ T` (§3.2 guarantee, **[cert: cert_shift_rule_guarantee; spec:
  OzakiScaling]**). cert_pack_words also checks it directly with the §3.2 rule (`shift_u`,
  the product's `largest_u`) on the rows that maximize ε: one nonzero entry, subnormal
  entries down to 2⁻¹⁰⁷⁴ (μ = 1, ε up to `eps_max(N)`), dominant entries and random rows,
  for both roundings and every N. So `bitlen(μ) + ε ≤ kPackXBits[W]` and no bit of X is
  lost. A debug build asserts this on the device (no NDEBUG; the unit test
  `pack_emitter` always runs with it).
- **Construction** (once per element, in `load`, after the unchanged §3.3 `quantize`):
  W = 2: `X = μ << ε` as one u64.
  **[v1.16] Defined shift.** The W = 2 shift is PTX `shl.b64` (inline asm, `shl_clamped`),
  whose count is clamped at 64 (result 0), instead of the C++ shift, which is undefined for
  a count ≥ 64. In the contract (|a'| ≤ T, the precondition above) nothing changes; a
  release build of a hypothetical σ bug that produced |a'| ≥ 2⁶⁴ stores wrong bytes
  instead of executing undefined behaviour. The debug assert stays. Masks were measured
  first and rejected: `ε & 63` adds 8–40 SASS instructions per W = 2 kernel,
  `ε < 64 ? μ << ε : 0` up to 112. With PTX shl the W = 2 kernels keep their registers and
  spills and their instruction counts within −2 % / +0.2 % (not byte-identical: the C++
  shift let the compiler assume ε < 64); the W = 3 kernels are byte-identical; C is
  bit-identical and the timing neutral (the v1.16 changelog).

  W = 3: the two words of μ shifted by `b = ε mod 32` with
  `__funnelshift_l`, then placed by `q = ⌊ε/32⌋ ∈ {0, 1, 2}` with selects (static indices
  only, no local memory; b ≤ 31 by construction and q ≥ 3 selects as q = 2, so W = 3 has
  no undefined shift). Zero, ±0 and non-finite elements give X = 0 (as μ = 0 before).
  For N ≤ 13 normal inputs always have ε = 0 (T < 2⁵¹ < 2⁵³ ≤ the smallest μ·2^ε with
  ε > 0 and a 53-bit μ), so X = μ; there is one emitter for every N, no N ≤ 13 special case.
- **Residue** of modulus l: `s = Σ_{j<W} dp4a(X_j, w_j(l))`, where the u8 lanes of the
  weight word `w_j(l)` are `2^(8(4j+i)) mod p_l`, i = 0..3, for the bytes 0..9 of X
  (bytes 10 and 11, the top lanes of `w_2`, have weight 0; X < 2⁷⁸ keeps them zero anyway).
  Then `s ≡ X (mod p_l)` and `s ≤ 10·255·255 < 12·255² < 2²⁰`.
  **[v1.16] Note:** 10·255·255 = 650,250 is a loose bound (it takes every weight as 255).
  The exact maxima, certified by cert_pack_words over the moduli each W serves, are:
  210,472 (W = 2, X < 2⁶³) and 264,156 (W = 3, X < 2⁷⁸, p = 253).
  With every one of the ten bytes at 255 the maximum would be 302,940 (p = 253); over
  X ≤ T(N) it is 258,500 (N = 20, p = 253). All of them are below 2¹⁹; only s < 2²⁴ is
  needed.
  `r = s − p_l·umulhi(s, ⌈2³²/p_l⌉)` is the non-negative `s mod p_l` for every s < 2²⁴
  (the A5 `mod_p_bytes32` lemma: s·e < 2³² for e = ⌈2³²/p⌉·p − 2³² < p ≤ 256;
  `mod_p_words32<W>`, device/modp.hpp; **[proof: PROOFS R5 `mod_p_bytes32`]**). For p = 256
  the weights are 1, 0, 0, …, so s is the low byte and the quotient is 0.
- **Sign and SMod** as before (A5): `SMod(r)`, negated for a' < 0 (SMod(−r mod p) =
  −SMod(r) for odd p; both give the byte (−r) mod 256 for p = 256).
- **The weights** (`PackWeights`, src/device/residue.hpp, `__constant__ c_pack_weights`) are
  derived at compile time from the generated `pow2_mod` table (e = 0..72 ≤ kEpsMax = 77,
  `static_assert(kEpsMax >= 72)`): no new generated table, about 12 bytes per modulus.
- **Bytes.** X and μ·2^ε are the same integer and `s ≡ X`, so `r` is `|a'| mod p_l`, the
  value the v1.13 emitter computed: every stored int8 is unchanged. No pack-format,
  layout, launch-shape or API change.

**W per plan: a template parameter.** Every pack kernel is instantiated for W = 2 and
W = 3, and the launch picks the instantiation from the plan's N (`pack_words(N)`). A
warp-uniform runtime branch on W was the alternative: the kernel's register allocation would
then be W = 3's at every N (the exploration measured W = 3 everywhere 5–8 % slower at
N ≤ 16). Cost of the templates, measured (Release, sm_89, one compile job): contract.cu
16.8 → 18.5 s (+10 %), `libimmerdink.so` 5.83 → 6.60 MB (+13 %). Accepted.
The occupancy query for the performance model (`kernel_occupancy`, §6c) reports the
W = 3 instantiation of each pack kernel (the model is unchanged in v1.14; §6c).
**[v1.16] W-aware occupancy.** `kernel_occupancy_w(kernel, W, out)` (src/plan.hpp) returns
one pack kernel's (KF16, KF4, KL8, KL4, pack) occupancy row for one W ∈ {2, 3} (W = 0:
the row `kernel_occupancy` reports; the other kernels have no W and take only W = 0).
`kernel_occupancy` keeps one row per kernel and reports the **most constrained**
instantiation: fewest CTAs per SM, then the most registers (on sm_89 that is W = 3: same
CTAs per SM for both W, most registers: the v1.14 row, so the shipped calibration is
unchanged). The model keeps one occupancy row per kernel (calibration format unchanged):
its features use the CTAs per SM, which are equal for both W (K2/K3: 40,768 B static
shared memory, 2 CTAs, at 84 / 117 registers; KF and KL at their launch-bound register
caps), so the one row is exact at every N. The GPU test (`contract_api`, through
the test hook `imd_test_kernel_occupancy_w`) asserts that equality for every pack kernel
and that the reported row is the most constrained one; if a later kernel change makes the
CTAs per SM differ by W, it fails, and the model then needs per-W rows. Those are already
measured: `imd_bench --methods occupancy` writes `kernels_w` (per pack kernel, the rows of
W = 2, 3; benchmark hook `imd_prof_kernel_occupancy_w`) next to `kernels`.

**Per-residue cost.** W `dp4a`, one `umulhi`, one multiply-subtract, SMod and the sign: the
same instruction stream for every element and every data. No ε registers (16 fewer live
registers at E = 16, W = 2), no `has_eps` test per modulus, no table load indexed by a
per-lane ε. Measured in the exploration (eval findings followup-branchfree, standalone
reproduction of the emitter): 2.5–3× fewer ps per residue on the old ε path, 12 % fewer at
N = 13; the product A/B is in the v1.14 changelog entry.

**Verification.**
- `spec/params.py` **cert_pack_words**: the W rule and `T(N) < 2^(32W)`; `|a'| ≤ T` and
  `bitlen(μ) + ε ≤ kPackXBits[W]` on the ε-maximizing rows above, RN and TRUNC, every N
  (with the largest ε reached reported per N); the weight bytes `2^(8i) mod p` for i ≤ 9
  within `pow2_mod`; `s_max < 2²⁰ ≤ 2²⁴`; the exact `umulhi` quotient for every s < 2²⁴ of
  every modulus (the e·2²⁴ < 2³² bound) and `Σ b_i w_i ≡ X` on random X of every bit length.
- `tests/test_mod_exhaustive.cpp` (CPU, the kernels' own `modp.hpp` code): the reduction
  over the whole range [0, 12·255²] for every modulus; `mod_p_words32<2>` / `<3>` against
  `X mod p` on edge values and 2²⁰ random X < 2⁶³ / 2⁷⁸ per modulus; the weight table.
- `tests/test_pack_emitter.cu` (GPU, `pack_emitter`): the emitter itself for E = 16, 8, 4
  and W = W(N), every N, every ε in 0..eps_max(N) with μ of every bit length allowed by
  X ≤ T(N) (both word selects q = 1, 2 at N = 17–20; every shift up to 62 at N = 16),
  ±T, T − 1, 2⁵³ ± 1, 2⁶³ − 1 (N = 16), 2⁶⁴ (N = 17), mixed ε in one warp, RN and TRUNC
  rounding paths, negative values, zeros and non-finite entries, against an exact host
  reference (u128 `|a'| mod p`, then SMod and the sign); debug asserts on.
- `tests/test_contract.py` suite `eps` (GPU, bit-exact vs spec/oracle.py on every launch
  path of `run_case`: KF E = 16 / E = 4, K2/K3 with moduli parts, KL E = 8 / E = 4): rows of
  subnormals scaled to ε ∈ [32, 77] at N = 17–20 and up to 62 at N = 16, ε > 0 at N ≤ 13
  (subnormals; **[v1.16]** N = 2, 5, 6, 7 added, ε up to eps_max(N) − 3 at N = 5–8),
  dominant entries, TRUNC; and `test_packed.py` cases of the same data
  (`imd_pack`, every byte). compute-sanitizer memcheck on a few of them.
- No TLA+ change: single-thread arithmetic on one element; no protocol, layout, lowering
  or pack-format change.

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
- **Realization [cert: cert_barrett; proof: PROOFS R5 `barrett_epilogue`]:**
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
  word → limbs); cert: cert_crt_peak, cert_crt_roundtrip; proof: PROOFS L2, R6]**
- The word is **unsigned**. A signed `__int128` would need `2P ≤ 2^{L_w−1}`, and signed
  overflow is UB.
- In the centering step, compare the sign bit with the borrow-out
  **[spec: CRTRealization pos_wordref_nosub; proof: PROOFS R6 `center_word`]**.
- "No division" means no wide division. `x + M_l y = P` can never occur exactly, so `≥` and
  `>` behave the same **[proof: PROOFS R6 `xp_ne_P`]**.
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

#### [v1.19] Segmented accumulation: the K-segmentation recipe **[proof: PROOFS L7; cert: cert_kseg_composition; oracle: contract_segments, test (13)]**

A **caller recipe with no API** (docs/NOTES-C1.md §2.7, §4.4; Q15 (a), confirmed by M-a,
§5.6 S2-13). It splits K into S bands and runs each band as its own call. On data whose
magnitudes along K are anti-correlated **in contiguous bands** (decay_opp-like), each band's
norm product ‖a^s‖‖b^s‖ is far below ‖a‖‖b‖, and fewer moduli reach a target. It costs S calls,
and it gives up §3.7's single final rounding.

**Recipe.**
1. **Bands.** Partition K into S bands, e.g. `oracle.segment_bounds(k, S)` (near-equal and
   contiguous). For a tensor, band s is a sub-view along one contracted label that A and B
   share: the same descriptors with that label's extent cut to the band, and both base
   pointers advanced by start × that label's stride in each operand. L7 holds for any
   partition; the gain needs bands that follow the magnitude profile. Cutting the outermost
   (slowest-varying) label of the canonical K order gives bands contiguous in that order;
   cutting another label gives a non-contiguous partition of it (step 3 says what the
   reference is then).
2. **Calls.** Band 1 is an ordinary call with the caller's α and β. Bands 2 … S follow in a
   fixed order on the same stream, with the same α and **β = 1**, so
   `C_s = fma(1, C_{s−1}, E_s) = RN(C_{s−1} + E_s)` (§3.7). Bands of equal k_s share one plan
   (only the base pointers move); unequal ones need one plan per distinct k_s. Every band is
   an ordinary call:
   - its own σ and τ, and its own non-finite scan (a row with an Inf or NaN takes the native
     path only in the band that holds it);
   - its own RN acceptance κ_s < T(N), so a k too long for RN at small N can run in bands;
   - its own N if the caller wants (L7 takes E_s per band);
   - packs per band: one pack per band sub-view, keyed by its k_s (§6b);
   - its own dispatch (§4): a band of k_s = 1 is K-empty and runs DIRECT, one fma with
     `|D_s − x_s| ≤ u|x_s|`, so L7 holds for it with E_s = 0.
3. **Result.** Deterministic for fixed bands, order and N. `oracle.contract_segments(A, B, C,
   N, rounding, bounds, alpha, beta)` is the literal composition of the band calls, each
   dispatched as the library dispatches it (`contract`, or DIRECT for k_s = 1), and the
   bit-exact reference. spec/test_oracle.py (13) checks it against the same calls through
   `contract_tensor` on moved sub-views and against an independent composition. It is an
   EMULATE recipe, with bands contiguous in canonical K order. For a non-contiguous
   partition (step 1) the reference is `contract_segments` on the GEMM view with its K
   positions permuted so that each band is contiguous, in its sub-view's canonical order.

**Error** (PROOFS L7; α = 1, β = 0; finite inputs, no overflow; for NORMWISE and ELEMENTWISE
also **every band's product D_s normal or zero**). Per output (i, j), with
`n_s = ‖a_i^s‖‖b_j^s‖`, E_s the §6f.2 E of band s (`imd_err_estimate`'s `unit_bound` at
(N, k_s, rounding) with **the band's** ℓ1 ratios, or 0 = √k_s; 0 for a K-empty band) and
`γ = (S−1)u/(1 − (S−1)u)`:
```
|C_S − x| ≤ Σ_s n_s·W_s,      W_s = u + (1+u)·E_s + γ·(1+u)·(1+E_s)
```
- `N_S = Σ_s n_s ≤ ‖a_i‖‖b_j‖` (Cauchy–Schwarz over the bands). With `W̄ = max_s W_s`:
  NORMWISE `≤ W̄·‖a_i‖‖b_j‖`; ELEMENTWISE `≤ W̄·ρ_S·|x|` for `ρ_S ≥ N_S/|x|` (ρ_S ≤ ρ);
  ABSOLUTE `≤ W̄·M̄ + (1+γ)·S·2⁻¹⁰⁷⁵` for `M̄ ≥ N_S` (the last term: a band's product that
  underflows gradually; the additions themselves are exact in the subnormal range).
- **Underflow is a new hazard**: a short band's product can be subnormal while x is not, and
  the caller does not see D_s. Only the ABSOLUTE form covers it, with its (1+γ)·S·2⁻¹⁰⁷⁵
  term; NORMWISE and ELEMENTWISE need every D_s normal or zero (PROOFS L7's hypothesis).
  A caller cannot check that hypothesis: D_s for s ≥ 2 is never materialized, and a normal
  C_S does not imply it. Without it every form holds with the same absolute term added,
  `(1+γ)·S·2⁻¹⁰⁷⁵ ≤ S·2⁻¹⁰⁷⁴` (`params.kseg_eta(S)`, a double; PROOFS L7c (3)), which
  matters only for outputs within a few ulps of the subnormal range. The rigorous ρ_S below
  carries it, so it needs no hypothesis.
- **Against §6f.2 with ν = √k** (ℓ1 ratios 0) at the same N: `E_s ≤ E(k)`, so the NORMWISE and
  ABSOLUTE bounds exceed the unsegmented ones by `γ(1+u)(1+E)·‖a‖‖b‖` only, about (S−1)u.
- **A whole-row ℓ1 ratio does not bound a band's**: a row (1, ε, …, ε) has a ratio near 1
  while a band without its first entry has √k_s. Use 0 (√k_s) or ratios measured per band;
  the certificate shows the bound with the whole row's ratio failing on such a band.
- **ELEMENTWISE is worse at high N**: W̄ρ_S carries `(u + γ(1+u))·ρ_S` where §6f.2 has u.
  At N = 20 an S = 2 output errs by 2.9 u|x|, beyond the bound the unsegmented call meets
  (`cert_kseg_composition`).
- α ≠ 1 or β ≠ 0: as §6f.2 (one more rounding for α; β's first fma is not expressed).

**Choosing N (rigorous; L7d).** For α = 1 (or α = 2^e with no RN(α·D_s) or C_s underflowing
or overflowing, which scales the bound exactly) and β = 0: another α adds one rounding per
band that L7d's margin does not cover. The gain is `N_S < ‖a‖‖b‖`, which the plain statistics
do not see. With the per-band norms (one pass over A and B, as §6g.11's norms):
1. Form the exact quotient t̄ and `t_eff = RN(t̄)` (one RN division):
   - NORMWISE target t relative to the plain norms: `t̄ = t/r` with
     `r ≥ max_ij N_S,ij/(‖a_i‖‖b_j‖)`, rounded upward. r = 1 is always valid (N_S ≤ ‖a‖‖b‖)
     and gains nothing; a cheap valid r is
     `min(1, Σ_s max_i(‖a_i^s‖/‖a_i‖)·max_j(‖b_j^s‖/‖b_j‖))` (the sum can exceed 1), the
     maxima over the rows and columns with ‖a_i‖ > 0 and ‖b_j‖ > 0 (an output with a zero
     row or column has N_S = 0 and x = 0, so it needs no ratio). It is an upper bound only
     if every band norm ‖a_i^s‖, ‖b_j^s‖ in it is an upper bound, every whole-row norm
     ‖a_i‖, ‖b_j‖ a LOWER bound, and each quotient, product and sum is rounded upward.
     Norms computed in RN carry relative errors of about k·u, which L7d's margin (about
     5u·t̄) does not cover: enclose them (square sums with directed rounding, or the RN
     result widened by an explicit relative bound, outward in each direction: γ_{k+1} for
     the square root of an RN sum of k squares, absent underflow and overflow).
   - ELEMENTWISE: `t̄ = t/ρ_S` with `ρ_S ≥ N_S/|x|` over the outputs that must meet the
     target, rounded upward. A rigorous ρ_S from a previous segmented output C_S (the same
     bands, α = 1, β = 0; no hypothesis on the D_s):
     `|x| ≥ |C_S| − Σ_s n_s W_s − (1+γ)·S·2⁻¹⁰⁷⁵ ≥ |C_S| − W̄·N_S − (1+γ)·S·2⁻¹⁰⁷⁵`, so
     `ρ_S ≤ N_S/(|C_S| − W̄·N_S − (1+γ)·S·2⁻¹⁰⁷⁵)` where the denominator is positive
     (PROOFS L7c (3), (7)).
     Evaluate it as §6g.11 does, every step rounded outward, with n ≥ N_S (the band norm
     products summed rounding upward) and U ≥ W̄ from `params.kseg_wbar_up(Ū, S)`
     (`Ū + S·u·(1 + Ū)` rounded upward, Ū ≥ every band's NORMWISE `bound`, γ_1 for a
     K-empty band): ρ̂_S = `params.rho_recipe(C_S, n, 1, U, eta=params.kseg_eta(S))`, the
     absolute term added to U·n rounding upward, certified only for a finite, normal C_S
     with a positive denominator (`cert_kseg_composition` (7), also on outputs whose last
     band's products underflow). Or take
     `rho_from_output`'s plain ρ̂ from an unsegmented β = 0 probe: valid since ρ_S ≤ ρ, but it
     gives up the gain. Either way, as in §6g.11, a ρ_S taken from a previous output predicts
     the next call's value; it does not bound it.
   - ABSOLUTE: `t̄ = t/M̄` with `M̄ ≥ N_S`, rounded upward.
   Each guarantee below holds up to the absolute term above, which only an underflowing
   D_s brings.
2. `t′ = RN(RN(min(t_eff, 1)·(1 − 2⁻⁵⁰)) − 4S·2⁻⁵³)` in FP64 (`params.kseg_request`;
   S ≤ 2²⁶). If `t′ ≤ 0`, the transform certifies no N and `params.kseg_request` returns
   None, never 0.0: a target of 0 means "no target, report only" to rank (§6f.1), which
   would return the cheapest configuration with no accuracy filter. Since W ≥ S·u, L7
   certifies no t̄ < S·u at any N, and between S·u and about 4S·u the transform is merely
   conservative: normwise targets near 1e−15 are out of reach (t_eff = 5e−16 or 8e−16 at
   S = 2, 1e−15 at S = 4).
3. Request NORMWISE target t′ > 0, basis BOUND (TYPICAL voids the certification), with ℓ1
   ratios 0: `imd_err_min_moduli` / rank on the whole contraction (E(k) bounds every E_s) or
   on each band's sub-view. Its N meets the target: `E(1+u) + u ≤ t′` implies `W ≤ t̄`, the
   exact quotient (L7d; t_eff = RN(t̄) can lie above t̄).

**When it pays** (M-a, the ≥ 2048³ class; immerdink-eval `findings/c1-evidence.md`).
- On banded anti-correlated synthetic profiles S = 2 saves 2–4 moduli at equal sampled max
  elementwise error (decay_opp 3–4, opp_nonsep 2–3; −0.5 to +1 on the other families). For
  example decay_opp 2048 × 2048 × 3600 at 1e−10 elementwise needs N = 16 instead of 20:
  18.8 against 17.4 TFLOPS-eq (1.61 against 1.73 ms) [M]. K-equilibration needs 1–4 moduli
  fewer than S = 2 on every such case, but its library form is closed (§7).
- On the real psi4 operands S = 2 saves at most one modulus (normwise 1e−15), where its time
  is still +4 % to +62 % over fast mode's [M] (R2b: 2 % faster, inside the model's error),
  and it loses elementwise accuracy at 1e−10 / 1e−12 on the outputs with ρ ≤ 1e8.
- Costs per call: S-fold K0, K5 and W-plane writes plus the β = 1 read of C ([E] +10–93 % at
  S = 2–4 on 4096³, DF and 8192³; NOTES-C1 §2.7). The model's own composition of the S calls,
  the method of the [M] figures above, prices them lower: +8–72 % at N = 14 (NOTES-C1 §2.7,
  M-b). That [M] composition is the reference for decisions; the [E] stays as a cross-check
  until a GPU measurement of the recipe replaces both (NOTES-C1 §5.8 MB-1).
- [M] figures at N ∉ {8, 10, 14, 16} are model-only until the small-N hold-out M-c passes
  (NOTES-C1 IR4). M-c ran on 2026-10-07 and failed (one target missed; NOTES-C1 Status,
  "M-c", and §5.9), so the marks stay; the MC-3 feature path (NOTES-C1 §5.9 MC-3, MC-3a) was
  run once on 2026-10-09 and did not pass (§6c.9, NOTES-C1 §5.11, independent review, ruling A,
  binding): the marks stay. NOTES-C1's M-b re-ran the [M] figures here on this worktree's
  library:
  unchanged (immerdink-eval `findings/c1-m-b.md`).

Use it for banded anti-correlated K profiles at normwise or absolute targets well above S·u,
not for elementwise targets near u.

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
  `k̂ = ⌈k⌉₁₆` and `n̂ = ⌈n⌉₁₆`. **[stage 2]** `k̂ = ⌈k⌉_A`, A = 64 for k > 96 (§4b: K4's
  64-byte load phase); 16-byte alignment stays the correctness rule.
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
[stage 2, §4b] k̂ = ⌈k⌉_A,  A = 64 if k > 96 else 16 (k̂ = ⌈k⌉₁₆ for every k ≤ 96)

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

## 4b. K4 load alignment [v1.13] (NOTES-K4ALIGN, ROADMAP A6)

**Status.** Stage 0 (one `k_hat_of`) and stage 1 (`K_c' = 131008`) are implemented. Stage 2
(per-plan K alignment `A ∈ {16, 64}`, resolved from k alone) is **implemented** (2026-10-06,
branch `k4-align-s2`) as designed here (the rulings of the independent review of
2026-10-05 and the must-fixes of the adversarial review of the design, same day, are
binding; the threshold is frozen from the 2026-10-05 sweep). `spec/params.py` and
`spec/oracle.py` run with `K_ALIGN_POLICY = "AUTO"`; their A16 paths stay exercised by the
certificates and the oracle suite. Text marked **[stage 2]** elsewhere in this document is
the implemented behaviour; the implementation notes (the one rule fixed for two packs, the
places where the code had a choice) are in the v1.13 stage 2 change list. The θ refit and
the `k4_select` re-check (§4b.6) were done on 2026-10-06: θ refit, rule unchanged (NOTES-B2
§8, NOTES-K4ALIGN §10).

### 4b.1 Mechanism (measured; CUTLASS `predicated_tile_access_iterator.h:204-213, 514-523`)

- K4 (CUTLASS 2.x, int8 A row-major and B column-major, alignment 16, threadblock K tile 64)
  reads RA/RB rows with row stride `lda = ldb = k̂` (the region and pack bases are
  256-aligned).
- A launch covers the K-chunk `[k0, k0 + kc)`. It loads the chunk's residue tile
  (`R = ((kc − 1) mod 64) + 1` elements) **first**, at `row + k0`, then full tiles at
  `row + k0 + R + 64·i`. Every main-loop load of row r therefore starts at phase

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
- **Class 32 is a 128-byte line crossing (measured, ncu, 64×64, 2048², N = 8):** at
  k̂ = 1056 one 64-byte segment in four starts at offset 96 of a 128-byte line, which
  predicts +25 % L2 requests at an unchanged sector count; measured +27 % requests, sectors
  1.000× per k, tensor pipe 46 → 38.5 %, time per k 1.22×. Class 16 (k̂ = 1040) reproduces
  the 128×256 pattern (1.91× requests, 1.51× sectors per k).
- Two facts are measured but **not** explained by the phase rule:
  - class 48 is 5–15 % slower than class 16, although both have the same set of phases;
  - the sector multipliers are 1.5× with half of the rows misaligned and 2.0× with all of
    them (a 64-byte segment touching 3 sectors instead of 2 predicts at most 1.5×).

### 4b.2 Stage 1 (implemented): `K_c' = ⌊K_c/64⌋·64 = 131008` **[cert: cert_kc, cert_row_phase]**

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
  - `k_chunks = ⌈k̂/131008⌉` gains one for `k̂ ∈ (j·131008, j·131056]` (exactly one while
    these windows are disjoint, j ≤ 2729, k̂ ≲ 3.58·10⁸; above, the difference can exceed one);
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
- Stage 0 (done with stage 1) is a refactor with no behaviour change: `k̂ = ⌈k⌉_A` has one
  definition, `k_hat_of(k, A)`, in libimmerdink_host (`src/host/k_hat.hpp`), mirrored by
  `spec/params.py k_hat_of`.

### 4b.3 Stage 2: the rule **[stage 2]** **[cert: cert_k_align; spec: K4Align, Pipeline RowPhaseAligned, EveryRowByteWritten; PackedOperand]**

```
A(k)  = 64  if k > K_PAD64_MIN = 96      (AUTO; k = the logical contraction length after
        16  otherwise                     §5 lowering, the k of imd_lowering_t)
k̂     = ⌈k⌉_A                            row stride of RA, RB and the packed residues,
                                          AND the GEMM K extent K4 runs (zero tail)
K_c'  = 131008                            (stage 1)
```
- **A is a pure function of k.** It does not depend on N, g, the shape, the device, the
  data, the calibration or any option. A plan and a pack of the same k always resolve the
  same A, so a packed plan and its pack never disagree in-process. The guarantee is that k
  is a key word (a pack and a slot of different k never match) and that A is resolved once
  per API call and then passed explicitly (§4b.5 "One resolver"); PackedOperand
  `AutoFormatsAgree` is only a sanity check of that (the model applies the same rule to
  the same k by construction).
  `K_PAD64_MIN` is defined in `spec/params.py` and generated into `constants.hpp`
  (`kKPad64Min`, with `kKAlignPad = 64`).
- **Why it fixes the loads.** With `k̂ ≡ 0 (mod 64)` every row starts on a multiple of 64
  (bases are 256-aligned; every row start is base + an integer times k̂, including the z0
  batch offset of a gridDim.z part, the j0 row offset of a gridDim.y part, the plane offset
  `g₀·b·r·k̂` of a residue group in a pack and the B role's rows past r); every full chunk
  starts and ends on a multiple of 64 (stage 1); the last chunk ends at k̂. So every load in
  CUTLASS's residue-first order is 64-aligned, residue tile included. K4Align checks this on
  the explicit address sequence; `cert_k_align` at the real values (k up to 524033, bases,
  z0 and j0 offsets).
- **What does not change** (`cert_k_align`): the launch structure. K_c', KF's and KL's tile
  lengths (powers of two ≥ 64) and the K4 tile are multiples of 64, so `⌈x/t⌉` is the same
  for x = ⌈k⌉₁₆ and ⌈k⌉₆₄: `k_chunks`, the K4 main-loop iterations per chunk, the KF / KL
  decisions (`k̂ ≤ 1024 ⇔ k ≤ 1024`), KL's tile and chunk count `nc`, and (k > 64) KF's tile
  `tk = pow2 ≥ k̂`. The K4 MMA work is the same too: CUTLASS computes a chunk's residue tile
  as a full 64-wide zero-filled tile either way. A changes only **bytes**: per residue row,
  `δ = ⌈k⌉₆₄ − ⌈k⌉₁₆ ∈ {0, 16, 32, 48}` more bytes are written by the producers and read by
  K4. Relative to ⌈k⌉₁₆ that is at most **1/3** (k = 129…144, ⌈k⌉₁₆ = 144 → 192), at most
  48/1040 = 4.6 % for k > 1024, and 0 for ⌈k⌉₁₆ ≡ 0 mod 64. (The threshold note's "43 % at
  k = 97–112" is an arithmetic slip: k = 97…112 pads 112 → 128, +14 %.)
- **Two plan-level differences that are not launch counts** (review 2026-10-05, must-fix 8;
  `cert_k_align`, `params.k4_add_chunk_keys` / `k4_tune_bytes`):
  - *Add-mod shape keys.* `k4_counts` (`k4_launches.hpp`) merges the last chunk's add key
    into the full-chunk key when the last chunk is itself full (≥ 3 chunks). With A = 64
    that happens for k̂ = ⌈k⌉₆₄ = n·K_c', n ≥ 3, while ⌈k⌉₁₆ < n·K_c' has a short last
    chunk: the A = 64 plan has **one add shape key fewer** per (z, n-part) size, so
    `add_shapes` and the plan host bytes differ (smaller). Exactly the k in
    (n·K_c' − 64, n·K_c' − 16], n ≥ 3; e.g. **k = 393008** (A = 16: chunks 131008, 131008,
    130992, two add keys; A = 64: 3 × 131008, one). k = 393009 and 262016 do not differ.
  - *The `k4_tune` cap.* `k4_tune_bytes` is 0 (no tuning) above 256 MiB
    (`kK4TuneMaxBytes`); the padded k̂ can cross it, so an A = 64 plan may not tune where
    the A = 16 plan did. Example (m, n, k, b, g) = (2000000, 1, 97, 1, 1): 256001792 B at
    A = 16, over the cap (0) at A = 64.
- **Why 96** (sweep 2026-10-05, immerdink-eval `findings/k4align-threshold.md`; k = 16…1024
  step 16, N ∈ {8, 14}, 2048², 16384×512, a DF-like `Qia,Qjb->iajb` (2304² × Q), 512²; A = 64
  emulated by running ⌈k⌉₆₄ and counting the logical 2mnk):
  - k̂ ≤ 64 has no main loop (only the residue tile), so padding is pure cost there
    (k = 16 → 64: −6 to −12 %; k = 32: −1 to −9 %; k = 48: about 0). Hence K_PAD64_MIN ≥ 64.
  - Mean call-time loss against the per-k best choice, over the 8 shape × N cases and all
    k: 96 → 0.30 %, 128 → 0.35 %, 64 → 0.36 %, 192 → 0.47 %, 512 → 1.30 %, never pad →
    3.80 %. 96 is the per-case optimum in 5 of 8 cases and within 0.03 pp in 7 of 8 (the
    exception, 16384×512 at N = 14, prefers 288 and loses 0.25 pp).
  - Against never padding, 96 makes the average call over k = 16…1024 +2.8 to +5.6 %
    faster (useful TFLOPS-eq). Example, 2048², N = 14: k = 1008 runs at 18.24 instead of
    15.70 TFLOPS-eq (0.464 instead of 0.539 ms, +16 %); k = 112: +3.4 %.
  - The worst single-k loss under 96 is 2.3–6.8 % on the large shapes (k = 144–208, classes
    16 / 32 under 128×256) and 10–12 % on 512² (k = 144 / 160, where the rule's
    configuration changes).
  - A rule that also looks at k̂ mod 64 reaches 0.26 %; not worth a second parameter.
  - **Re-checked with the real A = 64** (follow-up 2026-10-06, docs/NOTES-K4ALIGN.md §11;
    immerdink-eval `findings/followup-threshold.md`): A = 16 against A = 64 interleaved in one
    process (`imd_bench` duel `ka=`), 7 shapes × N ∈ {8, 14} × k = 65…385. **96 stays.** The
    best k-only rule on that data (a per-k oracle) is +0.065 % mean (mean loss vs the per-case
    best 0.228 %, against 0.29 % for 96), "never pad class 32"
    +0.03 to +0.05 %; padding-fraction rules (pad iff (⌈k⌉₆₄ − ⌈k⌉₁₆) ≤ f·⌈k⌉₁₆) lose 0.04
    to 1.2 % because class 16 is the most misaligned class and nets a gain; cuts 112…320
    lose 0.24 to 1.1 %. Even a per-case oracle knowing (m, n, N) gains only 0.29 %. The
    residual loss is skinny shapes (16384×512, 512×16384) at k̂ ≡ 16 / 32, −1 to −4 %, where
    the tall operand's producer bytes outweigh K4's gain. A rule in m or n is inadmissible
    anyway: a pack knows only its own free extent (r) and k, and both K4 operands share
    one k̂, so A must be resolvable from what both packs and the plan know (k, N, b).
- **Rejected:**
  - *Stride-only padding* (row stride ⌈k⌉₆₄, GEMM K extent ⌈k⌉₁₆): the residue tile of
    `extent mod 64` elements shifts every main-loop load; with the old K_c' it misaligns
    every row of every full chunk (K4Align `neg_stride_only`, `neg_partial_landing`:
    the row and residue-tile starts stay aligned, the main loop does not).
  - *A = 32 with K_c' = 131040*: leaves class 32 (+3 % on 128×256, +10–19 % on the smaller
    configurations; `neg_a32`, `neg_kc32`).
  - *A chosen by the performance model*: sizing must stay GPU-free and independent of the
    calibration (§6a, §6c); a k-only rule keeps `imd_workspace_size` a closed form.
  - *An `imd_options_t.k_align` field*: `imd_options_t` has no `struct_size`, so a trailing
    field is undefined behaviour for version-8 callers (§4b.5).
  - *A pack override forcing A = 16 for k > 96* (footprint-tight packs): it saves at most
    1/3 of the residue bytes (≤ 4.6 % for k > 1024) and would need an `imd_pack_desc_t`
    field. Not now.
  - *P2, a K4 "realign" variant* (deferred, §4b.10).

### 4b.4 Layout and footprint (exact) **[stage 2]** **[cert: cert_workspace, cert_pack_layout, cert_k_align; spec: Pipeline LayoutExact, PackLayoutExact, AccessesSafe at AlignK = 4·AlignN]**

Every formula of §4a and §6b.4 keeps its form with `k̂ = ⌈k⌉_A`, `A = A(k)`; `n̂ = ⌈n⌉₁₆`
and the pack tail's row rounding `⌈r⌉₁₆` are unchanged:
```
RA             g·b·m·k̂                     RB   g·b·n̂·k̂
pack res       N·b·r·k̂ + (⌈r⌉₁₆ − r)·k̂      (sig 2·b·r, flag b·r: unchanged)
k4_tune_bytes  align(g·b·m·k̂) + align(g·b·n̂·k̂) + align(g·b·m·n̂)
sA sB fA fB vec W: unchanged; every region still at a 256-aligned offset (later offsets
shift by the aligned difference)
```
Against A = 16, with `δ = ⌈k⌉₆₄ − ⌈k⌉₁₆`:
`ΔRA = g·b·m·δ`, `ΔRB = g·b·n̂·δ`, `Δres = (N·b·r + ⌈r⌉₁₆ − r)·δ`, `Δk4_tune = g·b·(m + n̂)·δ`
(before the 256-alignment of each region).
- **A = 16 for k ≤ 96**: every such workspace and pack is byte-for-byte the v1.13 stage-1
  one. The hand-computed literals of `tests/test_host_c.c` (k ≤ 33) are unchanged; the
  k = 180² = 32400 case (`test_host_c.c:166`, ≡ 16 mod 64) becomes k̂ = 32448: its sizes
  change, so its assertions are recomputed, not copied.
- Hand-computed examples (typed into `cert_workspace` / `cert_pack_layout`):
  - workspace (m, n, k, b, N, g) = (17, 33, 5, 3, 9, 2), A forced to 64: k̂ = 64, n̂ = 48;
    RA 6528 at 2048, RB 18432 at 8704, W 22032 at 27136; total **49408** (A = 16: 30720);
  - (1000, 3, 200, 2, 20, 7), AUTO (200 > 96): k̂ = 256; RA 3584000 at 26112, RB 57344 at
    3610112, W 640000 at 3667456; total **4307456**;
  - pack (b, r, k, N) = (1, 1, 1000, 14), AUTO: k̂ = 1024, res 14·1024 + 15·1024 = 29696 at
    512; total **30208** (format 1: 29952);
  - pack (3, 17, 5, 9) forced to format 2: res 29376 + 960 = 30336; total **30976** (AUTO:
    format 1, 8192).
- **The W / n̂ stride is excluded.** `ldw = n̂` (≡ 16 mod 32 possible) in the K4 epilogue's
  read-modify-write and in K5 is not changed or measured here. It is a later item of its
  own; it would need its own pack-format value (the B-role tail depends on ⌈r⌉₁₆), which
  the format-carries-alignment scheme makes cheap.

### 4b.5 Host API 9, pack format 2, the test hook **[stage 2]**

- **`IMD_HOST_API_VERSION` 8 → 9**, additive only:
  - `imd_lowering_t` version 6 (192 bytes; v5 is 184) appends `int32_t k_align` (the
    resolved A: 16 or 64; 0 when `lowered = 0`; **16** for a lowering without residue rows,
    i.e. NOOP, SCALE (k = 0), DIRECT and K-empty, whatever the hook says) and `int32_t
    reserved6` (0). It is `struct_size`-versioned and "writes only the fields that fit", so
    v8 callers are unaffected (tested: a caller passing the v5 size, 184, gets nothing
    written past it and no `k_align`).
  - **`k_hat` changes meaning**: it is `⌈k⌉_{k_align}` (it was always ⌈k⌉₁₆). Callers that
    recomputed k̂ themselves must read it instead; sizing goes only through
    libimmerdink_host (the integration contract, §6a).
  - `imd_options_t` does not change (no `struct_size`; a deliberate later break may give
    it one, and then an explicit `k_align` if wanted, before ehrenfest's B2 integration).
  - `imd_pack_desc_t` and `imd_pack_layout_t` do not change layout. `imd_pack_layout_t
    .format` reports the pack's format and `.k_hat` its ⌈k⌉_A.
- **Pack format 2** (§6b.4 with k̂ = ⌈k⌉₆₄). `IMD_PACK_FORMAT_A16 = 1` and
  `IMD_PACK_FORMAT_A64 = 2` are new constants; **`IMD_PACK_FORMAT` becomes "the newest
  format this library writes" (2)**, no longer the only one. `imd_pack_create` resolves the
  pack's A by AUTO(k) of the pack's k, so packs with k ≤ 96 are format 1, byte-identical
  to every earlier pack, and packs with k > 96 are format 2.
  - The key keeps its words; word 0 is the format: the pack's own, and for a plan's slot
    key the format of the plan's resolved A. k is already in the key, so k̂ follows.
  - **Format 1 stays valid**: nothing that a v9 library would pack as format 1 changes, and
    a format-1 pack is read by a format-1 plan exactly as before.
  - **Where the format lives, and the one real hazard** (review 2026-10-05, must-fix 1).
    The format exists only in the handle's key: pack bytes carry no header, and there is
    no way to import pack bytes into a handle (`immerdink.h`: a handle is made only by
    `imd_pack_create`). So a v9 plan never meets a pre-v9 handle, and inside one process a
    handle and a plan disagree only through the test hook (then PACK_MISMATCH FORMAT).
    The hazard is **pack bytes saved by the caller** under v8 (format 1, k̂ = ⌈k⌉₁₆) and
    later filled into a v9 handle of the same descriptor: for k > 96 that handle is
    format 2, the plan accepts it, and the old bytes are read with the wrong row stride,
    **silently**. Nothing in the library can detect this. Contract (§6a integration
    contract, §6b.5, release note): **a caller that persists pack bytes must persist
    `imd_pack_layout_t.format` and `.pack_bytes` with them, compare both with a fresh
    `imd_pack_layout` of the same descriptor before reuse, and re-pack on any difference.**
  - **Reason order** (must-fix 4; question 3). An **unknown** format word (not 1 or 2: a
    future or foreign key layout) is reported first, before MODULI. The 1-vs-2 difference
    is compared **last**, after every other check and key word of both packs (MODULI,
    ROUNDING, CLASSES, ORDER, then EXTENTS / STRIDES word by word). FORMAT therefore means
    "the same tensor in the other alignment" (in-process: only the hook), and a pack of
    k = 96 offered to a plan of k = 97 reports **EXTENTS**. The slot key's format is the
    **plan's own** (of its resolved A), never `kPackFormat` / `IMD_PACK_FORMAT`. DESIGN
    §6b.5, `oracle.lower_packed` / `key_mismatch` (done) and `analysis.cpp` (stage 2) use
    this one order; PackedOperand `ReasonIffMismatch`, `FormatMeansSameTensor` and
    `neg_reason_fmt_first` (the FORMAT-first order fails) check it.
- **The test hook** `imd_test_set_k_align(int32_t a)`: a test-only export of
  **libimmerdink_host** (not in a public header; ctypes, like the `imd_test_*` hooks of
  libimmerdink). `a = 0` restores AUTO; 16 or 64 forces A for every sizing query, lowering,
  plan and pack created afterwards in the process (an atomic process-global, read once per
  call; libimmerdink plans call the host library's resolver, so device-side plans follow);
  anything else returns 1 and changes nothing. It is in the host library because the
  sizing lives there: a hook in libimmerdink alone would let `imd_workspace_size` and the
  plan disagree (review: confirmed, question 2). Setting it between `imd_pack_create` and
  plan creation is the only way to make a pack and a plan disagree (PACK_MISMATCH FORMAT,
  tested). It is the one exception to §6a's "pure" (documented there).
  - **Where its state lives** (must-fix 3). Both libraries build with hidden visibility
    (`CXX_VISIBILITY_PRESET hidden`, `VISIBILITY_INLINES_HIDDEN`), so an inline variable
    or a function-local static in a header (`k_hat.hpp`) would exist once per `.so` and
    libimmerdink would never see the hook. The state is **one out-of-line
    `std::atomic<int32_t>` in a libimmerdink_host source file** (`src/host/k_hat.cpp`),
    read only through an `IMD_HOST_INTERNAL` accessor (`k_align_hook()`); the header
    declares the accessor, never the variable.
- **One resolver, once per API call** (must-fix 2). `k_align_of(k)` (`src/host/k_hat.cpp`):
  the hook if set, else AUTO(k). It is called **once** per API call, at the top:
  - in `analyze` (every sizing query, lowering and plan: `workspace_layout`, `plan_data`)
    and in `pack_lower` (`imd_pack_layout`, `imd_pack_create`, and per packed operand of a
    packed plan);
  - **[hygiene, host API 10] the hook itself is read once per API call.** An API call that
    analyses several things (a host query with pack descriptors: the plan and each
    descriptor; `imd_perf_predict(_pack)`; `imd_perf_rank`: every candidate's plan and
    packs) takes one snapshot of the hook at its entry (`KAlignSel sel = k_align_sel()`)
    and passes it down: `analyze(…, sel)`, `pack_lower(…, sel)`, `analyze_with(…, sel)`
    resolve `A = sel.of(k)` from it. So every A of one call comes from the same hook
    value (a concurrent `imd_test_set_k_align` cannot split a call), and the analysis of a
    host query with packs runs `analyze` once (it ran twice: `analyze_with`, then again in
    `analyze_packed`; now `analyze` then `match_packs`). `k_align_of(k)` remains as
    `k_align_sel().of(k)` for the single-analysis entry points (`imd_plan_create(_packed)`,
    `imd_pack_create`);
  - from then on A is an explicit argument: `WorkspaceShape` and `pack_layout` take A (as
    `params.workspace_size(k_align=)` / `pack_layout(fmt=)` do), `ExecPlan` gains an
    `int32_t k_align` field (in `ExecPlan`, **not** `K4Plan`: `static_assert(sizeof(K4Plan)
    == kK4PlanBytes)` feeds the plan host bytes), and `k4_tune_bytes`, the perf model and
    `imd_lowering` read it (and `k_hat`) from the plan;
  - **libimmerdink never calls the resolver**: device-side plans take A and k̂ from the
    host library's `ExecPlan`, so a plan cannot see two values of the hook.
  `spec/params.py k_align_of` / `oracle.k_align_of` mirror it.
- **Device code: no logic change.** Every kernel already indexes by the plan's k̂ and
  writes zeros on `[k, k̂)` (KF, K2/K3, KL, the B zero rows, imd_pack; the zero tile:
  h ≥ k gives μ = 0, residue 0). New asserts: `gemm_k4.cu` `k̂ % 16 == 0` (correctness)
  and `k̂ % A == 0` (plan contract, A from `ExecPlan.k_align`); the KF launch
  `(1 << lg_tk) ≥ k̂`. Two checks against code drift that TLA+ cannot see:
  - after a packed slot's key matches, `pl.k_hat == e.k_hat` (the handle's k̂ equals the
    plan's; `analysis.cpp` after the key comparison) or PACK_MISMATCH FORMAT;
  - a debug check (debug builds and a test) that every operand pointer `call_of` forms in
    `gemm_k4.cu` is ≡ 0 mod A. **[hygiene]** It runs as a host-only pre-pass over every
    K4 launch of the call (`k4_check_call`, before K0, so a debug-build violation fails the
    call with nothing queued; the counts of the test hook are unchanged: one per launch).
  **[hygiene]** KF's tile check `2^lg_tk ≥ k̂` is a plan-time check (`plan_data`, for a
  plan that takes KF: INVALID_VALUE, unreachable), no longer evaluated per call.
- **Python**: `python/immerdink/host.py` mirrors gain the lowering's `k_align` (struct
  v6) and the format constants; the `planner` fast path (§6e) calls C and follows. A
  `host.test_set_k_align(a)` helper binds the hook for tests. **[hygiene]** It clears
  `device.default_plan_cache` (a library-owned cache whose key has no A; tested). A
  `planner.Memo` is caller-owned and keyed by the query only, so it is **not** cleared: a
  hit after a hook change returns the result computed under the old A (tested); code that
  sets the hook uses a fresh Memo or calls `clear()`. `planner`'s own per-Query caches hold
  only the operand descriptors, never an A.

### 4b.6 Numerics, accuracy and the performance model **[stage 2]**

- **Bit-identical results for every A.** The zero tail contributes 0 to every product,
  CRT and add-mod do not depend on the chunking, and A changes no K order. The oracle's
  in-line path stays values-only; `oracle.pack()` takes the format (`fmt`, default the
  policy's), `pack_lower`, `pack_layout`, `pack_key` and `lower_packed` take the format /
  the plan's alignment. `test_oracle` checks contractions through format-1 and format-2
  packs (A, B, both; k across the threshold, and forced) bit-exact against the in-line run.
- **Accuracy and κ use the logical k** (§3.2 κ = ⌈√k⌉, §6f `imd_err_*`): identical across
  A. Test: the `err_*` fields and `kappa` of predict / rank / lowering are the same with
  the hook at 16 and at 64.
- **Performance model (§6c).** Its K4 `tile_macs` and byte terms use the plan's k̂ (the
  padded one: what the GEMM tiles and the bytes actually moved), so throughput must keep
  counting the **logical 2mnk** (TFLOPS-eq, `bench/summarize.py`, the useful flops of
  `imd_perf_predict`); the padded k̂ never enters a useful-flop count.
  - **No calibration format bump.** Format 2's slow-chunk rule ("slow iff k̂ mod 32 ≠ 0",
    exact since stage 1) stays exact: an A = 64 plan has k̂ ≡ 0 mod 64 (never slow), an
    A = 16 plan (k ≤ 96) is unchanged. The `perf.cpp:449-456` comments are corrected. θ was
    fitted on misaligned data; it is **refit** (`tools/calibrate.sh`) after stage 2, and
    `k4_select`'s preference for 128×256 is re-checked then (the smaller configurations
    gain the most from A = 64). **Done 2026-10-06** (NOTES-B2 §8): θ refit with the A = 16
    cases below; the rule is +0.65 % over the per-view fastest on a padded-k̂ grid, so it is
    unchanged.
  - **MMA work before the refit.** `tile_macs ∝ k̂` (`k4_config.cpp`) undercounts A = 16
    plans: CUTLASS runs ⌈kc/64⌉ full 64-wide tiles per chunk either way. The model's MMA
    term uses **Σ_c ⌈kc⌉₆₄** over the plan's chunks (A-independent MMA work); only the
    byte terms use the padded k̂.
  - **The refit needs A = 16 cases.** After stage 2 the slow-chunk term is exercised only
    by k ≤ 96, so it would be poorly determined. `tools/calibrate.sh` and `imd_bench` get
    a `--k-align {0,16,64}` flag (the hook) and the calibration grid includes A = 16-forced
    cases at k > 96. The real-implementation A/B sweep (§4b.8) uses the same flag.

### 4b.7 Spec (TLA+, certificates, oracle) **[stage 2]**

- **K4Align** (new, constant-level): CUTLASS's residue-first address sequence is modelled
  explicitly (residue tile at `start + k0`, then `start + k0 + R + T·i`), with every
  operand's absolute row start built from the real index arithmetic: §4a region offsets
  (256-aligned sums, k̂-dependent), z0 batch offsets of gridDim.z parts, j0 row offsets of
  gridDim.y parts, pack planes per residue group, the B role's rows past r. Invariants:
  `Aligned16` (every plan), `TileStartsAligned`, `MainLoopAligned` (A = 64 plans),
  `ExtentOK`, `PadBounded`. Negatives, each failing for its stated reason (the correctness
  invariants listed first, a companion positive shows each mutant is still correct):
  `neg_stride_only` and `neg_partial_landing` (MainLoopAligned: the tile starts stay
  aligned), `neg_kc32` / `neg_kc_old` (131040 / 131056 analogs: MainLoopAligned),
  `neg_a32` and `neg_base16` (TileStartsAligned); witnesses: A = 16 plans are misaligned,
  multi-chunk A = 64 plans are reached. Scaled TileK 64 → 8, 16 → 2, 256 → 32, K_c 131071 →
  23 (K_c' 16, alternatives 20 and 22), K_PAD64_MIN 96 → 12, k = 1…40.
- **Pipeline**: the stage-2 models set `AlignK = TileK = 4·AlignN` (A = 64 against the
  16-quantum n̂), k = 9 (k̂ = 16: 7 padding columns, 6 of them beyond the old quantum),
  K_c' = 8 (two chunks), on every producer path: K2/K3 (`pos_a64_g2`, with z split),
  batches (`pos_a64_batch2`), the n̂ split (`pos_a64_ysplit`), KF (`pos_a64_fused`), KL
  (`pos_a64_large`), packed A / B (B-role tail) / both (`pos_a64_packed_*`) and a forced
  small k (`pos_a64_forced_small`). New invariants: **`EveryRowByteWritten`** (every byte
  K4 reads, any K position < k̂, any column < n̂, was written in this call or by imd_pack:
  §4 "every byte the GEMM reads is written", stronger than NoStaleRead) and
  **`RowPhaseAligned(TileK)`** (CUTLASS's load sequence over Pipeline's own §4a / §6b.4
  addresses). Negatives: producers padding only to the old quantum (`neg_a64_pad_old`:
  NoStaleRead; its B-side-only variants on K3, KF and KL: NoStaleRead holds but
  EveryRowByteWritten fails), imd_pack padding to the old quantum (`neg_a64_pack_pad_old`
  EveryRowByteWritten, `_both` NoStaleRead), the pack tail sized with the old k̂
  (`neg_a64_pack_tail_old`: AccessesSafe) and K_c' ≡ 48 (`neg_a64_chunk_old`).
- **PackedOperand**: the format carries A (format 2 = 4·KAlign); the plan and the pack
  resolve it by AUTO(k) or the hook. Format-2 positives (σ, W, C equal; the B-role tail),
  `AutoFormatsAgree` (a sanity check only, §4b.3), `FormatMismatchRejected` (hook-forced
  mismatches are PACK_MISMATCH) and negatives without the key check (an A16 pack read by
  an A64 plan and the reverse: CEqual). **Reason order** (review 2026-10-05): `Reason`
  models §6b.5's order (early checks of every pack, then the key words, the 1-vs-2 FORMAT
  last); `ReasonIffMismatch` and `FormatMeansSameTensor` (FORMAT only when every other
  key part is equal) on `pos_reason_order` (format 2, every key mutation: a pack of
  another k across K_PAD64_MIN is CLASSES in the toy), `pos_reason_hook`,
  `pos_reason_both` and the key-check positives; `neg_reason_fmt_first` (the FORMAT-first
  order) violates `FormatMeansSameTensor`.
- **IndexMap** is unchanged: its single `PadQ` stays (its claims are about the lowering;
  the mixed quanta of the workspace and pack layout are Pipeline's, at AlignK ≠ AlignN, and
  `params.py`'s at the real values).
- **params.py**: `K_PAD64_MIN`, `K_ALIGN_PAD`, `k_align_auto` / `k_align_of(k, policy)`,
  `PACK_FORMATS`, `pack_format_of`, `k4_load_addresses`, `k4_add_chunk_keys`,
  `k4_tune_bytes`; `workspace_size`, `region_offsets` (`k_align=`) and `pack_layout`
  (`fmt=`) take A; `cert_workspace` and `cert_pack_layout` run over both A with
  hand-computed A = 64 examples; **`cert_k_align`** (new): AUTO, every CUTLASS load
  16-aligned (A16) / 64-aligned (A64) over the boundary k list and 0…1100 with bases, z0
  and j0 offsets, each rejected alternative misaligned, launch counts independent of A,
  the exact footprint deltas, the 1/3 overhead bound, the add-key merge (exactly the k of
  §4b.3, e.g. 393008) and the `k4_tune` cap crossing. `cert_row_phase` (stage 1) pins
  A = 16 (its windows are A = 16 rows).
- **The policy switch is one switch** (must-fix 5). `params.k_hat_of` / `oracle.k_hat_of`
  have no silent A = 16: A = None means the policy's A for k; every other default
  (`workspace_size`, `region_offsets`, `pack_layout`, `pack_key`, `pack`) follows the
  policy, and `PACK_FORMAT` (params, oracle: "the newest format written") is derived
  from it (1 under "A16", 2 under "AUTO"). The C++ constants move in the same commit:
  `pack.hpp kPackFormat` → the plan's own format at each use, `IMD_PACK_FORMAT` 1 → 2
  (`IMD_PACK_FORMAT_A16` / `_A64` added). `test_oracle.test_k_align` checks the defaults
  under both policies.

### 4b.8 Test plan (stage 2; GPU parts only while the machine is otherwise idle)

- **Poisoned workspace over the AUTO range, every producer path** (`test_contract`, new
  `test_k_align`): fill the whole workspace (and, for packs, the pack buffer before
  `imd_pack`) with 0xA5, then the call must be bit-exact against the oracle. **initcheck
  is a separate run** (must-fix 6): a `cudaMemset` of 0xA5 counts as initialised, which
  defeats initcheck, and the bit-exact check alone misses padding left unwritten on one
  side only (the other side's zeros cancel it: Pipeline `neg_a64_pad_old_k3`, `_kf`,
  `_kl`). So the same cases also run under `compute-sanitizer --tool initcheck` on an
  **unpoisoned fresh allocation** (workspace and pack buffer from a fresh `cudaMalloc`,
  never memset), which must be clean. The cases:
  - KF at k ∈ {97, 112, 600, 1000} (A = 64) and k ∈ {33, 96} (A = 16, unchanged);
  - KL at k ∈ {1040, 4080} (and `imd_test_set_fused(3)`: KL tiles of 64 rows, so rows
    span several chunks);
  - K2/K3 (`imd_test_set_fused(0)`) at k ∈ {97, 144, 1040} with g < N (later groups
    repacked over poisoned slots);
  - packed A, packed B (role B: the tail, b > 1, j0 > 0 through `imd_test_set_limits`
    max_n_tiles) and both packed, at k ∈ {97, 1000};
  - two chunks: k ∈ {131009, 131072} (k̂ = 131072, chunks 131008 + 64) on 512² and a skinny
    shape; **two and three chunks whose last chunk is a multiple of 64** (a full 64-wide
    residue tile; must-fix 7): k = 196608 (two chunks, 131008 + 65600) and k = 262017
    (k̂ = 262080: three chunks, 131008 + 131008 + 64); and with `imd_test_set_limits(k_chunk = 64·j)` on small shapes. With A = 64, also
    `set_limits(k_chunk = 16·odd)` for correctness only (it breaks the 64-alignment of
    the full chunks, so no alignment claim).
- **The all-zero padding check** of `check_stages` over [k, k̂) (`test_contract.py`, today
  reached only by `suite_stages`' KF) is extended (must-fix 6) to **KL** (k ∈ {1040, 4080})
  and to **K2/K3** with `imd_test_set_fused(0)`, at A = 16 and A = 64.
- **The stage-1 window tests pin A = 16** (must-fix 7). No multiple of 64 lies in
  (j·131008, j·131056], so under AUTO the window cases of `test_contract.py` (`contract_bigk`
  k̂ = 131024, 131040, 131056, 262064) would silently become 64-aligned tails. They set
  `imd_test_set_k_align(16)` (and restore AUTO); `run_case(k4_expect=...)` then still checks
  the window chunk counts.
- **Boundary k** for host queries (`test_host.py`, `gen_workspace_vectors.py`, vs
  `params.py` with `K_ALIGN_POLICY = "AUTO"`): k ∈ {1, 16, 17, 64, 65, 95, 96, 97, 112, 113,
  128, 129, 144, 145, 1023, 1024, 1025, 1040, 4080, 131008, 131009, 131072, 196608,
  196624, 262017, 392961, 393008, 393009}: `k_hat`, `k_align`, workspace and pack layouts,
  `k_chunks`, `gemm_launches`, `fused`, `fused_large`, `kernel_launches`, `k4_tune_bytes`,
  plan bytes and `add_shapes` (393008: one add key fewer at A = 64, §4b.3); the tune cap
  at (2000000, 1, 97, 1, 1). The hook at 16 and 64 for the same k. DIRECT, SCALE and
  K-empty lowerings report `k_align = 16`; a v5-size (184) lowering caller gets nothing
  past byte 184. The `host.py` struct v6 mirror is checked against `sizeof` and offsets.
- **Test call sites that compute k̂ or the format themselves** (must-fix 5) pass A
  explicitly: the plan's `k_align` from the lowering, or the hook's value. Today they rely
  on a default (line numbers of this branch):
  `tests/test_contract.py:135` (`check_stages` k̂), `:366` (stage padding), `:632`, `:645`,
  `:646` (`contract_bigk` chunk lists); `tests/test_host.py:208`, `:480` (`lay.format != 1`
  becomes "the format of `k_align_of(lay.k)`"), `:597`, `:679`;
  `tests/test_host_c.c:225` (`pl.format == IMD_PACK_FORMAT` for a k = 5 pack: format 1,
  but `IMD_PACK_FORMAT` becomes 2; the check becomes `== IMD_PACK_FORMAT_A16`).
- `test_host_c.c`: the hand-computed size literals stay (k ≤ 33), but the line-225 format
  check changes as above; new hand-computed cases at k = 200 (AUTO 64) and k = 5 forced
  to 64 (KF tk = 64); the 180² case recomputed.
- **Python `PlanCache` and the hook** (must-fix 9). The cache key (`device.py`: spec,
  shapes, strides, options, packs, device) does not include the hook, so after
  `test_set_k_align` it would return a plan of the old A. `host.test_set_k_align` clears
  the default `PlanCache` on every call, and the hook tests use fresh caches.
- `test_packed`: format-1 and format-2 packs byte-exact against `oracle.pack(fmt)`; packed
  contractions bit-exact; **PACK_MISMATCH FORMAT** when the hook changes between
  `imd_pack_create` and plan creation (both directions); `imd_pack_layout.format`.
- `test_contract` / `test_frontend`: every K4 configuration (`imd_test_set_k4_config`) at
  A ∈ {16, 64} (hook) for k ∈ {1, 16, 17, 48, 97, 1008, 1040, 4080, 196624}, bit-exact; the
  results of the hook at 16 and at 64 are identical.
- Accuracy: `err_bound`, `err_typical`, `kappa` identical across A.
- `compute-sanitizer --tool memcheck` on the last row / batch of RA, RB and the pack at
  A = 64 (k = 1000, 4080).
- TLA+ and certificates as in §4b.7; the oracle suite (`test_k_align`).
- Bench: re-run the threshold sweep's grid on the real implementation (no emulation)
  and confirm the gains below within noise, then recalibrate (§4b.6).

### 4b.9 Expected gain (sweep, emulated; useful TFLOPS-eq per call, time alongside)

| case | k | A = 16 | A = 64 | gain |
|---|---|---|---|---|
| 2048², N = 14 | 1008 | 15.70 TFLOPS-eq (0.539 ms) | 18.24 (0.464 ms) | +16 % |
| 2048², N = 8 | 1008 | 24.08 (0.351 ms) | 28.23 (0.300 ms) | +17 % |
| 16384×512, N = 14 | 1008 | 13.03 (1.298 ms) | 15.37 (1.101 ms) | +18 % |
| DF 2304² × Q, N = 14 | 1008 | 15.63 (0.685 ms) | 18.52 (0.578 ms) | +18.5 % |
| 2048², N = 14 | 112 | 4.96 (0.189 ms) | 5.13 (0.183 ms) | +3.4 % |
| 512², N = 8 | 176 | 3.34 (0.0276 ms) | 4.10 (0.0225 ms) | +23 % |
| 16384×512, N = 14 | 144 | 5.11 (0.473 ms) | 4.79 (0.505 ms) | −6.4 % (worst large-shape case) |

Mean over k = 16…1024: +2.8 to +5.6 % per call against never padding. Larger k (4080 → 4096,
2048², N = 14, 128×256, measured): 21.0 → 26.2 TFLOPS-eq (+24 %); 64×64 at 4080: +61 %.
The emulation reads ⌈k⌉₆₄ real columns in KF, so the gains are slightly understated.

### 4b.10 Risks and deferred items

- **Partial landing** (stride padded, extent or K_c' not) is worse than today; the single
  resolution per API call with A passed explicitly (§4b.5), `cert_k_align`, K4Align's
  negatives, the `call_of` debug check and the poisoned-workspace test guard it.
- **Resolver drift**: any site computing ⌈k⌉₁₆ directly, or calling the resolver a second
  time, would size or address the wrong rows; there is no such site after stage 0, the
  resolver runs only in `analyze` / `pack_lower`, `pl.k_hat == e.k_hat` is checked after a
  key match, and the boundary-k host tests compare every reported value with `params.py`.
- **The process-global test hook** makes host queries depend on test state; it is test-only,
  atomic, one variable in libimmerdink_host (§4b.5), and every test that sets it restores
  AUTO (and the Python plan cache is cleared).
- **Threshold**: a policy constant, not a format: moving it changes which packs are format
  2 (the key catches every disagreement), workspace sizes for the moved k, and nothing
  numerical.
- **P2, a K4 "realign" kernel (deferred).** A K4 variant that loads the residue tile last
  (or shifts it into shared memory) would remove the chunk-end term without padding, but
  still needs the stride padded, needs a custom `MmaMultistage` (ring buffer, wait depth,
  per-lane ldmatrix offsets) with zero-fill as the only out-of-bounds guard, and its gain is
  unmeasured. It stays a possible later option for A = 16 plans (k ≤ 96) if that range
  turns out to matter.
- **The W / n̂ stride** (excluded, §4b.4) and **an `imd_options_t` `struct_size` break**
  (with an explicit `k_align` option if wanted) are later items.

### 4b.11 Design questions (resolved by the adversarial review of 2026-10-05)

1. **Format mismatch: reject or adopt.** Reject (PACK_MISMATCH FORMAT). Adoption cannot be
   implemented anyway: the plan cannot see the format of caller-held bytes. The question
   rested on a case that cannot happen (a v9 plan never meets a pre-v9 handle); the real
   hazard is caller-persisted bytes, handled by the contract of §4b.5.
2. **The hook's home.** libimmerdink_host (where the sizing is), with its state as one
   out-of-line atomic there and A resolved once per API call (§4b.5); the one documented
   exception to §6a's "pure".
3. **FORMAT vs EXTENTS.** An early FORMAT check only for an unknown format word; the 1-vs-2
   difference after every other key word, so k = 96 against k = 97 reports EXTENTS
   (§4b.5 "Reason order", §6b.5).

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
    **[v1.19]** Per-call N ships as precision phases (§6g): a plan and a pack per N, no new
    API. An Ozaki-I slice mode stays parked with triggers (NOTES-C1 §3.3).

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
                                  176 in v4 [v1.6], 184 in v5 [v1.7], 192 in v6 [stage 2] */
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
  int32_t  k_align;            /* version 6 [stage 2, §4b]: the resolved A (16 | 64); k_hat =
                                  ceil(k)_A */
  int32_t  reserved6;          /* version 6: 0 */
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
  - **[stage 2] Version 6** (host API 9, §4b.5) appends `k_align`, the plan's resolved K
    alignment (16, or 64 for k > 96; 16 for NOOP, SCALE, DIRECT and K-empty, which have no
    residue rows), and `reserved6`. **`k_hat` is ⌈k⌉_{k_align}** from
    API 9 on (it was ⌈k⌉₁₆): a deliberate, documented change of meaning (review
    2026-10-05). A caller must take k̂ (and every size) from the library, never recompute
    ⌈k⌉₁₆: sizing goes only through libimmerdink_host.
  - `dispatch`: NOOP (b·m·n = 0), EMULATE (K0–K5; EMULATE mode, k ≥ 2), DIRECT (K0 + the
    native kernel; DIRECT mode or K-empty), SCALE (k = 0: K0 + the β kernel, §3.7). A call
    with α = 0 also runs only K0 and the β kernel; that is per call, not per plan.
- **Statuses** are those of §6; the queries can return SUCCESS, INVALID_VALUE,
  NOT_SUPPORTED and (host allocation of the O(rank) lowering tables) ALLOC_FAILED.
- **No GPU, no CUDA runtime, no driver.** The functions are **pure** (outputs depend only on
  the arguments; no allocation visible to the caller, no global or static mutable state)
  and **thread-safe** (any number of threads, concurrently with each other and with
  libimmerdink). They read the descriptor arrays only during the call. **[stage 2]** The one
  exception is test-only: `imd_test_set_k_align` (§4b.5), one out-of-line atomic
  process-global in a libimmerdink_host source file (behind an `IMD_HOST_INTERNAL`
  accessor; never a header inline or static, which hidden visibility would duplicate per
  `.so`), read once per API call; tests set and restore it, production never sets it.
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
- **[v1.19]** `imd_perf_calib_t`'s layout and size are unchanged by the optional key
  `warm.kl_drop` (§6c.9), but `param[IMD_PERF_PARAM_KL_DROP_SLOT]` (= 95, the last of the 96
  slots, unused before) is now meaningful: a caller that builds an `imd_perf_calib_t` by hand
  must zero `param` (memset the struct), or a stray value in that slot is read as the key.
- `imd_lowering_t` is versioned by `struct_size`: an older caller's smaller struct gets the
  fields it knows; a newer caller's larger struct keeps its unknown trailing fields
  untouched. Callers check `imd_host_api_version() >= ` the version they need.
- `imd_tensor_t` and `imd_options_t` are §6's; `imd_options_t` has no size field (v1.2 grew
  it; a further change would need a rebuild, as v1.2 did). That is why stage 2 (§4b) adds
  no option: its K alignment is a function of k.
- **Integration contract [stage 2].** Callers size workspaces and packs only through
  libimmerdink_host (never by mirroring §4a). Pack bytes carry no format header: a caller
  that **persists pack bytes** (across processes or library versions) must persist
  `imd_pack_layout_t.format` and `.pack_bytes` with them and re-pack whenever a fresh
  `imd_pack_layout` of the same descriptor reports a different format or size (k > 96
  moves from format 1 to 2 in API 9; the library cannot detect stale bytes; §4b.5, §6b.5).
- **[v1.19] Footprints across N and g (guaranteed from host API 13 on; §6g).** Fix the
  descriptors, `mode`, `vec_mode` and the packed roles. That also fixes k̂ (a function of k
  alone, §4b.3) and the KF / KL choice (a function of k̂ alone). Then:
  - **F1.** `imd_workspace_size` and `imd_workspace_size_packed` (each pack descriptor at the
    query's N and rounding) are nondecreasing in N at every fixed g ≥ 1, and nondecreasing in
    g ∈ [1, N]. With g = 0 (normalized to N) they are nondecreasing in N. This holds over the
    (N, g) where the query succeeds. Two statuses depend on (N, g). With packed roles and RN,
    the packs' κ ≥ T(N) makes the smallest N NOT_SUPPORTED (§6b.3; T grows with N). A size
    past uint64 (INVALID_VALUE, §5 row 5: RA, RB, W or the total, also of a packed role, and
    a pack's res or `pack_bytes`) can occur only at the largest N and g, because every size
    is nondecreasing in both. So at a fixed g the N where the query succeeds form an
    interval, not an up-set. **What is promised** (NOTES-C1 §5.4 S0-2, as amended): if the
    query succeeds at (N_max, g_max), then every (N, g) with 1 ≤ g ≤ N ≤ N_max and g ≤ g_max
    (g = 0 counts as N) either succeeds with at most that many bytes or, with packed roles
    and RN, is NOT_SUPPORTED by the packs' κ ≥ T(N). C-ws rests on this downward statement.
  - **F2.** `pack_bytes` (`imd_pack_layout`, `imd_pack_bytes`) is nondecreasing in N, over
    the N where `imd_pack_layout` reports it (SUCCESS or NOT_SUPPORTED). A size past uint64
    (INVALID_VALUE) can occur only at the largest N, since res grows with N.
  - **F3.** `imd_pack` writes every byte of [0, `pack_bytes`) (§6b.4). A packed contract
    reads only [0, `pack_bytes`) of a packed buffer: that is the buffer's ALIASING extent
    (§6b.5), and CompatSpec's `NoOOBRead` (spec/PackedOperand.tla) models it. **`imd_pack`
    writes no byte beyond `pack_bytes`**: published in v1.19, only after the GPU tests
    `pack_write_extent` and `pack_write_extent_memcheck` passed on every `imd_pack` launch
    shape, including KL's radix > 1 row mapping, with the coverage asserted by the test
    itself (NOTES-C1 IR17; §6g.8).
  - **Not covered: `imd_pack`'s own workspace** (`pack_ws_bytes` of `imd_pack_layout` /
    `imd_pack_bytes`). That it does not depend on N is I1 (§6g.3), a certificate, not a promise.
    A caller who sizes it once for several phases takes the maximum of the per-phase queries.
  - **Consequences** (proof in §6g.2). **C-ws:** one workspace of
    `imd_workspace_size(N_max, g_max)` serves every plan with the same descriptors, options
    and packed roles, N ≤ N_max and g ≤ g_max (g = 0 counts as N), with at most one call in
    flight per workspace (§6 rule 1). By F1, such a plan's query succeeds with no more bytes;
    an N that the packs' κ rejects has no plan. **C-pack:** one buffer of `pack_bytes(N_max)`
    holds the pack of the same descriptor at any N′ ≤ N_max. A caller may instead size per
    phase and re-allocate at a switch; both are exact through these queries.
  - These describe the current formulas and change no meaning, so they need no new API
    version; they are promises from API 13 on, which a later layout or pack format must keep
    (one that could not would be a new function or format, as above). Only the totals and the
    extents are promised: where the regions lie inside the workspace or the pack (I1), the
    plan device bytes (I2) and whether packing shrinks the workspace (I3) are certificates
    (`params.py cert_footprint_monotone`), not ABI, so a later format may change them.
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
tests with `IMMERDINK_HOST_ONLY=ON`, the system compiler and no nvcc on PATH);
**[v1.19]** `host_footprint_monotone` (F1, F2, C-ws, C-pack and I1–I3 on the library ==
`params.py`, §6g.8).

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
`k̂ = ⌈k⌉₁₆` (**[stage 2]** `⌈k⌉_A`, A = 64 iff k > 96: the pack's format, §4b.5).

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
   (e.g. if D4 or E1 needs another K alignment). **[stage 2]** The format word carries the
   K alignment: 1 = A16 (k̂ = ⌈k⌉₁₆), 2 = A64 (k̂ = ⌈k⌉₆₄); a pack's format is AUTO(k) of its
   k, a slot key's that of the plan's resolved A (§4b.5). It is word 0, but a known format
   (1 or 2) is **compared last** (§6b.5 plan-time order); only an unknown word is
   rejected first. **[hygiene] The slot key uses the resolved A of the plan's k
   (`Analysis::k_align`), not the plan's `ExecPlan::k_align` / `imd_lowering_t.k_align`.**
   They differ only for a plan without residue rows (DIRECT, K-empty; SCALE and NOOP have
   no slot to fill), where the lowering reports 16 because no row is ever laid out. Such a
   plan never reads the packed residues (DIRECT reads `data`), and §6b.2 keeps the mode out
   of the key, so a pack must match a DIRECT plan exactly when it matches the EMULATE plan
   of the same tensors: the pack's format is AUTO(k) of the same k (or the same hook), i.e.
   `sel.of(k)` = `Analysis::k_align`. Keying with 16 instead would reject every AUTO pack of
   k > 96 (format 2) from a DIRECT plan as FORMAT. The `pl.k_hat` drift check compares with
   ⌈k⌉ of that same A.
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
  **[v1.19]** No byte outside it is written (§6a F3, checked on every `imd_pack` launch shape).
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
  buffer must be 256-byte aligned (MISALIGNED). **[stage 2] Format 2** is this layout with
  k̂ = ⌈k⌉₆₄ (k > 96): every row and every K4 load is then 64-byte aligned (§4b).
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
the handle's key: a handle's pack is valid for any library version with the same format.
The bytes themselves carry no header and cannot be imported into a handle, so a caller that
persists them must persist the format with them (§6a integration contract). **[stage 2]**
`IMD_PACK_FORMAT_A16 = 1`, `IMD_PACK_FORMAT_A64 = 2`; `IMD_PACK_FORMAT` becomes the newest
format written (2), and which format a pack has is a function of its k (§4b.5): bytes saved
from a v8 handle of k > 96 (format 1) are **not** valid for a v9 handle of the same
descriptor (format 2), and the library cannot tell; the caller compares `format` and
`pack_bytes` and re-packs.

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
- **[v1.19] Re-packing in place** (precision phases, §6g; e.g. the same X at another N):
  - A packed buffer holds the pack of its **last** `imd_pack` only. Every contract that reads
    it must pass a handle whose key is that pack's key.
  - The library compares the handle with the plan, never the bytes with the handle
    (§6b.10 #12). So a contract that passes the previous pack's handle, with its plan, after a
    re-pack at another N reads the bytes as the old N and returns a wrong C with SUCCESS
    (§6b.10 #3's documented limit; shown by `packed_repack_n`).
  - A CUDA graph that captured a contract on the buffer must not be replayed after the buffer
    is re-packed with another key.
  - Ordering is the WAR rule above: re-pack only after every contract that read the old
    contents.
  - A buffer of `pack_bytes(N_max)` holds the pack at any N′ ≤ N_max (§6a C-pack). The bytes
    above `pack_bytes(N′)` keep stale planes that nothing reads.
  - Python's `PackedBuffer` checks the plain stream-ordered pattern on the host (§6g.5).
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
    **[v1.19]** C1(a) keeps it: one pack per N, re-packed in place into one buffer sized for
    the largest N (§6a C-pack, §6b.7, §6g). The slice format stays parked (NOTES-C1 §3.3).

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
  construction); then per packed operand, the caller's A first: `_FORMAT` (**[stage 2]**
  only an unknown format word, not 1 or 2), `_MODULI`,
  `_ROUNDING`, `_CLASSES` (position map; a K label X does not carry is `_CLASSES` by the
  §6b.10 #10 policy, checked explicitly: a pack with an equal stride-0 mode would otherwise
  match, soundly); then, both packed, `_ORDER` (the mapped K or L orders differ); then the
  plan adopts the orders (K and L from the packs, each packed operand's row class from its
  pack; §5 step 9 fusion redone in them) and compares each slot key word for word:
  `_EXTENTS` (b, r, k, a mode count or extent) or `_STRIDES`, the first differing word
  after word 0; **[stage 2]** then, every other word of both keys equal, `_FORMAT` if the
  pack's known format differs from the **plan's own** (never compared with
  `IMD_PACK_FORMAT`), then `pl.k_hat == e.k_hat` (else `_FORMAT`: a drift check); then
  §3.2 RN acceptance last. The same order is `oracle.lower_packed` / `key_mismatch`. imd_workspace_size_packed / _layout_packed stop before the
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
**[v1.19]** One carve-out: `immerdink/precision.py` (§6g.10–§6g.11) is a caller-side helper,
not part of the binding. Its one numeric, `rho_from_output`'s ρ̂, has its reference and
certificate in spec/params.py (`rho_recipe`, bit for bit; `cert_rho_recipe`), and its sizes
(`footprints()`) are sums and maxima of exact query results. It moves no data.

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
| `immerdink/precision.py` | [v1.19, §6g.10–§6g.11] precision phases chosen from targets on the host: `PrecisionPlan` (composes a Planner's queries per phase switch; computes in Python, so not in planner.py) and `rho_from_output` (the rigorous ρ of a previous output; needs numpy). A caller-side helper, the one carve-out from "no numerics, no sizing logic" above. Not imported by `import immerdink`. |

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
# [v1.19] precision phases (§6g): one buffer re-packed at several N, guarded on the host
p8 = imd.Pack(spec.with_moduli(8))                            # the same descriptor at N = 8
buf = imd.PackedBuffer(bufA)                                  # records the last Pack.pack
p8.pack(A, out=buf, workspace=pws, stream=s); buf.last_pack   # PackRecord(N=8, ...)
plan.contract(A, B, C, workspace=ws, packed=(buf, None))      # N = 14 plan: PackMismatchError

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
  tuple; None only when the C function accepts NULL (0 bytes needed). **[v1.19]** A packed
  buffer may also be a `PackedBuffer` (§6g.5).
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
| `KF fused` | `fused` (§4) | KF | 1 | `Σ_x (8E_x + g·pad_x·k̂ + 3rows_x) + 8V·[GENERATE]` | `g·Σ_x E_x` residues | `Σ_x rows_x` | `f·x₂` ([v1.17] `elem`, below); x₅ = the busiest SM's chain (below) |
| `KL A`, `KL B` | `fused_large` bit of x | KL | 1 | `8E_x + g·pad_x·k̂ + 3rows_x` | `g·E_x` | `16·rows_x·max(0, k̂ − 2·tk_x)` L2 re-read | `f·x₂` |
| `K2 pack A`, `K3 pack B` | groups of x not packed by KF / KL | PACK | launches (B: + the zero-rows launch per group if n̂ > n) | `Σ_groups (8E_x + g_i·pad_x·k̂)` | `Σ_groups g_i·E_x` | `f·x₂` ([v1.17] `dram`: x₁ − the L2-hit re-reads, below) | |
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
  separates the criteria. A format-1 calibration keeps mod 64 (v6-identical predictions up to
  v1.12). **[v1.13]** The format-1 rule tests each chunk length against 64 and follows
  K_c': every full chunk (131008) is now a multiple of 64 where 131056 was not, and the chunk
  sizes and counts change, so format-1 predictions for k̂ > 131008 are no longer
  v6-identical (single-chunk calls, k̂ ≤ 131008, are unchanged).
  **Per chunk (review 2026-10-05):** K4 reads chunk c of a row at `row·k̂ + c·K_c'` (lda = k̂,
  gemm_k4.cu), so the criterion is the row start of each chunk: slow iff `k̂ mod 32 ≠ 0` or
  `c·K_c' mod 32 ≠ 0`; K_c' = 131056 ≡ 16 (mod 32), so for an aligned k̂ the odd chunks are
  slow. This equals k̂ mod 32 for single-chunk calls (every measured case but two). **Evidence
  strength:** 128×256 (the rule's usual choice) has the full fine-k̂ sweep; 128×128 has the
  4864–5120 points, where k̂ = 4960 (mod 32 = 0, mod 64 = 32) is still +14 % (partly counter-
  evidence: that tile may also pay for the 64-wide residue tile); 64×128 and 64×64 are
  untested. The long-k check (512² × 262144 vs 262160; three chunks each, since 2·131056 =
  262112 < k̂, not two as first written; v1.10 read it as chunk 1 misaligned vs all misaligned)
  is inconclusive: K4 there is DRAM-bound and 262144 (a power-of-two row stride) is the slower
  one (NOTES-B2 §6).
  **[v1.13] Correction (measured, NOTES-K4ALIGN; §4b).** The per-chunk rule above uses the
  chunk *start*. The measured mechanism depends on the chunk *end*. CUTLASS 2.x loads a
  chunk's residue tile (`kc mod 64` elements) first, so row r's main-loop loads sit at phase
  `(r·k̂ + e_c) mod 64`, where `e_c = k0 + kc` is the chunk end.
  - With K_c' = 131056 (≡ 48 mod 64), an aligned k̂ had its chunk 0 misaligned (class 48),
    not the odd chunks. Later full chunks were not aligned either: chunk c ends at
    (c + 1)·131056 ≡ 48·(c + 1) mod 64, so 262144's chunk 1 (ending at 262112) was class 32,
    and only the last chunk (ending at k̂) was aligned. The 262144 call was slower because
    of its class-48 chunk 0, not because of a DRAM-channel effect of its power-of-two stride: 196608, also aligned, measured worst
    (2048² × 196608, N = 8: 23.4 TFLOPS-eq, 70.5 ms per call).
  - k̂ ≡ 16 mod 32 misaligns half of the rows (the phases cycle through all four 16-byte
    classes), not every row.
  - The slow-chunk count of format 2 happened to agree for every calibrated shape.
  - Since v1.13, K_c' = 131008 ≡ 0 mod 64. Every full chunk ends on a multiple of 64 and
    the last chunk ends at k̂, so the c·K_c' term is always 0 mod 32. The format-2 criterion
    becomes exactly "slow iff k̂ mod 32 ≠ 0" for the 16/48 phase classes. The class-32 cost
    (+3 % for 128×256 and +10 to +19 % for the smaller tiles, measured) has no term of its
    own.
  - The calibration format is unchanged (review ruling 2026-10-05: no format bump, refit θ).
    The shipped θ was fitted with the old chunk length; it was refit after stage 2
    (2026-10-06, NOTES-B2 §8).
  - **[stage 2]** (§4b.6) The rule stays exact with per-plan alignment: an A = 64 plan has
    k̂ ≡ 0 mod 64 and is never "slow"; A = 16 plans (k ≤ 96) are unchanged. No format bump;
    θ was refit after stage 2 and `k4_select` re-checked (unchanged; NOTES-B2 §8). The model's K4 and byte terms use
    the plan's (padded) k̂; useful flops stay the logical 2mnk.

- **The eps path — retired [v1.15]** (was: the `op_eps` features, reviewed 2026-10-05). Up to
  v1.14 the packs paid one more modular multiply per residue when ε > 0 (|a'| ≥ 2⁵³), per warp
  tile, so the pack cost depended on the **data**; the model took that as an input
  (`eps_fraction` f ∈ [0, 1] in the requests, a plan-only typical-data ramp
  f = clamp((D + 2.5)/1.5, 0, 1) with D = log₂T − ½·log₂k − 53 as the default, `eps_s` the
  eps terms at f = 1, the bracket `[time − f·eps_s, time + (1 − f)·eps_s]`, and the
  calibration identified `op_eps` on `imd_bench --data dominant` cases). Since v1.14
  (§3.4.1) the pack emitter has no eps path: |a'| is reduced from W(N) words by the same
  instructions for every ε, and the measured pack time no longer depends on the data
  (NOTES-BFEPS: dominant-entry data cost what typical data cost, every N, K2/K3, KF, KL and
  `imd_pack`). **[v1.15]** the model drops the dependence, with no ABI change (the fields'
  meaning change is part of host API 11's change set):
  - the `op_eps` feature slots (`kf` slot 4, `kl` slot 4, `pack` slot 3) are always 0, and the
    chain features (`kf.chain`, `pack.chain`) no longer carry the (1 + f) factor;
  - `imd_perf_request_t.eps_fraction` and `imd_perf_rank_request_t.eps_fraction` are
    **validated as before** (< 0, or 0 … 1; NaN and > 1 are INVALID_VALUE) **and then ignored**;
  - `imd_perf_t.eps_s`, `.eps_fraction` and the rank candidates' `eps_s`, `eps_fraction` are
    **always 0** (struct fields kept for the ABI);
  - the `op_eps` parameters stay in the struct and the key set (calibration format 2, no
    format bump: the remaining parameters keep their meaning, and a calibration belongs to the
    kernels it was fitted on, as after v1.13 stage 2). Their keys become **optional (absent =
    0)** in formats 1 and 2, any value ≥ 0 is accepted and has no effect, and
    `imd_perf_calib_format` writes them only when non-zero (an older file keeps its text
    exactly; a file fitted after v1.15 has none);
  - `bench/perf_fit.py` fits them to 0 (all-zero feature columns) and drops the dominant-data
    eps fraction (`record_eps`); `bench/sweep_calib.txt` drops its `--data dominant`
    cases; the hold-out keeps its three dominant-data cases (`v_dom`, `w_d800`, `w_dr1`) as a
    check that the data no longer matter.
  A pre-v1.15 calibration (the shipped one before the combined refit, NOTES-B2 §10) loaded by
  this library predicts without its eps terms; at N ≥ 15 on typical data that is up to the
  eps terms faster than it predicted before, which is what the v1.14 kernels measure.
- **[v1.17] KF's element chain and the PACK L2-hit re-reads** (NOTES-B2 §11; the analysis:
  evaluation workspace `findings/openitems-model-analysis.md`). Two **additive** features in
  the slots the eps path freed (KF x₄, PACK x₃). Every existing feature keeps its meaning and
  the new keys are optional (absent = 0), so a calibration without them predicts
  bit-identically (format 2 kept; §6c.2, §6c.5).
  - **KF x₄ = `elem`**, the busiest SM's per-element chain: KF's launch exactly as for
    `kf.chain` (one helper, `kf_launch_shape`: tiles, parts, E = 16 or 4, r = kernel.kf16 /
    kf4), `x₄ = ⌈B/(S·r)⌉ · ⌈tiles/grid_x⌉ · E`, i.e. `kf.chain` with u = E instead of
    E·⌈g/parts⌉. Each moduli part loads its tile, quantizes it and reduces |a'| to W words
    (§3.4.1) once per element before it emits that element's residues; this cost does not
    grow with the residue count, and the per-residue features (bytes, op, chain) priced it as
    if it did. Measured (REPEAT, 1024³, g = N): KF 30.5 µs at N = 8 and 35.8 µs at N = 14
    (+17 %), where the v1.15 refit predicted 23.5 → 51.8 µs (+120 %); the tiny CC call
    `v_cc12` N = 8 was −29 %. Not clock-bound (mostly the tile load's latency, like
    `pack.wave`). W is not in the term: scaling it by W/2 fitted worse (N ≥ 17 over-predicted),
    and kernel.kf16 / kf4 have the same CTAs per SM for W = 2 and 3 ([v1.16] W-aware
    occupancy, asserted by `contract_api`), so no per-W occupancy is needed.
  - **PACK x₃ = `dram`**, the phase bytes that are not L2 hits: `x₃ = x₁ − hit`, where `hit` =
    the operand re-reads of the phase's launches, `(groups − [x's first group is in this
    phase]) · 8·rows_x·k`, when the call's group working set `S = Σ_{in-line y} (8·rows_y·k +
    g·pad_y·k̂)` (the in-line operands plus one group's residues) is at most `L_hit` =
    `warm.pack_l2_bytes`, and 0 otherwise (always 0 when L_hit = 0). A hit byte costs
    `pack.byte`, every other byte `pack.byte + pack.dram`. [review 2026-10-06] **S is a proxy,
    not the bytes that pass through L2 between two re-reads of x:** it leaves out the W planes
    K4 writes for each group (`g·b·m·n̂` bytes, which K4's and K5's terms count) and the
    per-group residues K4 reads from an operand packed in advance with `imd_pack`. L_hit is
    fitted on square shapes only, where the planes are a fixed fraction of S; on shapes with
    large m·n and short k the proxy under-counts the set and predicts hits that miss (4096² ×
    512 at g = 4: S ≈ 50 MB, with the planes ≈ 117 MB; about 100 MB of re-reads priced at
    `pack.byte`, roughly 5 % of the call too fast). Counting the planes would put the 4090's
    step (1472³ ≈ 61 MB, 1792³ ≈ 90 MB with planes) around the physical 75.5 MB L2 instead of
    0.8 × L2; that is a feature change (a refit), left for when a case with S between 57 and
    77 MB, a short-k case with large m·n and a pre-packed case at g < N are in the sweep.
    Measured (g = 4, KL path, squares): K3 predicted 1.5–1.7× slow for S ≤ 57 MB (1280³,
    1536³) and 1.15–1.4× fast for S ≥ 77 MB (1792³ … 2560³), at every N; one byte rate
    averaged the two. The 4090 data bound L_hit to [57, 77) MB; `perf_fit.py` sets it to
    0.8 × the nominal L2 (60.4 MB, the cache term's residency fraction, §6c.2). Not clock-bound.
    Terms in W, the grid shape or the occupancy fitted to 0 in the analysis.
    **[v1.18]** With `warm.pack_l2_planes = 1` S counts one group's W planes,
    `+ g·b·m·n̂·(2·k_chunks − 1)`, and L_hit = 1.1 × the nominal L2 (§6c.8; the four new
    non-square cases that ruled the proxy out are in the calibration sweep).
  - Not adopted: the analysis's KF read / write byte split (residue writes cost ~1/3 of an
    input read per byte: they stay in L2 until K4). It needs a seventh KF slot, or a
    retirement of `kf.op` / `kf.row` (both fit 0 on the 4090, but retiring them changes the
    meaning of existing keys, which needs a format bump); offline, on the combined refit's
    records, it moved no hold-out statistic (median, p90, max per size class) by more than
    1 point over `elem` + `dram` (NOTES-B2 §11). [v1.18] Re-examined and again not adopted
    (§6c.8: `kf.row` is non-zero in format-1 files, so it cannot be retired without a format
    bump; gain ≤ 0.8 points).
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
MHz at 4096³; 2240–2270 MHz NVML after 4 min at 16384³, §6c.7 [v1.15 draft], where the
"about 1950 MHz" of v1.10 was most likely a single post-loop NVML sample). ehrenfest runs long repeated sequences. The model is
split into a time model at a given clock and a clock model per regime.

- **Time at clock f.** Each feature is either clock-bound (scales with the SM clock) or not:
  DRAM bytes (every `byte` feature, K4's DRAM term), `launch` and the call term are not; every
  other feature (op, row, chain, l2byte, K4's round model; [v1.15] the retired op_eps had
  no feature) is; [v1.17] `kf.elem` and
  `pack.dram` are not (a tile-load latency and DRAM bytes). With the calibration's
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
  within a timed loop). On the 4090: 2530 MHz (boost 2805). **[v1.17] Selection rule:** the
  scan (17 candidates, 25 MHz apart, from the boost clock down) is flat near its minimum and
  its argmin is noise (2026-10-06: 2455 / 2480 / 2505 / 2530 MHz gave 1.07 / 0.78 / 0.97 /
  0.88 %, and the folds of a nested leave-one-out chose 2405–2530 MHz). `perf_fit.py` takes
  the vertex of the least-squares parabola through the scan points within ±100 MHz of the
  argmin (clamped to that window; the argmin itself if the parabola is not convex or has
  fewer than 5 points) and prints the **nested** leave-one-case-out error (f_ref selected
  inside each fold by the same rule, the left-out case predicted at that fold's f_ref and
  power fit), which the notes report instead of the scan's minimum. `clock_mhz` in requests and
  results is on this scale.
- **Power at clock f** (an energy roofline): `P(f) = P_idle + [ (e_tc·O_tc + e_alu·O_alu) ·
  (f/f_ref)^γ + e_byte·D ] / t(f)`, with O_tc = K4's int8 ops (its phase `ops`), O_alu = the
  residue and CRT limb operations (the op features of KF, KL, PACK, K5) plus DIRECT's FP64
  flops, D = the compulsory DRAM bytes (Σ phase `bytes`), γ = 2 (dynamic energy per operation
  ∝ V², V ∝ f in the DVFS range; the fitter scans γ ∈ {1, 2, 3} and keeps the best: γ = 1 on
  the 4090, the windows span 2300–2800 MHz only). P(f) rises monotonically with f (numerator
  up, t(f) down). The coefficients are effective: e_byte (335 pJ/B on the 4090) carries the SM
  power of memory-bound work too, e_alu fitted to 0.
  **[v1.15, §6c.7]** With the optional `power.byte_f_pj` (e_byte^f, SM/L2-side energy per
  modelled DRAM byte, clock-scaled): `P(f) = P_idle + [ (e_tc·O_tc + e_alu·O_alu +
  e_byte^f·D)·(f/f_ref)^γ + e_byte·D ] / t(f)`. e_byte^f = 0 (absent) is the form above, bit
  for bit; P(f) stays monotone in f.
- **[v1.18]** With `power.clock_boost_mhz` > 0 the clock range ends at the boost clock f_top
  instead of f_ref, the time is flat above f_ref, and P(f) gains the busy power and the DRAM
  energy (§6c.8); with the v1.18 keys 0 the rules below are unchanged.
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
  **[v1.15, §6c.7]** `bench/perf_fit.py` fits the clock-scaled form by default: e_byte held
  fixed (`--byte-pj`, default 0, printed with the fit), NNLS of e_tc, e_alu, e_byte^f, γ = 1
  (`--power-gammas` scans other values by the power residual); `--power-form unscaled` is the
  v1.10 fit. The sustained windows get a 60 s pre-heat.
- **Default: SUSTAINED** (accepted, review 2026-10-05: ehrenfest runs long sequences; for calls
  below the cap both regimes coincide, so the default only changes the large, tensor-core-heavy
  calls, which in a sequence do run at the capped clock). Two known biases: **pessimistic** for a
  heavy call between light ones (the card's clock follows the sequence's average power, the
  model the call's own), and **optimistic** for long hot runs. [v1.15 draft, §6c.7: measured, not
  the ~20 % first stated: the settled time after 2–4 min at the cap is 0.6–3.9 % above the
  SUSTAINED prediction (16384³ −3.7 %, 8192³ −1.4 … −1.8 %, a DF 9216² × 3000 −0.6 … −0.9 %);
  of that, ≈ 1–1.8 % is thermal (the die settles at 74–75 °C; the calibration windows' 67–74 °C
  are end-of-window readings). 8192³ and DF are therefore mostly thermal; 16384³ is roughly
  half thermal and half a fit bias of the unscaled power model, which solves too high a
  clock for tensor-core-heavy calls.] A caller that has measured
  its settled clock passes it as `clock_mhz` (NVML scale).
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
per byte). **[v1.17]** `--l2 flush` evicts the L2 before each *timed* call only: a flushed
record's per-phase times (the profiled phase reps) are REPEAT state, and `perf_fit.py` now
raises them to COLD like every repeat record's phases (it treated them as COLD; no
calibration or hold-out record used `--l2 flush`, so no shipped fit was affected). **If the measured COLD / WARM difference is below a few
% of the call on every measured shape, the calibration keeps Δ = 0 (COLD = WARM, the option
stays as a reported no-op) and NOTES-B2 says so.** A format-1 calibration has Δ = 0.
**[v1.18]** The flush *reads* its buffer, so a flushed (and method pair's COLD) record starts
with a clean L2, which a REPEAT call does not have; and the REPEAT saving has a per-call part.
§6c.8 adds `warm.call_ns` (c, applied when res > 0; supported, **0 in the shipped
calibration** since perf-minor-noc), `warm.clean_byte` (ω) and the state
`IMD_PERF_CACHE_CLEAN` (COLD minus ω·min(F − I, L_w) on the call phase); COLD stays the default
(the model's state of a call inside a sequence, not a measured one, §6c.8).

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
- **[v1.17] Fit and data.** `perf_fit.py` writes `warm.pack_l2_bytes` = 0.8 × the records'
  nominal L2 (`--pack-l2-frac`; the 4090: 60.4 MB) into the seed calibration before any
  feature is computed, so fit and prediction see the same hit rule, and prints the PACK error
  over 0.6–1.0 × L2 as a diagnostic (no selection: the calibration set has one case on each
  side of the step). `CLOCK_BOUND`: kf (2, 3, 5), pack (2, 4). `bench/sweep_calib.txt` adds
  the step's cases (`c_l2s1472` N = 8 / 14 / 20, S = 52 MB, and `c_l2s1792` N = 14, S = 77 MB,
  at g = 4; `c_k3hit` / `c_k3miss`, 1024 / 4096 × 96 × 4096 at g = 2, S ≈ 46 / 150 MB; 1536³
  is a hold-out shape); `bench/sweep_perf_validate.txt` adds N = 18–20 cases (W = 3, §3.4.1:
  `v_n18a` … `v_n18d`).
- **File format 1** (`*.calib`): UTF-8 text, one `key = value` per line, `#` comments and
  blank lines ignored, keys unique, values to the end of the line (trimmed). Required:
  `format = 1`, `gpu`, `sm_count`, the three peaks, `l2_bytes` and **every** parameter
  (`<class>.<feature>`, ns per feature unit, e.g. `kl.byte`, `kl.op_eps`, `k4_64x64.mac2`,
  `k4_64x64.byte`; [v1.15] except `kf.op_eps`, `kl.op_eps`, `pack.op_eps`: optional, absent =
  0, no effect, written only when non-zero, §6c.1 "The eps path — retired"; [v1.17] they are
  **retired keys**: accepted in formats 1 and 2 (a number ≥ 0, at most once) and discarded,
  never written: their slots now hold `kf.elem` and `pack.dram`, and `kl.op_eps` has no slot);
  optional
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
  (sustained windows with 15 s pre-heat, ≈ 7 min; [v1.15] 60 s pre-heat, ≈ 17 min, and a
  2 s trace) feeds the power fit (`tools/calibrate.sh --sustained-only` runs just this sweep), `bench/sweep_cache.txt` (flush / repeat /
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
  - **[v1.15, host API 11] optional:** `power.byte_f_pj` (≥ 0; absent = 0), the clock-scaled
    byte energy of §6c.7. [v1.15] With the `op_eps` keys (§6c.1) it is one of the optional keys; in a format-1 file it is a format-2 key
    (rejected). A library older than API 11 rejects a file containing it (unknown key).
    `imd_perf_calib_format` writes it only when it is non-zero, so the text of a calibration
    without it is unchanged and format → parse stays bit-exact.
  - **[v1.15] Not backward-readable.** The format number stays 2, but a format-2 file written
    by an API-11 library generally **does not load in an API ≤ 10 library**: it omits the
    `op_eps` keys when they are 0 (every file fitted after v1.15, the shipped one included),
    which API ≤ 10 requires (MISSING_KEY), and it contains `power.byte_f_pj` when non-zero
    (unknown key). Only a file that still carries all three `op_eps` keys and no `byte_f`
    loads in both. "Format 2" names the key grammar this library reads, not a promise that
    older libraries read every file it writes. [v1.18] Likewise a file with any v1.18 key (the
    shipped one) does not load in API ≤ 12 (unknown key); `host_perf` checks that the shipped
    file omits the `op_eps` keys and carries `power.byte_f_pj` or a v1.18 key.
  - **[v1.17, host API 12] optional:** `kf.elem`, `pack.dram` (parameters, ≥ 0) and
    `warm.pack_l2_bytes` (L_hit of §6c.1, ≥ 0; the field `pack_l2_hit_bytes`), each absent
    = 0 and written only when non-zero; format-2 keys (rejected in a format-1 file). With
    all three absent a calibration predicts bit-identically to API 11 (tested): the features
    are additive, no existing key changes meaning. **Why format 2 is kept** (§6c.5 says a
    feature change is a new format): that rule exists so that a file is never read with
    features it was not fitted with; here an older file sees exactly its own features (the
    new parameters are 0) and a newer file is rejected by API ≤ 11 (unknown key), so no file
    is misread. A bump would be needed if an existing key's meaning changed (e.g. the KF
    read / write split of §6c.1, not adopted). A library older than API 12 rejects a file
    containing any of the three keys.
  - **[v1.18, host API 13] optional** (§6c.8): `power.clock_boost_mhz`, `power.act_w`,
    `power.act_gap_us`, `power.dram_pj`, `warm.call_ns`, `warm.clean_byte`,
    `warm.pack_l2_planes` (0 or 1), each absent = 0 and written only when non-zero; format-2
    keys. With all seven absent a calibration predicts bit-identically to API 12 (tested),
    and a library older than API 13 rejects a file containing any of them, so no *library*
    reads a file with features it was not fitted with and format 2 is kept for the reason
    above. [v1.18 review] Unlike v1.17's keys, three of them change what existing keys do when
    present (`warm.pack_l2_planes`: the S that `warm.pack_l2_bytes` thresholds; `power.act_w`
    > 0: γ is also the busy term's exponent; `warm.l2_bytes` also gates the DRAM energy), so a
    caller's **v7…v12-sized struct** does read the shipped file with features it was not
    fitted with: a stated exception, a degraded view (pass NULL; §6c.8 "Host API 13").
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
  **[v1.15] A version-7…10-sized struct** (a caller compiled against host API 7–10) has no
  `energy_byte_f_pj`: `imd_perf_calib_default`, `_parse`, `_load` (and the `_ex` forms) give
  it the **folded view** `energy_byte_pj := byte_pj + byte_f_pj` (byte_f dropped): the
  unscaled power form with the total byte energy at the scaled fit's e_tc and f_ref. Like the
  v6 view, it is a **degraded, unvalidated view**: that combination was never fitted (an
  unscaled fit of the same windows selects its own e_tc, e_byte and γ, not the folded
  0.708 pJ/op / 301 pJ/B at γ = 1 and f_ref 2480 MHz), it is not what an API-10 library would
  predict (an API-10 library cannot load a v1.15 file at all, see the format note above, and
  would apply its eps terms), and its clock solve for capped calls is not measured. Such
  callers should pass NULL (the full shipped calibration). The fold happens **only** in
  `imd_perf_calib_default`, `_parse`, `_load` (and the `_ex` forms); the predictors and
  `imd_perf_calib_format` do not fold: they read such a struct with byte_f = 0 (its bytes end
  before the field), which is the folded struct those functions returned. With byte_f = 0
  (every file before v1.15) the view is the full calibration.
  **[v1.17] A version-7…11-sized struct** has no `pack_l2_hit_bytes`. [review 2026-10-06]
  The predictors and `imd_perf_calib_format` read it as `warm_l2_bytes` when `pack.dram` > 0
  and as 0 otherwise (the parse / default / load functions write nothing past the caller's
  size, as before). With `pack.dram` = 0 (every file before v1.17) L_hit has no effect, so
  the view is the full calibration and its text gains no key. For the v1.17 shipped file the
  fallback was exact (its `warm.l2_bytes` = `warm.pack_l2_bytes` = 60397977.6 B, 0.8 × L2).
  **[v1.18 review]** For the v1.18 shipped file it is **not**: L_w = 67.9 MB (64.2 MB since perf-minor-noc), L_hit = 83 MB
  and S counts K4's planes, and the v1.18 fields read as 0, so an API-7…11 caller gets a
  degraded, unvalidated view (pass NULL), as for any other file with `pack.dram` > 0. The test
  checks the exact case on a v1.17-style file (L_hit = L_w, no v1.18 keys) and that the
  shipped file's v11 view is exactly that degraded calibration.
  (v1.17 as first written read it as 0: every PACK byte at the DRAM rate, a worse prediction
  than the old calibration gave such callers.)
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
                 double eps_fraction; /* [v1.15] validated (< 0 or 0..1), ignored */ }
        imd_perf_request_t;
typedef struct {
  int32_t phase, cls;  int64_t launches;
  double  time_s, bytes, ops, bound_s;  double x[IMD_PERF_MAX_PARAMS];
} imd_perf_phase_t;
typedef struct {
  uint32_t struct_size;
  int32_t  status, dispatch, k4_config, sm_count, num_phases;
  double   time_s, bound_s, flops_eq, tflops_eq;
  double   eps_s, eps_fraction;     /* [v1.15] always 0 (were: eps terms at f = 1; f) */
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
  only `eps_fraction` is used ([v1.15]: validated, then ignored).
- **Calibration functions**: parse / load fill `out` (with `out->struct_size` as the
  caller's size; fields past it are not written) or return INVALID_VALUE (with
  `*error_line`, 0 when not tied to a line: a missing key, an unreadable file);
  `imd_perf_calib_format` writes the canonical text (every key, numbers in the shortest round-trip form) and its
  length (without the NUL), INSUFFICIENT_WORKSPACE if `capacity` ≤ length (the length is
  still reported, as `snprintf`). A calibration whose `format` is not 1 or whose `sm_count`
  < 1 is INVALID_VALUE in every function that takes one.
- **[hygiene, host API 10] Explicit parse errors.** Before, a rejected calibration said only
  INVALID_VALUE and a line number, and a broken *shipped* calibration (an over-long `notes`
  line in the embedded file: every string value must fit its field, NUL included) made
  every query with a NULL calibration return INVALID_VALUE with no hint why. The calibration
  **format does not change**; what is added:
  ```c
  enum { IMD_PERF_CALIB_OK = 0,
         IMD_PERF_CALIB_ERR_ARGUMENT = 1,     /* NULL / undersized out or err, NULL text  */
         IMD_PERF_CALIB_ERR_FILE = 2,         /* load: cannot open / read, or >= 1 MiB     */
         IMD_PERF_CALIB_ERR_SYNTAX = 3,       /* a line that is not blank, '#', key = value */
         IMD_PERF_CALIB_ERR_UNKNOWN_KEY = 4,  /* also a known key of no slot (bad class)  */
         IMD_PERF_CALIB_ERR_REPEATED_KEY = 5,
         IMD_PERF_CALIB_ERR_VALUE = 6,        /* not a number, wrong count, out of range   */
         IMD_PERF_CALIB_ERR_TOO_LONG = 7,     /* a string value >= its field's bytes       */
         IMD_PERF_CALIB_ERR_MISSING_KEY = 8,  /* a key the format requires (line 0)        */
         IMD_PERF_CALIB_ERR_FORMAT2_KEY = 9,  /* a format-2 key in a format-1 file          */
         IMD_PERF_CALIB_ERR_INCONSISTENT = 10 /* cross-key rule: power clocks, tie_ratio  */ };
  typedef struct {
    uint32_t struct_size;  /* sizeof(imd_perf_calib_error_t) as the caller compiled it   */
    int32_t  line;         /* 1-based line of the error; 0: not tied to a line           */
    int32_t  reason;       /* IMD_PERF_CALIB_ERR_* (IMD_PERF_CALIB_OK on SUCCESS)        */
    int32_t  reserved;     /* 0                                                          */
    char     key[64];      /* the offending or missing key, NUL-terminated, truncated; "" */
    char     message[192]; /* "line 9: notes: value is 133 bytes, the field holds 127"    */
  } imd_perf_calib_error_t;                                   /* 272 bytes, version 1     */
  imd_status_t imd_perf_calib_parse_ex(const char* text, size_t length, imd_perf_calib_t* out,
                                       imd_perf_calib_error_t* err);  /* err may be NULL */
  imd_status_t imd_perf_calib_load_ex(const char* path, imd_perf_calib_t* out,
                                      imd_perf_calib_error_t* err);
  imd_status_t imd_perf_calib_default_ex(imd_perf_calib_t* out, imd_perf_calib_error_t* err);
  const char*  imd_perf_calib_error_name(int32_t reason);  /* "IMD_PERF_CALIB_ERR_..."; NULL */
  ```
  The `_ex` functions return exactly the status of the function without `_ex` and fill
  `*err` (its first `struct_size` bytes; `struct_size` < 16 is INVALID_VALUE with nothing
  written): on SUCCESS line 0, reason OK, key and message empty; otherwise the **first**
  error in file order (the same line the old `error_line` reports; the old functions are
  now wrappers and keep their results bit for bit). The required-key check after the last
  line names the first missing key in canonical order (`imd_perf_calib_format`'s order).
  The message is ASCII-safe (a non-printable byte of the key is written as `?`). Pure and
  thread-safe like every other host function.
  - **The shipped calibration** is still parsed once; its parse error (if any) is kept with
    it (immutable). `imd_perf_calib_default(_ex)` return INVALID_VALUE with that error
    (message prefixed `shipped calibration: `), and the functions taking a NULL
    calibration still return INVALID_VALUE: the diagnosis is one `imd_perf_calib_default_ex`
    away, and the Python layer (`host.Calibration`, `planner`) raises with it. A broken
    shipped file is meant to be impossible to build: **CMake checks the embedded file at
    configure time** (every `gpu`, `date`, `version`, `clocks`, `notes` value must fit its
    field; a `FATAL_ERROR` names the line), `host_c_api` and `host_perf` parse it, and
    `bench/perf_fit.py` checks what it writes (below).
    **[hygiene-b] Consequence: in a CMake build the broken-shipped-calibration path is
    unreachable** (a too-long string fails configure; every other parse error of the shipped
    file fails `host_c_api` / `host_perf`), so it has **no test-only override and no direct
    test**: what it would return is the `_ex` result of the same text, which the tests cover
    reason by reason on non-shipped files (`calib_strings_too_long`'s vector among them), and
    the NULL-calibration functions only forward that kept error.
  - **Generation-time check** (`bench/perf_fit.py`): the metadata strings are checked
    against the field sizes of the ctypes mirror (UTF-8 bytes, NUL included; no line break;
    no leading or trailing blank, which the parser would trim), a caller's `--notes` or
    `--version` that does not fit is an error (the generated defaults, like `clocks`, are
    cut to fit at a UTF-8 boundary; an empty `--notes ""` / `--version ""` means the
    generated default, as before host API 10), and
    the written file is re-loaded (`Calibration.load`) and must format back to the same
    text. `perf_fit.py --check FILE...` runs the same checks and the C parser on existing
    files (exit 1 with every problem, one per line).
  - Python: `host.CalibrationError(ValueError)` with `.line`, `.reason` (the enum name),
    `.key`, and the message; `Calibration.parse / load / default` raise it.

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
  double   eps_fraction;           /* as imd_perf_request_t ([v1.15] ignored)           */
} imd_perf_rank_request_t;
typedef struct {
  int32_t  num_moduli, residue_group, vec_mode, packed;  /* packed: bit 0 / 1 = caller's A / B */
  int32_t  status, feasible, k4_config, dispatch;
  double   time_s, contract_s, pack_s, tflops_eq, eps_s, eps_fraction;  /* [v1.15] eps_*: 0 */
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
  double energy_byte_f_pj;  /* [v1.15, API 11] clock-scaled byte energy (power.byte_f_pj)   */
  double pack_l2_hit_bytes; /* [v1.17, API 12] L_hit of the PACK hit rule (warm.pack_l2_bytes) */
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
- **[v1.15] host API 11:** `imd_perf_calib_t` grows by `energy_byte_f_pj` (8 bytes, 3672 →
  3680). A struct of the v7–v10 size (3672) is read with byte_f = 0 and receives the folded
  view from the calibration functions (§6c.2). Python: `Calibration.power` has
  `"power.byte_f_pj"`; `host.API_VERSION` 11.
- **[v1.17] host API 12:** `imd_perf_calib_t` grows by `pack_l2_hit_bytes` (8 bytes, 3680 →
  3688); a struct of the v7–v11 size is read with it 0 (§6c.2). `imd_perf_param_name`
  returns `"elem"` for KF slot 4 and `"dram"` for PACK slot 3 (were `"op_eps"`) and NULL for
  KL slot 4 (retired). Python: `Calibration.warm` has `"warm.pack_l2_bytes"`,
  `host.FORMAT2_PARAMS` has `kf.elem` and `pack.dram`; `host.API_VERSION` 12.
- **[v1.18] host API 13:** `imd_perf_calib_t` grows by `clock_boost_mhz`, `power_act_w`,
  `power_act_gap_us`, `energy_dram_pj`, `warm_call_ns`, `warm_clean_byte_ns`,
  `pack_l2_planes` (7 doubles, 3688 → 3744); a struct of the v7–v12 size is read with them 0
  (a degraded view of a v1.18 file: pass NULL). `IMD_PERF_CACHE_CLEAN = 3` (request and rank
  request; 4 is INVALID_VALUE). `clock_mhz` out is the clock used, up to the boost clock.
  Python: `host.API_VERSION` 13, `PERF_CACHES["CLEAN"]`, the keys in `Calibration.power` /
  `.warm`. §6c.8.
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
bracket and the default ramp (exact log₂T) ([v1.15]: replaced, see below), rank: every
footprint equals `imd_workspace_size(_packed)` / `imd_plan_bytes` / `imd_pack_layout` exactly,
the budget is respected, the order is as specified; `perf_calib_shipped`: the shipped file
parses and its sm_count is 128. `host_cuda_free` unchanged (the model is plain C++).

[v1.10] `host_perf` adds: the Python transcription of every chain feature (it re-implements the
shared launch rules independently: KL, K2 / K3, K5 variants, grid caps) against the C features
on the random descriptors; a format-1 calibration (the v6 shipped file, kept as a test vector)
predicts bit-identically to the v6 formulas (chain parameters 0, no clock scaling) and round-
trips as format 1; format-2 parse errors (a missing `kernel.` line, a band with p10 > p90,
resident 0); BURST at `clock_mhz = f_ref` equals BURST; SUSTAINED ≤ BURST in speed (never
faster) and = BURST when P(f_ref) ≤ cap; P(f_s) = cap within 1e-9 relative when capped ([v1.17]: P(f_s ∓ 0.05 MHz) brackets the cap, the bisection's resolution); a
larger clock never predicts a longer time; COLD ≥ WARM; bands contain the time and scale with
it; rank: ties = the feasible candidates within tie_ratio of the best's time, ordered by total_bytes,
`raw_rank` is a permutation reproducing the v6 order, `order = TIME` and a v6-sized request give
the v6 order. `host_c_api`: the v7 sizes, v6-sized structs still accepted. The kernel-side launch
refactor (rules moved to `analysis.hpp`) is covered by the existing bit-exact GPU tests.

[v1.15] `host_perf` (C and Python) adds, for `power.byte_f_pj`: parse / format round trip with
the key (bit-exact) and the shipped text unchanged without it; the key optional in format 2
(absent = 0), rejected in format 1, negative rejected, repeated rejected; the folded view of a
v10-sized (and v9-sized: the same size) struct from default / parse / load (`byte_pj + byte_f`,
byte_f absent) and its predictions equal to the full struct's with e_byte = byte + byte_f,
byte_f = 0; predictions with an explicit `byte_f_pj = 0` bit-identical to the file without the
key (every regime, a set of shapes); monotonicity: with γ = 1 and f_s < f_ref, moving byte
energy from e_byte into e_byte^f never lowers the SUSTAINED clock, raising e_byte^f never
raises it, and the 2026-10-05 refit pair (e_tc 0.650 + e_byte 335 vs e_tc 0.691 + e_byte^f
296) lowers the solved clock of tensor-core-heavy calls (8192³, 16384³) as §6c.7 expects;
a capped solve with byte_f > 0 brackets the cap: P(f_s − 0.05 MHz) ≤ cap ≤ P(f_s + 0.05 MHz).

[v1.15, eps retired] `host_perf` (Python) replaces the eps-bracket / default-ramp / "heavy eps"
checks by: for N in 2..20 and k across the old ramp, every request fraction (−1, 0, 0.5, 1)
predicts bit-identically (time, phases, features, clock) and reports `eps_s = eps_fraction =
0`; the `op_eps` feature slots are 0 in every phase; a calibration with large `op_eps` values
predicts bit-identically to the same one with them 0; the `op_eps` keys optional (absent = 0)
in formats 1 and 2, written only when non-zero (the old shipped text round-trips unchanged); a
fraction > 1 or NaN still INVALID_VALUE (`host_c_api`). Monotonicity (doubling m, n or k never
lowers the predicted time) is checked against the shipped and a synthetic calibration.

[v1.17] `host_perf` (Python) adds: the Python transcription of `kf.elem` (the KF launch rules
re-implemented) and of the PACK hit rule (S against L_hit on both sides of the threshold,
hit = re-reads only) against the C features on the random descriptors; with `kf.elem`,
`pack.dram` and `warm.pack_l2_bytes` absent, or explicitly 0, predictions are bit-identical
to the file without them (the shipped API-11 calibration as a test vector, every regime and
cache state); `pack.dram` > 0 with L_hit 0 charges every byte; the three keys optional,
rejected in format 1, negative rejected, written only when non-zero and round-tripping
bit-exactly; the retired `op_eps` keys accepted and discarded in both formats (not written);
the v11-sized struct reads L_hit as 0. `host_c_api`: the v12 size and offset. Monotonicity
(doubling m, n or k never lowers the predicted time) is re-checked with the new terms.

[v1.18] `host_perf` adds the checks listed in §6c.8 "Tests" (absent = 0 bit-identical, the
keys' grammar, S with planes transcribed, CLEAN / REPEAT terms, boost clock, busy power and
DRAM energy monotone, the v12-sized view); `host_c_api` the v13 size, offsets and the CLEAN
state.

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
λ-fastest kernel), never per-shape factors; a feature change is a new file format ([v1.17]
unless it is purely additive with optional keys, absent = 0, so no file is read with features
it was not fitted with; §6c.2).

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

### 6c.7 Long hot runs and clock-scaled DRAM energy **[v1.15]**

Status: measured (NOTES-B2 §9, branch `followup-hotclock`), decided (independent reviewer,
2026-10-06, the questions at the end) and **implemented** (2026-10-06, branch `hotclock`,
host API 11: the key, the folded view, `call_power`, `perf_fit.py`, the 60 s pre-heat sweep;
"Implementation" below). The shipped calibration was refit with the key in the combined refit
(`followups-combined`, NOTES-B2 §10).

**Measured (2026-10-06, stock clocks, 450 W cap; `imd_bench --methods sustained
--sustain-trace-s 2`).** Six windows of 2–4 min, N = 14, from a cold card (≈ 45 °C idle; the
plan setup brings the die to 51–62 °C before the window) and from a hot one, NVML power,
clock, temperature, clock-event reasons and fan sampled every 20 ms, the energy counter never
read. Settled = the last 30 s. The SUSTAINED column is the shipped calibration's prediction:

| call | early (6–16 s) | settled | die °C, fan | SUSTAINED (vs settled) |
|---|---|---|---|---|
| 16384³ (cold, 240 s) | 38.54 TFLOPS-eq, 228.2 ms, 2306 MHz | 37.71 TFLOPS-eq, 233.3 ms, 2251 MHz | 74.4, 59 % | 224.7 ms (−3.7 %) |
| 16384³ (hot, 120 s) | 37.95, 231.8 ms | 37.63, 233.8 ms, 2241 MHz | 74.8, 61 % | (−3.9 %) |
| 8192³ (cold, 180 s) | 35.60, 30.88 ms, 2325 MHz | 35.00, 31.42 ms, 2280 MHz | 75.1, 61 % | 30.86 ms (−1.8 %) |
| 8192³ (120 s) | 35.58, 30.90 ms | 35.13, 31.30 ms, 2293 MHz | 73.2, 57 % | (−1.4 %) |
| DF `acQ,bdQ->acbd` 96⁴, Q = 3000 (150 s) | 31.87, 15.99 ms | 31.69, 16.08 ms, 2306 MHz | 75.5, 63 % | 15.93 ms (−0.9 %) |
| the same, cold start ¹ | | 31.79, 16.03 ms, 2315 MHz | 74.2, 60 % | (−0.6 %) |

¹ 9–39 s of this window were disturbed by another GPU client (below the cap, 20–27 ms calls)
and are excluded.

- Every window is at the software power cap (reason 0x4) the whole time; no thermal reason
  ever. The die settles at **74–75 °C for every capped call** (the card dissipates the same
  450 W), with the fan still ramping (57–63 %) at the end.
- The heat-up costs **0.15–0.22 % of time per °C** (clock −3.4 … −5 MHz per °C, NVML), i.e.
  +1.3 … +2.7 % from the first seconds to settled. This is the leakage power rising with
  temperature at a fixed cap; a fitted static term is 1.5–2 W per °C.
- **No ~1950 MHz settled clock.** 16384³ settles at 2240–2270 MHz (2 s means 2110–2340 MHz,
  the limiter oscillates). A single NVML read after a loop can land far below the mean (this
  16384³ window's post-loop read was 2085 MHz). The v1.10 "~1950 MHz" figure was most likely
  such a read. The pre-BIOS energy windows of 2026-10-04 (10 s, after a 3 s warm-up) ran
  16384³ at 36.7 TFLOPS-eq and 452 W, which does not fit a 1950 MHz clock (that would give
  about 33).

**What the residual is.** Two parts.
- *Thermal, ≈ 1–1.8 %.* The calibration's sustained windows (15 s pre-heat) ended at
  67–74 °C (end-of-window readings; the windows start cooler), the settled state is 74–75 °C,
  at 0.15–0.22 % per °C. For 8192³ (−1.4 … −1.8 %) and DF (−0.6 … −0.9 %) this is most or all
  of the gap; for 16384³ (−3.7 … −3.9 %) it is roughly half.
- *Fit bias of the unscaled form (the other half at 16384³).* e_byte is not clock-scaled, so
  the fit of the shipped form puts more of the windows' power into e_byte (335 vs 296 pJ/B)
  and less into e_tc: **0.650 pJ/op shipped vs 0.691 in the clock-scaled form**. A
  tensor-core-heavy call then gets too little power at a given clock, the bisection's solved
  clock is too high and the predicted time is **optimistic**. (The scaled part is named below
  as SM/L2-side energy per modelled DRAM byte: the memory clock is fixed at 10251 MHz, so the
  part that follows the SM clock is the on-chip side of the traffic, not DRAM itself.)

Refits of the same 14 calibration windows (NNLS, idle 60 W, as `bench/perf_fit.py`),
predicting the six settled windows as a hold-out. None of the six 2–4 min windows is in the
fit, but only 16384³ is a new shape: 8192³ at N = 14 is a calibration shape (its 15 s window
is in the fit), and the DF call is in the `u_tdf` family (`acQ,bdQ->acbd`, 64⁴ × 2048 in the
fit). So the hold-out is a check of the long-run state, and of shape only at 16384³:

| power-model form (fit on the calibration windows only) | γ | 16384³ | 8192³ | DF | median |err| |
|---|---|---|---|---|---|
| shipped: e_tc·O·(f/f_ref)^γ + e_byte·D | 1 | −3.7 / −3.9 % | −1.8 / −1.4 % | −0.6 / −0.9 % | 1.58 % |
| e_tc·O·(f/f_ref)^γ + e_byte·D·(f/f_ref)^γ (fit: e_tc 0.691, e_byte 296) | 1 | −1.8 / −2.0 % | −0.9 / −0.5 % | −0.3 / −0.6 % | 0.72 % |
| the same | 2 | −0.6 / −0.8 % | −0.4 / −0.1 % | +0.2 / −0.2 % | 0.30 % |
| split e_byte⁰·D + e_byte^f·D·(f/f_ref)^γ (fit γ = 2: 77 + 183 pJ/B) | 2 | −1.1 / −1.3 % | −0.7 / −0.3 % | +0.1 / −0.2 % | 0.48 % |
| shipped form + leakage term a·(T − 70 °C), at the settled T | 1 | −2.9 / −3.0 % | −0.8 / −0.9 % | +0.2 / +0.2 % | 0.82 % |

- On the calibration windows' own leave-one-case-out check, the clock-scaled form also gives
  better capped times: median 1.63 → 0.50 % at γ = 1. Its power residual improves from 3.2 to
  1.9 %. The split form's fit puts e_byte⁰ = 0 at γ = 1.
- Mixed (memory-heavier) calls measured the same day: 8192³ at g = 2 (37.69 ms, 449 W):
  shipped −0.4 %, clock-scaled −0.2 %. 12288 × 384 × 1040 (0.853 ms, 447 W): shipped −7.7 %,
  clock-scaled −9.2 %. This call is dominated by the time model's known −7 … −10 % on this
  shape (NOTES-B2 §8), and it sits just at the cap, where the two forms differ in whether
  it is capped at all.
- BURST and every explicit `clock_mhz` are unchanged by construction, so the hold-out
  (bench/perf_validate.py, BURST) is unaffected.

**Design (the smallest extension; implemented as below).**
1. **Clock-scaled byte energy.** Add one calibration key, `power.byte_f_pj`: the SM/L2-side
   energy per modelled DRAM byte, which scales with the SM clock (the memory clock is fixed at
   10251 MHz, so DRAM's own energy per byte does not). Then
   `P(f) = P_idle + [ (e_tc·O_tc + e_alu·O_alu + e_byte^f·D)·(f/f_ref)^γ + e_byte·D ] / t(f)`.
   - **Host API 11** (followup-hygiene takes 10): one appended `imd_perf_calib_t` field,
     `energy_byte_f_pj`.
   - **Parser:** in format 2, `power.byte_f_pj` is optional (absent = 0, ≥ 0 when present);
     the other format-2 keys stay required. A format-1 file means 0. Existing files predict
     exactly what they do today. A library older than API 11 does not know the key and rejects
     a file that contains it (unknown key: a loud parse error, acceptable).
   - **Older callers (v9/v10 view).** A caller whose `struct_size` is below the API-11 size
     has no room for `energy_byte_f_pj`. `imd_perf_calib_default`, `imd_perf_calib_parse` and
     `imd_perf_calib_load` (and the `_ex` forms) give it a folded view,
     `energy_byte_pj := byte_pj + byte_f_pj` (the unscaled form with the total byte energy),
     in the same way as the v6 view (`to_format1_params`). The predictors do not fold; they
     read such a struct with byte_f = 0 (§6c.2). [review 2026-10-06] The folded view is a
     degraded, unvalidated view, not an API-10 prediction (§6c.2): callers should pass NULL.
   - **Fit:** `power.byte_pj` and `power.byte_f_pj` are collinear on the windows (D·(f/f_ref)^γ
     and D differ only through the 2300–2800 MHz clock spread), so `bench/perf_fit.py` holds
     `byte_pj` fixed (0, or a stated prior written into the fit record) and fits three NNLS
     columns: tc, alu, byte_f. On the 2026-10-05 windows at γ = 1 and byte_pj = 0 it gives
     e_tc 0.691 pJ/op and e_byte^f 296 pJ/B.
   - `perf.cpp call_power` changes by one term. The bisection is unchanged: P(f) is still
     monotone in f.
   - **Refit: needs GPU.** The sustained sweep is re-run with the longer pre-heat (item 2),
     and the form comparison and the γ selection of the table above are redone on those
     windows during the combined refit; only then are the power.* keys written.
   - Tests: `host_perf` gets a v9-size and a v10-size caller (folded view: equal to the
     shipped-form prediction with e_byte = byte + byte_f), the identity at byte_f = 0, a
     format-2 file without the key, and a SUSTAINED case whose solved clock moves the expected
     way (down for a tensor-core-heavy call) when e_byte is moved into e_byte^f.
2. **No THERMAL regime.** A hot run settles at the same die temperature for every capped
   call; beyond the calibration state the thermal effect is ≈ 1–1.8 %. A third regime with a
   temperature input would add an API field and a calibration of the fan curve (ambient
   dependent) for that term. Instead, the recipe in §6c.1 stays: a caller that has measured
   its settled clock passes `clock_mhz`, and **SUSTAINED is documented as +1–2 % optimistic
   for minute-scale runs** (NOTES-B2 §9 gives the settled clocks above). The calibration's
   sustained sweep moves to **`--sustain-warm-s 60`**, so the windows start nearer the
   settled 74–75 °C (+~45 s × 14 windows ≈ 10 GPU minutes per calibration).
3. **Bench (done, not a library change):** `imd_bench --methods sustained --sustain-trace-s
   T` adds a time series to the record: per T seconds the mean s per call, power, clock,
   temperature, OR of the clock-event reasons, and fan. It also adds the die temperature at
   the start and end of every sustained window.

**Design decisions (independent reviewer, 2026-10-06; binding).**
- Q1. **Split key** `power.byte_f_pj` (backward compatible, optional), with `power.byte_pj`
  held fixed in the fit (the two are collinear), not a `power.byte_gamma` exponent key (which
  would change what an old file's e_byte means).
- Q2. **Keep γ = 1** and the power-residual rule. The settled-window hold-out prefers γ = 2
  (0.30 % vs 0.72 %), the calibration-window LOO prefers γ = 1 (0.50 % vs 1.60 %); the
  settled set is only three shapes. Re-checked on the 60 s pre-heat windows at the refit.
- Q3. **No THERMAL regime.** Document the +1–2 % drift of minute-scale runs, keep the
  `clock_mhz` recipe, and use `--sustain-warm-s 60` for the calibration windows; the form
  comparison and γ are redone on those windows in the combined refit (GPU).

**Implementation (2026-10-06, branch `hotclock`; host API 11; NOTES-B2 §9.1).**
- Library: the key, `call_power`, the folded view and the format rule as in the Design list
  above (§6c.2, §6c.3); tests §6c.4 [v1.15]. Nothing else changes; with the shipped file
  (no key) every prediction is bit-identical to host API 10.
- `bench/perf_fit.py --power-form scaled` (default; `--byte-pj` = the fixed e_byte, default
  0, printed with the fit and written as `power.byte_pj`; γ = 1, `--power-gammas` to scan by
  the power residual) / `--power-form unscaled` (v1.10). On the 2026-10-05 windows it
  reproduces e_tc 0.691 pJ/op, e_byte^f 296 pJ/B (and 0.6498 + 335 unscaled).
- `bench/sweep_sustained.txt`: `--sustain-warm-s 60 --sustain-trace-s 2`;
  `tools/calibrate.sh --sustained-only` measures just that sweep.
- **New windows (60 s pre-heat, 14 windows, 2026-10-06, ~20 min of GPU):** they end at
  73–77 °C (median 75; the 15 s windows: 67–74, median 73), i.e. at the settled state of
  long hot runs. Capped windows run 0.8–1.5 % slower than on 2026-10-05.
- **Form comparison on the new windows** (fit on them; leave-one-case-out, and the six
  2–4 min settled windows of 2026-10-05 as a hold-out):

  | form (fit on the 60 s windows) | γ | e_tc pJ/op | e_byte (^f) pJ/B | power LOO med | capped time LOO med (C pipeline) | settled hold-out med / max |err| |
  |---|---|---|---|---|---|---|
  | unscaled (shipped form) | 1 | 0.663 | 345 | 3.6 % | 1.98 % | 0.62 / 2.9 % (16384³ −2.7, −2.9) |
  | unscaled | 2 | 0.722 | 322 | 3.5 % | (0.96 %) | 0.91 / 1.7 % |
  | **scaled** (byte_pj 0) | **1** | **0.707** | **308** | **2.0 %** | **0.64 %** (0.40) | **0.69 / 1.0 %** |
  | scaled | 2 | 0.806 | 254 | 2.6 % | 1.37 % (1.37) | 1.01 / 1.5 % |
  | split (byte free) | 1 | 0.707 | 0 + 308 | 2.0 % | (0.40 %) | 0.69 / 1.0 % |
  | split | 2 | 0.787 | 72 + 198 | 2.4 % | (1.01 %) | 0.85 / 1.4 % |

  (Capped-time LOO: the perf_fit pipeline with the C model; in parentheses the follow-up's
  analyze.py, which solves the clock at the measured windows' rows.) **γ = 1 is confirmed
  (Q2):** best power residual and best LOO; on the hot windows γ = 2 is no longer better on
  the settled hold-out (1.01 vs 0.69 %), so the 2026-10-05 γ = 2 advantage was the cool
  windows' bias. The split form fits e_byte = 0 at γ = 1 again, so `--byte-pj 0` stays.
  The scaled form removes the 16384³ bias (−2.9 → −0.9 %); the settled hold-out is now
  centred (−0.9 … +1.0 %), with no residual sign by shape. A leakage term is not identifiable
  on the new windows (all at 73–77 °C), as Q3 expects: no THERMAL regime.
- **Not shipped:** no refit here. The combined refit runs `perf_fit.py` (scaled form, γ = 1,
  `--byte-pj 0`) with these windows (`--sustained`) and writes `power.byte_f_pj`.
- **Shipped (combined refit, 2026-10-06, branch `followups-combined`, NOTES-B2 §10):** fitted
  on the 9 windows of this run with N ≤ 14 (the 5 N = 16 windows ran the pre-v1.14 pack
  emitter with its eps path and were left out): e_tc 0.708 pJ/op, e_byte 0, e_byte^f 301
  pJ/B, γ = 1, f_ref 2480 MHz; sustained leave-one-case-out median 0.8 % (capped windows
  −1.4 … +0.8 %, u_sq2048 −1.4 %). [review 2026-10-06] **Not an independent figure:** f_ref
  was chosen by minimising this same leave-one-out median over 17 candidates, and the capped
  clock is predicted 2–4 % high, the time error being small because the time model's bias
  cancels it (NOTES-B2 §10). [v1.17] Superseded by the v1.17 refit (all 14 windows
  recollected, f_ref 2491 MHz by the parabola rule, nested leave-one-out 1.19 %; NOTES-B2
  §11). [review 2026-10-06] The capped clock is still predicted 0.8–3.4 % high (12288³: 2335
  vs 2258 MHz) and u_s1024 N = 16 4.3 % low. Every one of the 14 windows ran at the cap
  (447–450 W), including u_s512 and u_s1024 at 2525–2775 MHz; the model puts four of them
  below it: u_s512 345 / 371 W (−23 / −17 %), u_s1024 N = 8 423 W (−6 %) and u_rmm 258 W
  (−43 %). So power is predicted low on every small window that runs at a high clock, not
  only on r_mm: the power model needs a clock-dependent term for small calls.
  **[v1.18]** Both gaps are addressed by §6c.8 (busy power, boost clock, DRAM energy).

### 6c.8 Cache-state terms, the PACK set with K4's planes, busy power and the boost clock **[v1.18]**

Status: designed from the two offline analyses of 2026-10-06 (evaluation workspace
`findings/perfminor-time.md`, `findings/perfminor-power.md`), implemented on branch
`perf-minor` (host API 13), refit and validated in NOTES-B2 §12 (shipped: every part below
marked adopted; the v1.18 changelog has the figures). **[perf-minor-noc]** The per-call REPEAT
saving c (`warm.call_ns`) is **dropped from the shipped fit** by the independent reviewer's
verdict (NOTES-B2 §12 "Decision: c dropped"): not identifiable from the pair records; the key
stays supported (optional, 0 = off) for a future calibration, and the question stays open with
the launch-timing check below. Every new calibration key
is optional (absent = 0) and calibration format 2 is kept: a file without them predicts
bit-identically to host API 12 (tested), and API ≤ 12 rejects a file with any of them
(unknown key), so no library reads a file with features it was not fitted with (§6c.2 "Why
format 2 is kept"; [v1.18 review] a caller's v7…v12-sized struct is the stated exception,
"Host API 13" below). Changed text elsewhere in §6c is marked **[v1.18]** and points here.

**What the analyses found (the reasons for each term).**
- *The COLD / REPEAT pairs are not what the v1.10 cache term assumes.* `imd_bench --l2 flush`
  and method pair's COLD *read* a 2 × L2 buffer before the timed call, so a COLD record starts
  with a **clean** L2, while a REPEAT call starts with up to L_w of the previous call's dirty
  lines (C, the W planes, the residues) and writes them back during the call. On every large
  pair whose inputs cannot be resident (res = 0) COLD is 7–18 µs *faster* than REPEAT (x_s2048,
  x_r1, x_r2, 1536³), which the v1.10 form (`Δ·res`) predicts as 0. The REPEAT saving of small
  calls also has a per-call part (first-touch latency of the resident inputs) that a per-byte
  Δ cannot carry, so Δ was 1.4× too large on 1024³ at N = 8 and the saving collapsed too early
  at N = 14 (1024³ N = 14 hold-out call +13.9 %). The KF "overlap" hypothesis (latency and
  bandwidth terms in a roofline or soft-max) fits worse than the sum in every variant (KF
  terms do add); the 1024³ error is the cache term's.
  **[v1.18 review]** The dirty-write-back reading is a hypothesis, not a measured mechanism,
  and the WARM records argue against it: WARM follows a call that left dirty lines, yet on
  every res = 0 pair WARM − REPEAT is as large as CLEAN − REPEAT or larger (x_r1 −7.0 vs
  −5.4 %, x_s2048 −2.8 vs −1.6 %; held out −12.6 vs −6.1 µs and −28.0 vs −20.8 µs). A timing
  difference fits at least as well: the CLEAN and WARM timed calls start queued behind a GPU
  operation (the flush kernel, the untimed producer call), while REPEAT, and every plain imd
  REPEAT record, starts on an idle stream after a host sync. c and ω are therefore effect-size
  terms fitted to the pairs; which of the two readings is right is open (NOTES-B2 §12
  "Remaining": a REPEAT variant with a dummy kernel before the timed call and a COLD whose
  flush *writes*), and the split between c and ω is not checkable from these pairs.
- *S, the PACK group working set of v1.17, is a proxy that misses K4's W planes.* On four new
  non-square / tensor cases it makes the wrong hit / miss call on three (4096² × 512 g = 4:
  S = 50 MB "hit", measured miss; 4096 × 128 × 1280 and 1600³ g = 4: 65 / 61 MB "miss",
  measured hits). No threshold on the v1.17 S separates them; with one group's W planes counted
  every case (and the calibration's 1472³ / 1792³) is consistent with L_hit ∈ (77.8, 89.8) MB.
- *The SUSTAINED power was evaluated at the wrong clock and missed a busy-GPU power.* The
  solve cannot go above f_ref (2491 MHz, the time model's effective clock), but light calls
  run at 2625–2790 MHz (NVML); at their measured clock the v1.15 form gets u_s512 right. The
  energy per operation rises faster than f (locked 2205 MHz vs stock: exponent ≈ 2.2 up to
  2790 MHz; the shipped γ = 1 over-predicts s512 at 2205 MHz by 33 %), and the DIRECT kernel
  draws 349 W at 0.26 TFLOPS-eq, which no op- or byte-energy term explains: busy SMs draw power
  whatever they compute. Fitted on the 14 calibration windows, the v1.15 form misses 11 new
  windows by a median of 21 % (max 82 %); only forms with a busy-GPU term generalise.

**Time model: the cache state (P1).** Two optional format-2 keys, `warm.call_ns` (c, ns ≥ 0)
and `warm.clean_byte` (ω, ns per byte ≥ 0), and a fourth cache state:

```c
enum { ..., IMD_PERF_CACHE_CLEAN = 3 };  /* the inputs evicted and the L2 clean (flushed by
                                             reads, e.g. imd_bench --l2 flush); host API 13 */
```

- The parameters θ stay **COLD** costs, and COLD stays the default. COLD is a **modelling
  convention**, not a measured state: *the inputs are not resident* (res = 0) and the call pays
  what a REPEAT call with res = 0 pays, i.e. the CLEAN credit is not applied. It is meant as the
  state of a call that follows other kernels in a sequence (ehrenfest); whether such a call's
  L2 really holds dirty lines it must write back, or the measured CLEAN / REPEAT gap is launch
  timing, is open (the reading above). [Design question of the time analysis, decided here:
  the alternative, "COLD = CLEAN" (what the bench's flushed records measure), would make every
  default prediction of a large call ω·L_w faster.] **[v1.18 review]** COLD itself is never
  measured. For a call with res = 0 it equals the measured REPEAT; for res > 0 it is CLEAN +
  ω·min(F − I, L_w), an extrapolation of the additive form that **no record validates** (with
  the shipped calibration, no c, COLD is 1.8 / 4.1 / 3.8 % slower than CLEAN on 256³ / 512³ /
  1024³ at N = 14, at most ω·L_w ≈ 6.7 µs on large calls; with c it was 2.8 / 6.7 / 6.3 % and
  11.5 µs). If the timing reading above is the right one, a call queued in a sequence is closer
  to CLEAN and the default is pessimistic by up to ω·L_w. **[perf-minor-noc]** With c fitted,
  the fitter adds c back onto every REPEAT phase record, so θ (the COLD costs) carried ≈ 1.2 µs
  per small call that no COLD measurement backs (64³ N = 14 COLD 16.68 µs with c, 15.45 without;
  REPEAT 15.37 / 15.36): one reason c is 0 in the shipped file.
- REPEAT / WARM: `cache_ns = −Δ·res − c·[res > 0]`, split over the reading phases by
  `in_bytes / I` and clamped at 0 per phase as in v1.10 (res = `res_repeat`); c = 0 in the
  shipped calibration (perf-minor-noc), so there REPEAT = COLD − Δ·res as in v1.10.
- CLEAN: `res = 0` (as COLD) and the **call** phase gains `−ω·min(F − I, L_w)` (an effect-size
  credit, read as the write-back of up to L_w dirty bytes that a clean L2 does not pay, a
  hypothesis; F − I is the call's written footprint). ω ≈ 0.10 ps/B (shipped, no c; 0.17 with
  c) is far below a DRAM write rate, so the credit is below 20 % of the time to write F − I
  bytes and a CLEAN time is never negative. Bands as in every state.
  **[v1.18 review]** The credit saturates at ω·L_w (≈ 6.7 µs shipped; 11.5 µs with c) too
  early: every res = 0 pair whose measured clean saving is larger is under-predicted (x_r1
  N = 14 and x_s1536 N = 8 by 10.7 µs in-sample, the held-out v_x3072 by 14.1 µs and v_x1280
  by 4.7 µs; with c 5.9 / 9.3 / 0.1 µs), so `min(F − I, L_w)` is probably the wrong form for
  large write-backs (open). WARM stays modelled as REPEAT although the pairs
  measure WARM − REPEAT of −4.5 / −7.2 / −11.9 % on the held-out pairs and down to −7 % on the
  calibration pairs (the model: 0; open, with the timing question above).
- **Fit** (`perf_fit.fit_warm`): L_w scanned over 0.30–1.20 × L2 in steps of 0.05; for each,
  (Δ, c, ω) ≥ 0 minimise the relative squared error of the pairs' measured `CLEAN − REPEAT`
  (method pair's COLD is CLEAN) against the C model's `(CLEAN − COLD) + (COLD − REPEAT)` at
  unit parameters (exact NNLS over the 3 columns; no formula in Python). The REPEAT phases of
  the calibration records are raised to COLD by the C model's per-phase `COLD − REPEAT` as
  before (c enters through the split; ω is a call-phase term and never enters the raise). The
  "Δ = 0 when every measured effect is below 3 %" rule is kept. `--cache-form v110` is the
  v1.10 fit and `--cache-terms` fits a subset of (Δ, c, ω); **[perf-minor-noc]** the default
  is `delta,omega` (c = 0; `delta,c,omega` is the perf-minor fit). Shipped (no c): L_w = 0.85 ×
  L2, Δ = 1.31 ps/B, ω = 0.104 ps/B, pair relative SSE 0.0222 (with c: 0.90 × L2, 1.29, c =
  1.23 µs, 0.169, 0.0136).
- **Which records use CLEAN:** method pair's COLD and the totals of `--l2 flush` records (their
  phases stay REPEAT, v1.17). Method packed's interleaved calls stay COLD (their L2 holds the
  other calls' dirty lines).
- **Which records determine the parameters:** Δ and L_w the pairs whose res changes with N
  (1024³ N = 8–16; L_w is pinned by the N = 14 → 16 drop), c the 256³ / 512³ pairs, ω the
  res = 0 pairs (2048³, 4096 × 768², 256 × 3000², 1536³). `bench/sweep_cache.txt` gains 1024³
  at N = 8 / 10 / 12, 1536³ (N = 8 / 14) and 768³ (N = 8 / 16); the hold-out
  (`bench/sweep_perf_validate.txt`) gains three pairs on shapes the fit does not see (1280³
  N = 14, 3072 × 1024 × 1024 N = 8 (i = 3072, j = k = 1024), 2048 × 768 × 768 N = 14) and
  `perf_validate.py` reports
  every pair's measured and predicted `CLEAN − REPEAT` (an effect size, no target).
- Offline (the model-fix records + the open-items 1024³ pairs, fit `dirty + c`): L_w = 0.90 ×
  L2, Δ = 1.34 ps/B, c = 1.41 µs, ω = 0.19 ps/B; 1024³ N = 14 +13.9 → +5.5 %; the two held-out
  pairs 1536³ (−13.1 µs predicted vs −17.3 / −12.7 measured) and 768³ (+10.4 / +7.7 vs +11.0 /
  +10.5) are predicted where the v1.10 form gives 0 and +13.4.

**Time model: the PACK set with K4's planes (P3).** One optional format-2 key,
`warm.pack_l2_planes` ∈ {0, 1} (absent = 0: the v1.17 S). With 1,

`S = Σ_{in-line y} (8·rows_y·k + g·pad_y·k̂) + g·b·m·n̂·(2·k_chunks − 1)`

(one group's W planes, K4's group-bytes rule, with the plan's `k_chunks`), and L_hit stays
`warm.pack_l2_bytes`; `perf_fit.py --pack-l2-frac` defaults to **1.1** × the nominal L2
(83 MB on the 4090, the middle of the measured bracket (77.8, 89.8) MB; a fixed fraction, not
fitted, as 0.8 was) and writes `warm.pack_l2_planes = 1`. A pre-packed operand's per-group
residues (`g·pad_y·k̂` of a packed y) are **not** added: `imd_bench --methods packed` has no
`--g`, so no g < N packed case exists to check them (left for when it does). Offline: no case of
the hold-out, the calibration set or the open-items records changes its hit decision (L_hit
was not identifiable from them); the four new cases go from 6.9 % (≥ 1 ms) and 5.6 / 21.1 /
26.4 % (100 µs – 1 ms) to 0.3 % and 4.3 / 16.0 / 20.7 %. `bench/sweep_calib.txt` gains the
four cases (the hope, **not borne out** [v1.18 review]: that hits of known bytes make the hit
price `pack.byte` identifiable; it still fits 0, NOTES-B2 §12), and the hold-out gains three
S-rule shapes not used to choose L_hit (as built: `v_s1920` 1920² × 1152, `v_w3584` 3584² ×
512 and `v_df64` DF `acQ,bdQ` a…d = 64, Q = 768, all g = 4; the draft named 2048² × 1024 and
6144 × 256 × 1536). [v1.18 review] Their S (68, 95, 142 MB) all lie outside the bracket
(75.5, 90.6) MB: they test the planes term, not the threshold L_hit itself, which is
untested on held-out data.

**Power: busy power, DRAM energy and the boost clock (S6).** Four optional format-2 keys:
`power.clock_boost_mhz` (f_top, MHz; 0 = `clock_eff_mhz`), `power.act_w` (P_act, W at f_ref),
`power.act_gap_us` (g, µs per launch) and `power.dram_pj` (e_dram, pJ per DRAM byte,
unscaled):

```
P(f) = P_idle + P_act·b·(f/f_ref)^(1+γ)
     + [ (e_tc·O_tc + e_alu·O_alu + e_byte^f·D)·(f/f_ref)^γ + e_byte·D + e_dram·D_dram ] / t(f)
t(f)   = the call's time at f in the requested cache state (the cache terms included)
b      = max(0, 1 − launches·g / t_B)      (the busy share; t_B = the call's COLD time at f_ref)
D_dram = D if the call's footprint F > warm.l2_bytes, else 0
```

- **[v1.18 review] t(f) is the predicted time of the request's cache state.** As first built,
  the power and the clock solve divided by the COLD time in every state, while `fit_power`
  builds its columns with the windows' measured time, which is REPEAT state (the same call in
  a loop) and the windows are predicted as REPEAT: on the 8 small windows the model's COLD
  time is 9–30 % above its REPEAT time (u_s1024 N = 8 30 %, p_s512n8 30 %, u_s512 23 / 17 %),
  so their predicted power was too low and their clock too high. Now t(f) is the time the
  prediction reports (each phase's time at f plus its cache term, clamped at 0 per phase as in
  the time path); a COLD request is unchanged (no cache term). The cache terms do not depend
  on f, so t(f) stays non-increasing and P(f) monotone. t_B stays the COLD time at f_ref (the
  fit's busy column is the C model's b of the COLD state too, so fit and prediction agree).
- **[v1.18 review] Known limits** (open, NOTES-B2 §12): D_dram is a step in the footprint at
  `warm.l2_bytes`, the cache fit's L_w, so the power term is coupled to the cache fit (a refit
  with no cache effect has L_w = 0 and charges every call's bytes; BURST power jumps 502 →
  585 W between 992³ and 1024³ at N = 14); the phases' own DRAM-byte features would be the
  better measure. The BURST `power_w` is the power at f_top and is **not capped** (585–631 W at
  1024³–1536³ N = 14: what the call would draw at the boost clock, not a reading).

- P_act is an energy per busy cycle at the voltage of f (∝ f·V², hence the exponent 1 + γ);
  b uses the fixed BURST time t_B, not t(f), so P(f) stays monotone in f (t(f) is
  non-increasing, every term non-decreasing) and the bisection argument holds. With the four
  keys 0 the two new terms are +0 and the clock rule below is the v1.10 one: bit-identical.
- **Clock regimes** (amends §6c.1 "Clock regimes"). `f_top` = `clock_boost_mhz` if > 0, else
  `clock_eff_mhz`. BURST: f = f_top (a short call runs at the boost clock; its power is
  evaluated there). SUSTAINED: f = f_top if P(f_top) ≤ cap, else the f in [f_min, f_top] with
  P(f) = cap (bisection, as before). An explicit request `clock_mhz` is clamped to f_top
  (values at or above it mean BURST). **The time** is unchanged above f_ref: every clock-bound
  term scales by `f_ref / min(f, f_ref)`, so BURST time is the v1.17 BURST time and a solved
  clock above f_ref costs no time. `imd_perf_t.clock_mhz` (and the candidates') reports the
  clock used, now on the NVML scale up to f_top: the clock at which `power_w` was evaluated
  (reviewer question Q1 of the power analysis, decided: report the solved clock; host API 13
  marks the change; with boost 0 it is the v1.17 value).
- **γ = 2.5 as a prior** (existing key). With every calibration window at the cap, the
  in-sample power residual cannot identify γ (it chose 1); the clock-sweep evidence (locked
  2205 MHz vs stock, exponent ≈ 2.2) and the nested checks favour 2–3, with 2.5 and 3 within
  0.1 point. `perf_fit.py --power-gammas` still scans by the power residual when several are
  given.
- **Fit** (`perf_fit.py --power-form busy`, the new default; `scaled` is the v1.15 form,
  `unscaled` the v1.10 one): NNLS, relative in P − P_idle, of the five columns e_tc, e_alu,
  e_byte^f, P_act, e_dram at each window's measured (f, t); `power.byte_pj` held at `--byte-pj`
  (0), g = `--act-gap-us` (1 µs, a prior: 1.5 µs is equivalent on the windows), P_idle 60 W,
  f_top = the median NVML clock of the windows that never touched the power cap (clock-event
  reason 0x4 absent from their trace; `--clock-boost-mhz` overrides; none: 0). The f_ref scan
  goes down to 2255 MHz (23 candidates; it stopped at 2405 and one fold chose that end).
  **[v1.18 review]** The leave-one-out and nested checks recompute f_top from each fold's own
  training windows (as first built it was the median over all windows, so the left-out
  uncapped window set its own boost clock); a given `--clock-boost-mhz` stays fixed in every
  fold. A busy fit without any uncapped window warns (f_top 0: the solve stops at f_ref while
  P_act is still fitted).
- **Windows.** `bench/sweep_sustained.txt` gains the 12 windows of the power analysis (small:
  256³ / 384³ N = 14, 512³ N = 8, 768³, 1536³ N = 8; KL / DRAM-heavy r_mm-like 1000 × 64 ×
  50000, 384 × 96 × 40000, 4000 × 96 × 40000, 8192 × 256 × 8192; K5-heavy 4096² × 256 and
  8192² × 64; DIRECT 1024³), ≈ +16 min of sweep time.
- **Bench fix:** `imd_bench --methods sustained --mode direct` wrote the variant "RN g=N
  sustained" without " mode=direct", so `perf_fit.py` predicted a DIRECT window as EMULATE;
  the variant now carries the mode as method imd's does.
- Offline (25 windows, nested leave-one-case-out with f_ref chosen in each fold): call time
  |err| median 2.26 → 1.57 %, clock |err| median 6.7 → 1.0 %, capped-clock bias of the large
  calls +4.1 → +1.0 %, SUSTAINED power max |err| 82 → 8.4 %, the locked 2205 MHz check −2.0 …
  +3.0 % (shipped +2 … +33 %). On the 14 calibration windows alone the nested time median
  goes 1.21 → 1.38 %: the shipped f_ref cancelled a +1.9 % clock bias against the time
  model's own error at the capped clock, which is now visible (−1.6 … +1.4 %).

**Not adopted (evidence in the analyses; NOTES-B2 §12).**
- *KF read / write byte split* (P2 of the time analysis: residue writes cheaper than input
  reads). KF has no free slot: the analysis proposed retiring `kf.row` (slot 3), but `kf.row`
  is non-zero in format-1 files (the v6 test vector: 1.52 ns per row), so its retirement
  changes an existing key's meaning, which needs a format bump (§6c.2). Its offline gain over
  P1 alone is ≤ 0.8 points per class (hold-out 100 µs – 1 ms p90 6.4 → 5.6 %), with the KF
  phase median and the open-items records slightly worse; not worth a format change.
- *A KF / PACK / KL roofline* (refuted: worse than the sum in every variant), *a per-modulus
  E = 4 KF term* (fits 3.6 ns per step, ≈ 0; the 352³ N = 14–20 series alternates by parity
  within the 1.024 µs timer step), *a fixed L2-hit price for PACK* (improves the ≥ 1 ms calls,
  worsens the K2 / K3 phase p90; a chosen number, not an identified one: the hit cases in the
  calibration sweep let the fit price hits through `pack.byte` instead; [v1.18 review] not
  borne out: `pack.byte` fits 0 with the four S cases too, the hit cost sits in `pack.wave` /
  `pack.dram`).

**Host API 13** (§6c.3): `imd_perf_calib_t` grows by seven doubles (3688 → 3744 bytes):
`clock_boost_mhz`, `power_act_w`, `power_act_gap_us`, `energy_dram_pj`, `warm_call_ns`,
`warm_clean_byte_ns`, `pack_l2_planes`; `IMD_PERF_CACHE_CLEAN = 3`; `clock_mhz` out may exceed
`clock_eff_mhz` (above). A struct of a v7…v12 size is read with the seven fields 0 (the
parse / default / load functions write nothing past its size, as before): for a file with the
v1.18 keys (the shipped one) that is a **degraded, unvalidated view** (v1.17's S at L_hit
83 MB, no busy power at γ = 2.5, no boost, no per-call saving); such callers should pass NULL.
**[v1.18 review] This is an exception to §6c.2's "no file is read with features it was not
fitted with"**, stated here rather than enforced: three v1.18 keys change what existing keys
do when present (`warm.pack_l2_planes` changes the S that `warm.pack_l2_bytes` thresholds,
`power.act_w` > 0 makes γ also the exponent of the busy term, and `warm.l2_bytes` now also
gates the DRAM energy), so an API-7…12-sized struct reads the shipped file as a model that was
never fitted. Measured with the shipped file (COLD, N = 14): SUSTAINED no longer reaches the
cap on large calls (4096³ N = 14: 4396 µs at 340 W against NULL's 4608 µs at the 450 W cap; with c 4397 / 342 / 4614), and BURST is ≈ −7 % on the 4096² × 512 and
3584² × 512 g = 4 shapes (v1.17's S calls them hits). Refusing such a struct was considered
(`imd_perf_calib_default` / `_parse` failing for a size below v13 when a v1.18 key is set);
not done because the degraded-view rule of v6 / v7…v10 / v7…v11 callers is the established
contract, no known caller passes its own struct (ehrenfest passes NULL), and a refusal would
turn a degraded prediction into a failure of every query. The v7…v11 L_hit fallback below is
likewise no longer exact for the shipped file.
`calib_valid` / the parser: every new value ≥ 0 and finite, `warm.pack_l2_planes` 0 or 1
(VALUE), `power.clock_boost_mhz` 0 or ≥ `power.clock_eff_mhz` (INCONSISTENT). Python:
`host.API_VERSION` 13, `PERF_CACHES["CLEAN"]`, the keys in `Calibration.power` / `.warm`.

**Tests** (`host_perf`, C and Python, §6c.4 [v1.18]): with the seven keys absent or explicitly
0 every prediction (every regime and cache state, a set of shapes) is bit-identical to the
calibration without them; the keys optional, rejected in format 1, negative / repeated
rejected, `pack_l2_planes` ∉ {0, 1} and boost < clock_eff rejected, written only when non-zero
and round-tripping bit-exactly; the Python transcription of S with planes against the C
features on both sides of L_hit; CLEAN = COLD minus `ω·min(F − I, L_w)` on the call phase
only, REPEAT = COLD minus `Δ·res + c` exactly when res > 0; BURST with a boost reports f_top
and the v1.17 BURST time; SUSTAINED ≤ BURST in speed, the capped solve brackets the cap within
0.05 MHz with P_act > 0 and e_dram > 0, a higher clock never predicts a longer time, P(f)
monotone in f; a v12-sized struct reads the fields as 0. [v1.18 review] At a fixed clock
(P − idle)·t is the same in every cache state (the power divides by the state's own time), a
faster state draws more power, and the SUSTAINED REPEAT / WARM / CLEAN solve brackets the cap;
the v11 view is exact for a v1.17-style file and is the documented degraded view of the
shipped one. `host_c_api`: the v13 size and
offsets, `IMD_PERF_CACHE_CLEAN` accepted, 4 rejected.

### 6c.9 The KL share of the REPEAT saving, `warm.kl_drop` **[v1.19]**

Status: designed for MC-3a (NOTES-C1 §5.9, the second review's amendment of MC-3); the account
below comes from the **allowed records only**: the calibration records, the cache sweep, the
Q24 cases (`calib_n.jsonl`) and the v1.18 hold-out, never `mc.jsonl` (evaluation workspace
`findings/c1-mc3a.md`, `scripts/c1/mc3a/account.py`, shipped calibration, BURST, the phase marks
against the model's per-phase REPEAT and COLD predictions). The mechanisms are **[I]**, not
measured. **Outcome (2026-10-09, NOTES-C1 §5.11): the key was fitted once (d = 0.70, with Q24
in the fit) and every §6c.5 class target is met on the fresh narrow-KL hold-out, the v1.18
hold-out (both sets), `mc.jsonl` (with and without the repeat) and a fresh full M-c run, but one
pre-registered in-sample condition is not met under its literal reading (two K4 entries, from
Q24's records and not from the key), so no pass was declared and the candidate calibration is
not shipped: the key is supported (optional, absent = 0, tested) and absent from the shipped
file, and the figures in this section are those of the shipped v1.18 calibration.
Independent review (2026-10-09, binding, ruling A): that condition is read against the shipped
fit's `perf_fit` table, as written, so MC-3a does not pass; M-c stays FAILED, the shipped
calibration stays, `mc3a.calib` stays in the evaluation workspace, no second key is tried, and
M-d2 runs on the shipped calibration and its pre-registered bars.**

**What the phase marks show.** For a phase with mark m and model times R (REPEAT) and C (COLD),
the saving the model credits is C − R and the share the mark keeps is (C − m) / (C − R).
- *KL A, the wide operand's pack, keeps part of its saving, and the part depends on the call,
  not on the footprint rule.* Q24's three narrow-KL shapes (`q_kl1`, `q_kl2`, `q_kl4`; savings
  17.6–24.0 µs): 5–52 % (medians 31 / 19 / 39 %), the share rising with N while the credit stays
  flat. The in-sample records with g < N (`c_k3hit`, `c_skin1280`, `p_k3c`, `c_k3miss`,
  `c_l2s1472`): −74 to 35 %, median 16 %; `p_k3c` at g = N, N = 14 / 16 (a credit of 14.5 / 7.2
  µs, near the end of the window): 113 / 152 %. The rule `res = clamp(L_w − (F − I), 0, I)` (§6c.1
  "Cache state") cannot separate them from the KF calls below.
- *KF keeps its saving, and more than the rule credits at large N.* Most calibration and
  hold-out calls at g = N keep 80–130 % (`c_s768`, `c_s1000`, `c_rkf1` at N = 8, `k_*`, `w_*`,
  `v_s900`, `v_cc12`); the exceptions are small shapes (`c_tcc8`, `v_s512`, `v_rsm1`: 8–73 %)
  and calls where the rule's saving has already collapsed (`c_rkf1` N = 14 211 %, `k_b` 193–285
  %). `q_kf1` keeps 80 % at N = 4 and 142 % at N = 10 (a credit of 25.7 → 13.1 µs), the hold-out's
  `v_sq1024` 90 % at N = 8 and 160 % at N = 14. With g < N, KF keeps nothing: `c_s1000` g = 4
  (three calls) and `c_s768` g = 3 (six) −10 to −2 %, `v_s512` g = 4 26–29 %.
  The cache sweep's `x_s1024` pairs (whole calls) measure CLEAN − REPEAT = 16.4 / 13.3 / 12.3 /
  8.3 / 4.0 µs at N = 8 / 10 / 12 / 14 / 16 against the model's 18.4 / 17.8 / 17.1 / 9.8 /
  0.9 µs: the window's collapse is in the pairs (it is where L_w was fitted), while the KF phase
  marks of `v_sq1024` and `q_kf1` do not show it (KF REPEAT predicted 44.5 µs, mark 35.7 µs at
  N = 14). The marks' share rising with N is, to the extent it is not that collapse, the slope:
  KF's modeled COLD time grows 1.6 µs per modulus on `v_sq1024` (N = 8 to 14) and 1.9 on `q_kf1`
  (N = 4 to 7), the marks 1.0.
  KF is not changed by this section (below).
- *KL B, the narrow operand's pack, is slow on the multi-pass launch, in every cache state.*
  Q24's narrow packs (96–144 rows, rows of 3 chunks, 1024-thread CTAs, a grid of 12–18 CTAs):
  m / R 1.50–1.81 and m / C 1.31–1.65 (savings 1.4–2.0 µs). The narrow packs in the
  calibration (`c_rskin2`, `c_rskin2g`, `p_k3a`, `p_k3b`, `c_skin1280`; rows of one chunk) are
  at 0.87–1.20, the g = 4 packs of `p_k3c` and `c_l2s1472` at 1.10–1.44. Q24's excess is in its
  N slope (0.64–0.68 µs per modulus) and its intercept (17.8–18.3 µs at N → 0).
- *The two KL effects need not be one.* A KL A that kept all of its saving would still leave KL B
  under-predicted, and the call totals cancel against COLD only by accident (NOTES-C1 §5.9).

**The key.** One optional format-2 key, `warm.kl_drop` = d ∈ [0, 1], absent = 0:
- In REPEAT and WARM, the saving of a **KL** phase (`KL A`, `KL B`, `pack KL`) is (1 − d) times
  the rule's `res · Δ` share of that phase (the per-call part c, if fitted, is not scaled).
  KF, K5, COLD, CLEAN and every other phase are exactly as in §6c.8; with d = 0 every prediction
  is bit-identical to the calibration without the key (`host_perf`, v1.19).
- d = 1 is the **path** candidate of the MC-3 text (b), no saving on a KL read; d = 0 the rule.
  The **traffic** candidate (a) (K4's split-K W planes added to F − I) moves the saving the wrong
  way: Q24's share of the saving rises with N while the planes grow with N. A rule in F − I
  alone cannot work (NOTES-C1 §5.9 MC-3: KF's sq1024 keeps 99–157 % at a modeled F − I of
  49–53 MB where the Jacobi view's KL A keeps 27–50 % at 13–31 MB; both sides of that statement
  are now also visible on allowed records, `v_sq1024`, `q_kf1`, `q_kl1`).
- **Fit** (`perf_fit.fit_kl_drop`, after the cache term): for d = 0, 0.05, …, 1 the KL class is
  refitted (exact NNLS) on the KL phase marks raised to COLD with (1 − d) times the model's
  saving, and the REPEAT-state prediction θ · x − (1 − d)(C − R) of **every KL phase** (A and B,
  every record with the narrow-KL shapes of Q24 among them) is scored against its mark by
  relative squared error; the smallest sum wins. It is chosen on the KL A / KL B phase marks, not
  on call totals, and the scan is printed. The scan is opt-in, `--kl-drop scan`: the default is
  `--kl-drop 0` (key absent, the v1.18 rule), so a routine recalibration (`tools/calibrate.sh`)
  cannot ship a nonzero `warm.kl_drop`; `--kl-drop D` fixes it.
- **Where it is stored.** `imd_perf_calib_t` keeps its size (host API 13, calibration format 2):
  the key lives in the last parameter slot, `param[IMD_PERF_PARAM_KL_DROP_SLOT]` (the macro,
  `IMD_PERF_PARAM_CAPACITY - 1` = 95; the capacity of 96 slots holds four classes more than there
  are, and a `static_assert` keeps the classes' slots before it; the text file names the key, so
  it does not depend on the slot, and a hand-built `imd_perf_calib_t` must zero `param`),
  zeroed in a format-1 view and checked to lie in [0, 1]. A library without it rejects a file
  that has the key (unknown key), so no file is read with a feature it was not fitted with.
  Python: `Calibration.warm["warm.kl_drop"]`.

**Not modeled.**
- *KL B's multi-pass time (open model defect, tracked as [GitHub issue #7](https://github.com/Yurgenschmidt/immerdink/issues/7) (label backlog)).* The candidate named by the v1.8 data, a compute
  chain `nc · values · g` steps on the busiest SM (`kl` slot 4, the retired `op_eps` slot),
  fits Q24's narrow packs at 57 ns per step (A and B alike: 1.36 and 0.68 µs per modulus over 24
  and 12 steps) but the calibration's wide launches reject that coefficient: the NNLS gives
  6–13 ns per step with the chain over every launch, over the underfilled ones or over the
  4-value ones, and the leave-one-shape-out median error of a held-out Q24 shape's KL B goes
  13.7 → 12.7 % at the best of them, with d = 0.75 (offline, KL class only, allowed records;
  `scripts/c1/mc3a/kl_cv.py`, diff of the trial in `results/c1/mc3a/chain_exploration.diff`).
  A feature restricted to underfilled launches would fit, but its switch breaks the monotonicity
  of the prediction in the shape (a bigger shape would take less). So no second key; the
  defect is recorded with its size and a trigger in NOTES-B2 §12 and §7 below, and tracked as
  [GitHub issue #7](https://github.com/Yurgenschmidt/immerdink/issues/7) (label backlog). The narrow packs
  of Q24 enter the refit, which absorbs part of it into `kl.wave` and `kl.launch` (NOTES-C1 §5.11
  reports by how much).
- *KF's saving against N and with g < N.* No key. The marks above are the evidence; the Q24 KF
  cases (`q_kf1`) enter the refit, and a KF shape of the fresh hold-out (NOTES-C1 §5.11) tests
  it. Open if that misses (§7).

**Tests** (`host_perf`, C and Python, v1.19): absent or explicit 0 is bit-identical across
every regime, cache state and a set of KL / KF shapes; the grammar (optional, format 2 only, a
number in [0, 1], not repeated; negative, above 1 and `nan` rejected), written only when
non-zero, round trip; the rule on a KL narrow-n case, the Jacobi view and a KF case: a KL
phase's REPEAT time is COLD − (1 − d)(COLD − REPEAT at d = 0), WARM = REPEAT, d = 1 puts a KL
read at its COLD time, COLD / CLEAN and every non-KL phase are unchanged (a mutant that applies
the share to every phase, and one that ignores the key, are caught); `host_c_api`: the slot (95), the
round trip, the three rejections.

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

## 6g. Adaptive precision: precision phases (ROADMAP C1) [v1.19]

Design: docs/NOTES-C1.md, stage 0 (§4.1), with its resolved questions Q1–Q3 and the
independent reviewer's binding decisions IR8–IR10, IR14, IR17 and IR20; stage 1 (§4.2,
§6g.10–§6g.12 here), with Q4, Q5 and IR2, IR3, IR7, IR8, IR15 and IR24. Iterative methods
need full precision only near convergence. So a caller runs each contraction at the smallest
N its current target needs, and switches N between calls, at iteration boundaries where it
returns to the host anyway.

**Decision.**
- **C1(a), per-call N: a plan and a pack per N, on host API 13.** There is no new function,
  struct or option. A **phase** is the set of plans (and packs) at one N.
- **Parked:** one plan serving a range of N. The kernels already select by N on every call:
  K5's instantiation comes from the plan's N, the pack emitter's word count W from N, KF and
  KL's group size from min(g, N), and the group loop runs over N. So a range call is
  launch-identical to the plan at that N and gains nothing per call. It would save only plan
  creation (11–12 µs for a GENERATE plan, NOTES-M4) and some host bytes, at the price of new
  entry points, a new lowering and four TLA+ modules. NOTES-C1 §3.2 keeps it with its trigger.
- **Parked with triggers:** an Ozaki-I slice mode (C1(c)), in which one pack serves every
  N′ ≤ N. No consumer needs one operand at two precisions at once, which is the only case a
  prefix format pays for (NOTES-C1 §1.3, §3.3).
- **C1(b), choosing N from a target, stays on the host** with the §6f queries
  (`imd_err_min_moduli`, rank with an accuracy). Stage 1 adds a thin Python `PrecisionPlan`
  on top of them (§6g.10) and the rigorous ρ-from-output helper `rho_from_output` (§6g.11),
  with no ABI change. Choosing N on the device is rejected, and device statistics (NOTES-C1
  stage 5) are deferred with triggers.
- **The policy stays with the caller:** immerdink never changes N on its own (ehrenfest D8:
  precision is user-directed).

### 6g.1 What a phase is

- Each phase is a plain plan at its N. Every §3 rule, certificate and proof applies
  unchanged, and results are bit-exact against `oracle.contract(..., N)` at that phase's N.
  There is no numerics change.
- σ depends on N only through the plan constant R·2^F (§3.2: R = (2T(N) − κ)² for RN and
  T(N)² for TRUNC). So a pack serves exactly one N: N is word 1 of its key (§6b.10 #15), and
  a plan at another N gets PACK_MISMATCH `_MODULI`.
- Plans are created before the iteration loop. RESIDENT plan creation synchronizes and is
  not capturable (§6), and with `k4_tune` on, each plan autotunes. `PlanCache.freeze()`
  asserts that every plan exists before a capture.
- A packed operand that changes every iteration is re-packed every iteration anyway. An
  invariant one is re-packed once per phase change, in place (§6b.7, §6g.4).

### 6g.2 Sizing once: F1–F3, C-ws, C-pack **[cert: cert_footprint_monotone]** **[test: host_footprint_monotone]**

The guarantees and their consequences are stated in §6a ("Footprints across N and g"). The
proof:
- **F1.** Every §4a region is either independent of (N, g) or has N or g as a factor:
  RA = g·b·m·k̂, RB = g·b·n̂·k̂ and W = N·b·m·n̂ (W is last); sA, sB, fA, fB and vec depend on
  neither. align₂₅₆ is monotone, and the total is a sum of aligned regions in a fixed order.
  A packed role only drops sX, fX and RX (§6b.6). The statuses: the query checks §5 row 5 on
  the in-line layout (each region and the total, a packed role's regions included,
  `workspace_layout`), then each pack descriptor (its sizes, then κ ≥ T(N) under RN), then
  PACK_MISMATCH. Every size checked is nondecreasing in N and g (no factor is 0 on this path:
  b, m, n ≥ 1, k̂ ≥ 16, g ≥ 1), so the (N, g) past uint64 are closed upward. κ ≥ T(N) holds
  on a down-set of N, and nothing else depends on (N, g). If (N_max, g_max) succeeds, no
  smaller (N, g) passes uint64, so each either succeeds or is rejected by κ: the downward
  statement.
- **F2.** In the pack (§6b.4), res = N·b·r·k̂ + (⌈r⌉₁₆ − r)·k̂ is affine in N, and sig and
  flag do not contain N. The sizes are checked before κ, so INVALID_VALUE occurs on an up-set
  of N.
- **F3.** The first half is §6b.4. The read extent is §6b.5's ALIASING interval: K4 reads at
  most the last byte of res (in the B role, the end of the tail; `cert_pack_layout`), and K5
  reads σ and the flags. The second half, no byte beyond `pack_bytes`, holds if each pack
  kernel writes only regions of the layout: `kp_prep` the alignment gaps, the tail and the last
  gap, KL the σ, flag and plane rows (§6b.11). That was an inference [I] until
  `pack_write_extent` (sentinels around an exactly sized window) and
  `pack_write_extent_memcheck` (exactly sized allocations under memcheck) checked it on every
  `imd_pack` launch shape: the 42 KL shapes, among them the radix > 1 row mapping a
  batch-fastest X takes, plus `kp_prep`'s two instances and a gathered K (§6g.8). Only then did
  it become ABI (NOTES-C1 IR17).
- **C-ws.** The query succeeds at (N_max, g_max), so by F1's downward statement it succeeds
  at (N_max, g) and at every (N, g) that has a plan. There ws(N, g) ≤ ws(N_max, g) ≤
  ws(N_max, g_max) by F1, which applies because g ≤ N ≤ N_max (`options_ok`), and the
  contract checks accept a larger workspace (§6). The
  bytes a call at another N left behind are never read: every byte a call reads, that call
  wrote (Pipeline's junk-initialized workspace invariants `NoStaleRead` and
  `EveryRowByteWritten`). Each call uses its own plan's region offsets. (HostAPI's
  `neg_ws_shared` is about concurrent calls, not about reuse across plans.)
- **C-pack.** It follows from F2, F3's first half and the read extent: `imd_pack` at N′
  writes all of [0, `pack_bytes(N′)`), and a contract at N′ reads only that range. It needs
  neither the offsets (I1) nor F3's second half.
- **Outside F1–F3:** `imd_pack`'s own workspace. Its independence of N is I1 (§6g.3), so a
  caller who sizes it once for every phase takes the maximum of the per-phase `pack_ws_bytes`
  (the tests `packed_repack_n` and `workspace_shared_n` do).

### 6g.3 Internal facts: certificates, not ABI

These hold for today's formulas, but no caller needs them, and a future format may break
them: a K-weight region after res (NOTES-C1 stage 6), a slice format, a range plan. Under
§6a a published promise cannot be withdrawn, so they stay certificates that a format change
must revisit (`params.py cert_footprint_monotone`, checked on the library by
`host_footprint_monotone`):
- **I1.** In `imd_pack_layout`, the sig and flag offsets and bytes, the res offset and
  `pack_ws_bytes` do not depend on N, and res(N′) is a prefix of res(N_max).
- **I2.** The plan device bytes (`imd_plan_bytes`, `imd_lowering_packed`) do not depend on N
  or g. The host bytes do, through the K4 shape keys.
- **I3.** Packing more roles never increases ws(N, g). A memory budget that allows several
  pack modes takes the maximum over them from the queries, so no caller relies on I3.

### 6g.4 Re-packing one buffer

The rule is §6b.7's [v1.19] item. Its hazard, a stale handle with its plan reading a
re-packed buffer and returning a wrong C with SUCCESS, already existed for any re-pack with
another key; precision phases make re-packing routine. A device-side digest header in the
buffer is rejected (NOTES-C1 Q3). It would reverse §6b.10 #12 and §6b.4's "the bytes are a
pure function of (key, values)". Checking it needs a synchronization that kernel scope
forbids, and it could not protect graph replays anyway.

### 6g.5 Python

- `PackSpec.with_moduli(N)` is the same pack descriptor at another N (`dataclasses.replace`;
  the orders are kept).
- `PackedBuffer(buf, nbytes=None)` wraps a caller-owned packed buffer: a contiguous CUDA
  array, an (int pointer, nbytes) tuple, or an int pointer with `nbytes`.
  - `Pack.pack(X, PackedBuffer)` records the pack's `PackRecord` (N, rounding, `pack_bytes`,
    key digest) on the host at enqueue, under a lock. A call the library rejects keeps the
    previous record; a CUDA_ERROR clears it.
  - `Plan.contract` and `immerdink.contract` refuse a `PackedBuffer` whose record differs
    from the plan slot's pack, or that has none, with `PackMismatchError` ("buffer last
    packed at N=8, plan expects N=14"), before any C call. The error's `reason` stays None:
    it means the library's plan-time `IMD_PACK_MISMATCH_*` reason, and this refusal is a
    host-side comparison, so the message alone names the N, rounding or key difference.
  - `Pack.record` is the `PackRecord` of a pack's key: what a `PackedBuffer` records when
    that pack writes it, and what a plan made with that pack expects. `PackedBuffer.last_pack`
    is the record of the last successful pack into the buffer (None before the first one, or
    after a CUDA_ERROR).
  - It allocates nothing, and raw buffers behave as before.
- It is a safety net, not a soundness argument. The digest is the library's FNV-1a
  `key_digest`, for diagnostics only. The record follows host enqueue order, not GPU order:
  under graph replay, or across streams ordered by events, it can differ from what the GPU
  sees, so it guards the plain stream-ordered pattern only. `Pack.pack_ptr` and any other
  write bypass it. It also assumes one host thread per buffer: the lock covers the record,
  not the call, so a `Pack.pack` into the buffer from another thread between a contract's
  check and its enqueue is not seen (the call then reads the new bytes). A caller that packs
  and contracts on one buffer from several threads serializes them itself.
- Its lock-and-compare costs 0.14 µs per `Plan.contract` (`bench/python_overhead.py`,
  "ik,kj->ij" 16³, N = 8, median host time: 6.26 µs with a `PackedBuffer` against 6.12 µs
  with the raw buffer of the same packed plan; the bare ctypes call takes 3.98 µs; immerdink-eval
  `results/c1/stage0_python_overhead.json`).

### 6g.6 First call at a new N; capture (NOTES-C1 IR10)

**Measured** (`bench/first_call.py --sweep`, 2026-10-07, RTX 4090, quiet machine, under the
GPU lock). The case is "ik,kj->ij" with m = n = k = 1024 and 4096, RN, g = N, GENERATE
vectors, under `CUDA_MODULE_LOADING=LAZY` (the CUDA default). Each (size, N, mode) runs in a
fresh process that first plans and calls at another N, so the numbers are a switch, not process
start-up. Call times run from just before the enqueue to completion, the same measure for the
steady call (median of 20) and the first one. Records: immerdink-eval
`results/c1/stage0_first_call.jsonl` (`bench/first_call.py --table` prints this table from
them, with the GPU-event times alongside), summary `findings/c1-stage0-first-call.md`. The
library was built at 1e6e709 (host API 13); the summary gives its sha256. **Provenance limit:**
the 16 records carry only a date, no commit, library sha256 or host API. They were written
(05:04) by `bench/first_call.py` before its first commit (8b58101, 05:13), and that run is the
only one; the script's later changes (0197351, d22a686, and the final review's host-API
field) touch its table and provenance code only. So the build named above is the summary's
account of the library in place at the time, not a field of the records. **Re-run
2026-10-09** on commit 243d9c0 (`libimmerdink.so` `2c998dbc…`, `libimmerdink_host.so`
`35e6f89d…`, host API 13, dirty false; all 16 records name them; immerdink-eval
`results/c1/first_call_243d9c0.jsonl`): the first call at a never-launched N costs 11.7-37.5 µs
more enqueue time than a steady call and, from enqueue to completion, at most 7.4 µs more
(1024^3) and -0.4 ... +2.1 µs (4096^3), except one sample (4096^3, N = 14: +89 µs, 2.0 % of the
4.4 ms call; the design-phase run had about +9 µs; reported, not repeated). Steady / first
call: 1024^3 27.1 / 25.1 (N = 6), 23.6 / 22.2, 20.6 / 19.2, 15.5 / 15.0 TFLOPS-eq (0.079 /
0.086 ... 0.139 / 0.143 ms); 4096^3 61.2 / 61.2, 49.6 / 49.6, 41.4 / 41.4, 31.1 / 30.5
(2.245 ... 4.417 / 4.506 ms). First plan creation 0.011-0.157 ms; re-pack of A 0.018-0.023 ms
(1024^3) and 0.276-0.421 ms (4096^3); cold capture first / steady replay 1.00-1.10, so the
conclusions below stand, with no plan-time preload.

| m = n = k | N | steady call | first call at the new N | first plan creation | re-pack of A (GPU) | cold capture: first / steady replay |
|---:|---:|---|---|---:|---:|---:|
| 1024 | 6 | 27.1 TFLOPS-eq, 0.079 ms | 25.1 TFLOPS-eq, 0.086 ms | 0.141 ms | 0.018 ms | 1.10 |
| 1024 | 8 | 23.5 TFLOPS-eq, 0.091 ms | 22.3 TFLOPS-eq, 0.096 ms | 0.013 ms | 0.019 ms | 1.09 |
| 1024 | 10 | 20.6 TFLOPS-eq, 0.104 ms | 19.3 TFLOPS-eq, 0.112 ms | 0.150 ms | 0.020 ms | 1.08 |
| 1024 | 14 | 15.4 TFLOPS-eq, 0.139 ms | 14.9 TFLOPS-eq, 0.144 ms | 0.013 ms | 0.023 ms | 1.06 |
| 4096 | 6 | 61.2 TFLOPS-eq, 2.244 ms | 61.2 TFLOPS-eq, 2.244 ms | 0.012 ms | 0.275 ms | 1.00 |
| 4096 | 8 | 49.4 TFLOPS-eq, 2.781 ms | 49.4 TFLOPS-eq, 2.782 ms | 0.012 ms | 0.305 ms | 1.00 |
| 4096 | 10 | 41.2 TFLOPS-eq, 3.333 ms | 41.4 TFLOPS-eq, 3.320 ms | 0.011 ms | 0.344 ms | 0.98 |
| 4096 | 14 | 31.3 TFLOPS-eq, 4.395 ms | 31.2 TFLOPS-eq, 4.403 ms | 0.012 ms | 0.423 ms | 0.99 |

- **The switch is cheap.** The first call at a never-launched N spends 12–35 µs more host time
  enqueueing than a steady call (consistent with the lazy load of that N's kernels [I]); much
  of it overlaps the GPU (46–68 % of the extra enqueue time at 1024³, 75 % or more at
  4096³), and from enqueue to completion the first call takes at most 9 µs more (5–7 µs at
  1024³; at 4096³ between −13 and +9 µs, under 0.3 % of the call).
- **Plan creation is not part of the switch.** The first `imd_plan_create` at the new N takes
  0.011–0.150 ms of host time, against 0.004–0.005 ms afterwards: a one-time cost per N per
  process, paid when the plans are made before the iteration loop (§6g.1). The 0.14–0.15 ms
  cases (1024³, N = 6 and 10) probably load a K4 configuration the earlier N did not use,
  since plan creation reads the K4 kernels' attributes [I].
- **Re-packing costs more.** Re-packing one operand at the new N takes 0.018–0.023 ms of GPU
  time at 1024² and 0.28–0.42 ms at 4096², more than the first call's extra time (3–5× at
  1024³). It recurs at every phase switch (every iteration for an operand that changes),
  while the first call's extra time is paid once per N per process.
- **Cold capture works.** `graph_phases` captures phases whose N never ran in its process
  (bit-exact), and here the first replay of a cold capture is 0.98–1.10× the steady replay.
  NOTES-C1 IR10's bar (a capture failure, or a first replay above 2× the steady one) is not
  met, so **no plan-time preload is added**: `imd_plan_create` is unchanged.
- The process's own first call, at any N, is start-up rather than a switch: its enqueue takes
  0.8–1.9 ms, once per process.

### 6g.7 Spec

**No TLA+ change** (NOTES-C1 R1, IR20; the §6f.5 argument):
- The re-pack ordering is modelled. A re-pack is StreamSpec's post = "repack" with the WAR
  edge (`neg_stream_no_war`), and a contract after it is the pack → contract RAW case
  (`neg_stream_no_raw`).
- The stale handle is §6b.10 #3's documented limit. The library never compares bytes with
  handles, so a witness would only restate the caller rule.
- The layout prefix is a static formula (`pack_layout`): certificate I1, not a protocol
  property. F3's write extent is likewise checked by a GPU test with memcheck, not by a TLA+
  write-footprint invariant, which would only restate that formula.
- Pipeline, IndexMap and HostAPI do not change: per-N plans and handles are existing objects,
  and C-ws rests on Pipeline's existing invariants.
- The optional StreamSpec model with ⟨version, key⟩ cells (NOTES-C1 §4.1) is built only if
  the N-range plan is revived or Eric asks for it.

**[cert] `cert_footprint_monotone`:** F1, F2, C-ws, C-pack and I1–I3 on the §4a and §6b.4
formulas: every (N, g), every packed-role set, both vec modes, A = 16 and 64, a grid and
random shapes, and the per-region structure of the F1 proof. Mutations (a workspace that
falls at one N, a packed set that grows the workspace, a res offset that moves with N) fail
it.

### 6g.8 Tests

- **CPU.**
  - `host_footprint_monotone` (`tests/test_footprint.py`): F1, F2, C-ws, C-pack and I1–I3 on
    libimmerdink_host, every value equal to params. The grid is m, n ∈ {1, 15, 16, 17, 255,
    256, 257}, k ∈ {2, 96, 97, 1024, 1025, 131008, 131009}, b ∈ {1, 3}, N = 2..20, g = 0..N,
    both vec modes, the caller's packed roles none / A / B / AB, and the K alignment AUTO and
    A16. Then 10⁴ random legal contractions (role swaps, diagonals, permuted and negative
    strides; A = 16, 64 or AUTO). RN acceptance is NOT_SUPPORTED exactly where κ ≥ T(N).
    Then two "ik,kj->ij" shapes whose sizes pass uint64: m = 2^30, k = 2^31, n = 16 (RA from
    g = 8 on, and A's pack from N = 8 on) and m = n = 2^30, k = 2 (W from N = 16 on). They run
    at every (N, g), RN and TRUNC, both vec modes, every packed set. Every status equals
    params: INVALID_VALUE where a size passes uint64, then the packs' κ rule. At each g, the
    statuses over N are a NOT_SUPPORTED prefix, a SUCCESS interval and an INVALID_VALUE
    suffix; with A packed and RN at g = 0 that is NOT_SUPPORTED at N = 2..4, SUCCESS at 5..7
    and INVALID_VALUE at 8..20. C-ws's downward statement is checked on every pair of (N, g)
    points. On the grid and the random cases, it is checked step by step.
  - `python_device`, CPU part: `PackedBuffer` through a fake libimmerdink (each refusal
    happens before any C call; the record on success, rejection and CUDA_ERROR; `nbytes`),
    and `PackSpec.with_moduli`.
- **GPU** (`tests/test_phases.py`, one fresh process per suite, under the GPU lock, with the
  allocation shim):
  - `packed_repack_n`: X is packed at N = 14 into a buffer of `pack_bytes(14)` and
    contracted, then re-packed at N = 8 in place and contracted with the N = 8 plan; both
    bit-exact, every packed byte equal to `oracle.pack`. Bytes above `pack_bytes(8)` keep a
    sentinel. The stale N = 14 plan with the `PackedBuffer` raises `PackMismatchError` with
    no launch; with the raw buffer it returns SUCCESS and a wrong C (the documented limit).
    Then back to N = 14. Both post-swap roles (B with tail rows), each case's role asserted.
  - `workspace_shared_n`: one workspace of `imd_workspace_size(N = 14, g = 0)` serves plans
    at (N, g) ∈ {5, 8, 14} × several g, both vec modes, called alternately on one stream
    (KF; KL with the role swap). Then the same with A re-packed before each call into one
    buffer of `pack_bytes(14)`, and one `imd_pack` workspace of the maximum per-N
    `pack_ws_bytes`. All bit-exact; no allocation; the guard zones untouched.
  - `graph_phases`: one graph per phase, N = 6 (warm) and N = 10 (**cold**: captured before
    any launch at N = 10 in the process, under `CUDA_MODULE_LOADING=LAZY`), sharing one
    workspace and replayed alternately on new data, bit-exact. A cold packed phase too:
    `imd_pack` and `imd_contract_packed` at N = 12, never launched, in one graph. The script
    refuses to run it after another suite in the same process.
  - `pack_write_extent` (F3's evidence): 113 packs. The suite computes each pack's KL launch
    from the rule (`pack_pivot`, `pack_kl_lg_rows_for`, `kl_launch`; its transcription of
    `pack_pivot` is checked against the library's on every case's plan view, and its input,
    X's fused lists, against the library's through the pack key's digest, which
    `imd_pack_layout` reports equal to the oracle's on every case) and fails unless
    the cases cover all 42 shapes: along rows, tiles of 64, 32 and 16 rows, and of 8 rows with
    one or two chunks on chip or with re-read chunks, each with radix 1 and radix > 1
    (a batch-fastest X; two row modes); along k̄, tiles of 64 down to 1 row, the 1-row tile
    with one, two or more chunks; each with W = 2 and 3. They also cover `kp_prep`'s
    nine-mode instance (an X with nine fused modes), a gathered K (two K modes X cannot fuse),
    and both post-swap roles. RN and TRUNC, A = 16 and 64 (where k > 96 allows it),
    r ∈ {1, 15, 16, 17, 33} (12, 35 and 36 in the multi-mode cases), b ∈ {1, 3, 5, 12},
    N ∈ {2, 3, 8, 14, 17, 20}. Each pack is written twice into an exactly sized window of a
    buffer filled with a sentinel (0xA5, then 0x5A). Every byte outside the window keeps the
    sentinel, every byte inside is the same both times (all written), and small packs (72)
    equal `oracle.pack`; the same holds for `imd_pack`'s workspace. Each pack, in an exactly
    sized allocation, is then read by `imd_contract_packed` with the in-line partner on every
    producer path (KF or KL by rule, K1 + K2/K3, KL, KL with 64-row tiles; every 12th case
    also under each of the four K4 configurations): 612 packed contracts, each bit-identical
    to the plain `imd_contract`, which must change C and (77 small cases) equal the oracle.
  - `pack_write_extent_memcheck` (`tests/memcheck_pack_extent.sh`): the same 113 packs and
    612 contracts, with the same coverage assertion, every allocation exactly sized, under
    `compute-sanitizer --tool memcheck --padding 4096`: clean. A pack allocation 16 bytes
    short (`imd_pack` told `pack_bytes`) must be reported as an invalid global write, which
    shows the check fires.

### 6g.9 Risks and limits

- The re-pack rule, and not replaying a graph after a re-pack with another key, are caller
  obligations the library cannot check (§6b.7). `PackedBuffer` covers only stream-ordered
  host code.
- Plans per N multiply the RESIDENT vector bytes, 8(2m + 2n + 2k + 3b) per plan, by the
  number of levels (GENERATE plans hold none), and with `k4_tune` on, the plan-time
  autotuning.
- A future pack format with offsets that depend on N would break I1, and the certificate
  would catch it. C-pack does not depend on I1.
- The gain depends on the final N the consumer needs and on the shape (NOTES-C1 §2.3–§2.4,
  §4.10). Gains at N ∉ {8, 10, 14, 16} rest on the performance model outside its hold-out
  until NOTES-C1's M-c passes.
  - M-c ran on 2026-10-07 and **failed**: it missed §6c.5's < 100 µs median (15.8 % against
    15 %; a repeat gives 16.1 %). The binding verdicts are NOTES-C1 §5.9; the details are in
    NOTES-C1 Status ("M-c").
  - Two effects set the missed median, and each is necessary. One is a small-N optimism on
    the KF path: sq1024 at N = 4 is 16–21 points more optimistic than the mean of its
    anchors' errors in the first run, 15–16 in the repeat (8.2 / 15.4 points against the
    N = 8 anchor alone, repeat 12.8 / 6.8, since its error rises with N), and those two
    calls are the median.
    The other is the KL path with a narrow n: the Jacobi view is predicted 26–29 % fast at
    every N ≤ 10, the anchor N = 8 included, and −14 to −15 % at N = 12–14.
  - The Jacobi error is two KL effects, which cancel at the call total against the COLD
    prediction (NOTES-C1 Status "Cause", §5.9 MC-3). KL A, the wide operand's pack, keeps
    only 27–50 % of its modeled REPEAT credit (§6c.1 "Cache state"). KL B, the narrow
    operand's pack, takes ×1.5–2.0 its prediction in every cache state: a KL path error,
    not a cache effect.
  - The Q24 refit was not adopted (NOTES-C1 §5.9 MC-1), so the marks stay, with no partial
    pass (MC-2). The path to a pass is the MC-3 feature(s) as amended (NOTES-C1 §5.9 MC-3,
    MC-3a): §6c text from a per-phase account, at most two optional keys (the REPEAT credit;
    the KL narrow-pack time) fitted against the phase marks, a fresh narrow-KL hold-out, the
    v1.18 hold-out judged with and without its 7 cases that share a GEMM view with a fit
    case, and a fresh M-c run. M-d2 waited for it.
  - **MC-3a ran on 2026-10-09 (NOTES-C1 §5.11; §6c.9).** One key (`warm.kl_drop`, fitted 0.70)
    was refitted once with Q24; every §6c.5 class target is met on the fresh narrow-KL
    hold-out (4.3 / 15.7 and 2.7 / 6.6 %), the v1.18 hold-out on both sets, `mc.jsonl` with and
    without the repeat and a fresh full M-c run (1.8 / 7.6, 1.8 / 7.4, 3.3 / 13.2 %). One
    pre-registered in-sample condition fails under its literal reading (two K4 entries,
    +1.6 and +1.0 points, from Q24's records), so no pass is declared: M-c stays failed, the
    marks stay, the shipped calibration is unchanged, and MC-3 is closed without a
    calibration change. Independent review (2026-10-09, binding, ruling A, NOTES-C1 §5.11): the
    in-sample condition is read against the shipped fit's table, so MC-3a does not pass; the
    shipped calibration stays, `mc3a.calib` stays in immerdink-eval, `warm.kl_drop` stays in
    the library and the fitter, and no second key is tried. C1's done condition
    ("M-c has passed") was amended by Eric's decision of 2026-10-09, after M-d2's pass (C1 is
    done as amended; M-c stays failed and the marks stay).
  - Indicative, not M-d2, and never used to set its bars: the schedule ending at N = 14,
    computed from M-c's single-call medians, is 0.003–0.023 above its [M] ratio on 2000³,
    4096³, DF, QZ and the PPL view, and 0.005–0.028 below on 1024³ and skinny. On the Jacobi
    view it is 0.044–0.047 above (0.830 against 0.786).
  - **M-d2 ran on 2026-10-09 and PASSES** (NOTES-C1 §5.12; immerdink-eval `findings/c1-m-d2.md`;
    commit 243d9c0, driver `dbbbfe18…`, `libimmerdink.so` `2c998dbc…`, host API 13). The
    schedule 5 x N6 + 4 x N8 + 4 x N10 + 3 x N14 against fixed N = 14 on the shipped
    calibration, normal / decay_opp (bar, band; TFLOPS-eq first, time alongside): PPL view
    0.7693 / 0.7688 (0.750, 0.6375-0.8000; S14 6.0 TFLOPS-eq, 45.2 ms against F14 4.6, 58.8
    ms), DF 0.6948 / 0.6977 (0.664, 0.5644-0.7636; 39.3, 27.7 ms against 27.3-27.4, 39.8-39.9
    ms), 4096^3 0.6957 / 0.6915 (0.673, 0.5721-0.7740; 43.9-44.2, 49.8-50.1 ms against 30.5,
    72.0 ms); the Jacobi view (reported) 0.8290 / 0.8278. All eight first records are
    admissible and bit-exact; every ratio is 0.019-0.034 above its prediction. This is a time
    result: M-c stays failed and the marks stay. M-d2 waited for the MC-3 path (NOTES-C1 §5.9 "M-d2
    ordering", binding; upheld in review, §5.10 MD2-0: MC-3a first, closed without a
    calibration change only as MC-3a's recorded outcome or by Eric's decision); MC-3a's
    recorded outcome (NOTES-C1 §5.11) closes it, so M-d2 may run on its pre-registered bars.
    - Its protocol and driver are pre-registered: `bench/accuracy/md2_time.py`,
      re-registered after two reviews at 35eeb46 and, with the NaN-filled bit-exact check, at
      fce29e9 (sha256 `dbbbfe18…`; NOTES-C1 §4.3).
    - The review (approve-with-fixes; NOTES-C1 §5.10, binding) upheld MD2-0 to MD2-7.
      The fixes:
      - the judge decides per system *and* data set, by rules (a)–(d): contaminated or
        not bit-exact records are inadmissible, and a bit-exact failure stops M-d2; one
        re-run at most; the first admissible record decides;
      - provenance checks: a clean tree, the driver's sha256, the fixed protocol
        parameters, the bars' configuration per record, and the library's embedded
        calibration equal to the registered file.
    - The re-review (approve-with-fixes; NOTES-C1 §5.10 "Re-review", binding) closed the
      paths outside one records file: one normative records file, every M-d2 record in any
      file reported; the driver frozen from the first M-d2 record on (MD2-8 amended);
      nothing continues after two inadmissible attempts, STOPPED or a refused file without
      a recorded reviewer verdict or Eric's decision. Also: one library per records file,
      the registered bar re-checked on the loaded library, `--md2-cleared` citing the
      closure's commit and section, and the selftest as ctest `md2_time_selftest`.
    - On the shipped calibration, its `predict` reproduces the bars: 0.750 (the PPL view),
      0.664 (DF) and 0.673 (4096³), with bands of ± 15 % capped at 0.8.
    - [M, model-only, outside hold-out] (NOTES-C1 §5.10 MD2-4): with the PPL view's
      invariant B packed per phase, the schedule is predicted at 0.876 of fixed N = 14
      packed, against 0.750 in-line. The re-packs of the 680 MB operand compete with packed
      calls of 0.67–1.59 ms. Like for like, packing B gives P14/S14 = 0.509 at the same
      schedule and PF14/F14 = 0.436 at fixed N = 14 (§6g.12, caller note).
  - NOTES-C1's M-b re-ran these [M] figures on this worktree's library (host API 13): every
    design-phase record reproduces byte for byte, and `PrecisionPlan`'s own choices give the
    same schedule ratios (immerdink-eval `findings/c1-m-b.md`).
- `PrecisionPlan` (§6g.10) cannot express a self-contraction that reads one share group's
  single pack through both roles: a role is `"A"` or `"B"`, and a member listed twice is
  rejected. The DF-CCSD self-contractions R1, R2, hh and Lld1 need it (equal A and B pack
  keys at every level 6–14; NOTES-C1 M-b). The candidate extension is a member role `"A=B"`
  (both roles read the group's pack, its cost counted once; not the pack mode `"AB"`, which
  means two separate packs). **Trigger** (ROADMAP C1): the ehrenfest-side integration of the
  B2 performance model declaring DF-factor sharing.

### 6g.10 Choosing N per phase: `PrecisionPlan` (stage 1) **[test: host_precision]**

C1(b) on the host. A thin helper composes the §6f queries per phase: the target → N rule of
`imd_err_min_moduli`, then one rank over the allowed levels. **No ABI change** (host API
13; NOTES-C1 §4.9). The policy stays with the caller: residual → target is caller code
(`bench/accuracy/schedule.py` is an example), and immerdink never changes N on its own.

```python
from immerdink import planner, precision
from immerdink.host import Accuracy
pp = precision.PrecisionPlan(planner.Planner(), contractions, levels=(6, 8, 10, 12, 14),
                             rounding="RN", mode="EMULATE", share=(), packs=None, reuse=None,
                             budget=0, monotone=True, vec_modes=("GENERATE",), perf=None)
# contractions: {name: (spec, extents[, strides][, calls_per_iteration])},
#               or {name: dict(spec=, extents=, strides=, calls=, beta=)} (beta: C is read)
pp.configs(iterations=None)   # {name: {level: Choice or None}}: create these plans first
ph = pp.phase({name: Accuracy(...)}, iterations=None, fail_budget=None)
ph.N, ph.g, ph.vec_mode, ph.mode, ph.packed, ph.status    # dicts by name (N: the level)
ph.pack_bytes, ph.reuse, ph.assumed_reuse, ph.err_bound, ph.err_typical
ph.time_s, ph.tflops_eq       # one iteration, calls-weighted (TFLOPS-eq = sum 2bmnk / time)
ph.choices[name].options()    # Plan keywords; .pack_specs: the packed roles' PackSpecs
pp.footprints()               # exact bytes per level and their maxima
pp.reset()                    # forget the monotone floor (a new solve, geometry or restart)
```

- **Placement.** `python/immerdink/precision.py`, not planner.py. `PrecisionPlan` computes in
  Python (levels, the monotone floor, share groups, the per-role pack cost), and §6e.4's
  "nothing is computed in Python" covers planner.py, whose hot-path queries are unchanged.
  It calls only a `planner.Planner`'s queries (`min_moduli`, `rank`, `pack_desc_for`,
  `predict_pack`, `predict` for a band, `workspace_size`, `lowering`) and `host.pack_layout`:
  no GPU, no data. The Planner's calibration and memo are used; `perf` sets the regime, the
  cache state and the clock of the rank and pack predictions.
- **Levels.** Increasing N in 2..20; 3–5 levels are recommended, since every level is a plan
  (and a pack) per contraction. `levels` may end with the sentinel `"DIRECT"` (NOTES-C1
  IR15): native FP64 when no emulated level meets a target. It is not in the default:
  DIRECT runs at 1:64 on the 4090.
- **Per-role pack cost** (NOTES-C1 IR2). Rank takes one int32 `reuse` for every packed
  operand, but a contraction's two roles can be reused differently (an invariant integral
  against amplitudes remade every iteration). So `phase()` makes one rank call per
  contraction with reuse = 1 and every candidate, and re-keys each one:
  `time_s = contract_s + Σ_{packed roles} pack_s(role, N) / reuse(role)`, added in the order
  A, B, with `pack_s(role, N)` = `predict_pack` of `pack_desc_for(role)` at N, memoized per
  (contraction, role, N). The re-keying is exact: a candidate's `time_s` is its only
  reuse-dependent field (its `pack_s` is the sum of the two roles' predictions, which
  `host_precision` checks), and feasibility (status, budget, accuracy) is the candidate's,
  unchanged. Within a level, choices are ordered feasible first by the composed time, then
  by rank's own tie order (total_bytes, N, g, vec mode, packed). Between levels of equal
  composed time (one contraction's, or a share group's sum) the smaller N wins, the
  least-memory choice by F1; no test pins it, since the model has produced no such exact
  tie. The band `time_lo_s` / `time_hi_s` is
  composed the same way from the contract's prediction. A per-role reuse inside
  `imd_perf_rank_request_t` is not adopted; it is revisited only if a hot-path consumer needs
  it inside one C rank call.
- **Reuse per role**, `reuse={name: {"A" | "B": kind}}`: `"call"` = 1, `"iteration"` (the
  default: the pack is remade every iteration) = the calls per iteration, `"phase"` = those ×
  `phase()`'s `iterations` (the phase length; without it, as `"iteration"`, conservative, and
  `Phase.assumed_reuse` records it), or an int. An invariant operand is re-packed at each
  phase switch, so `"phase"` is its value.
- **The phase rule**, per contraction c (NOTES-C1 §4.2):
  1. n_c = `min_moduli(target, levels[0] .. the top emulated level)`. A lowering error
     raises. The RN κ ≥ T(N) levels are +∞ in the estimate, so they are skipped. If no
     emulated level meets the target: DIRECT if listed and its bound (γ_k, §6f.2) meets it,
     status `DIRECT`; otherwise the top emulated level, status `MISSED` (its candidates are
     ranked with the target, so their err fields are in its metric, and chosen by status and
     budget).
  2. The floor f_c is the smallest level ≥ n_c.
  3. With `monotone` (the default), f_c = max(f_c, the previous phase's final level of c),
     DIRECT being the top. `reset()` clears it.
  4. Outside a share group: the best composed candidate over the levels ≥ f_c (all g, the
     vec modes, the allowed pack modes, the target, the budget). This equals the fastest
     `configs()[c][N]` over N ≥ f_c (ties: the smaller N), also where time is not linear in
     N (§2.3 of NOTES-C1); it replaces §6f.3's `rank([n, n + 1])`, the two-level case.
  5. A share group (below) is ranked jointly; a member whose floor is DIRECT leaves it for
     the phase.
  6. **BUDGET** (NOTES-C1 IR8): if no level ≥ the floor fits the budget, the floor level
     and its SUCCESS candidate with the least total_bytes, status `BUDGET` if that candidate
     meets the target, otherwise `MISSED`. `MISSED` takes precedence over `BUDGET`, and
     `BUDGET` over `DIRECT`; otherwise the status is `MET`. (The rule never puts a member
     at a level that misses a met target, so the guard is a safety net; `host_precision`
     exercises it directly.)
  7. **TYPICAL** targets: p in §6f.2 is per call. With `iterations` and `fail_budget` P,
     every TYPICAL target of the phase gets p = P / (iterations × Σ_c calls per iteration):
     a union bound over the phase's calls. Without them, each Accuracy's own `fail_prob`.
     BOUND is the default basis (NOTES-C1 Q5); TYPICAL for intermediate phases is an explicit
     opt-in.
- **Pack-mode masks and the budget** (NOTES-C1 IR8). `packs={name: modes}` (any of `"NONE"`,
  `"A"`, `"B"`, `"AB"`; default in-line) are ranked at every level, so the packing may differ
  by N: a large DF block can stay in-line at N = 14 and be packed at lower N. `budget` keeps
  rank's meaning, immerdink-owned bytes per contraction (workspace + plan + packs + pack
  workspace). The caller subtracts its own resident tensors, the FP64 source of every packed
  operand included (§6b.10 #4); a joint budget over several contractions is caller policy,
  for which `footprints()` reports per-level totals.
- **Share groups** (NOTES-C1 IR24), `share=[(group, members)]`: contractions that read one
  packed operand use one N. A member is `(name, "A" | "B")`, the role that reads the group's
  pack, or a bare name (equal N only). A shared role is always packed (the masks apply to the
  other roles), and its pack cost is divided by the calls of every member that reads it
  (`"iteration"`: their summed calls per iteration; an int: the total); the members' reuse
  kinds for it must agree. The group floor is the members' maximum f_c. For each level
  N ≥ that floor, the group time is Σ_members calls × the member's best composed time at N;
  the least wins (ties: the smaller N) for every member. Time is usually monotone in N, so
  this normally returns the floor, but not when the packing and reuse terms differ by level.
  Groups that share a contraction are merged for N. The shared roles must read one pack:
  `PrecisionPlan` checks at construction that their pack keys (`imd_pack_layout`'s N,
  rounding, pack_bytes and key digest, as `PackRecord` compares them) are equal at every
  level where the members have a plan, and that some level can pack them, and raises
  ValueError otherwise. (The final review of C1/C2 found a group whose members read different
  operands accepted and priced as one pack; `footprints()` then sized one buffer where two
  were needed.) Plan creation's PACK_MISMATCH stays the backstop for a caller's own `Pack`.
  *What the raise costs* [M, model-only, outside hold-out at levels ∉ {8, 10, 14, 16}]
  (NOTES-C1 §3.3, M-b: ehrenfest ccsd.py's pairwise DF contractions at psi4 sizes): nothing
  in 11 of 12 forcing-rule cases on the real operands, whose group members share a floor, and
  +16.5 % per iteration in the twelfth, raised through a contraction in two groups, on the
  default levels (6, 8, 10, 12, 14); +8.1 % in the twelfth on the integer levels 6–14 or on
  6/8/9/10/14. With one member's floor a level (ΔN = 2) from the rest's, from −32 % (the
  second pack of a factor re-packed every iteration costs more than the raise) to +25 % of the
  separate-packs time per iteration (the raised member dominates the group); a caller can
  split such a group. The joint choice was the group floor in every case measured. Limits: a
  contraction that reads one factor through both roles (R1, R2, hh, Lld1) has its B role
  in-line, since a share-group member has one role (NOTES-C1 Status, open item), and the norm
  products are the bare (t1 = 0) factors'. These are not what a slice mode could save (§7).
  **A member at DIRECT** (its floor, step 1 or 3) reads no pack, so it leaves the group's N
  coupling for that phase (independent review of stage 1, MF1): it is settled alone, as
  outside a group (in-line, g = 1, status `DIRECT`, `BUDGET` or `MISSED`), and the other
  members are ranked jointly over the emulated levels ≥ their maximum f_c. It links nothing
  in that phase either: groups merged only through it are ranked separately. Before this
  fix, every member was dragged to DIRECT, and a member whose target DIRECT's γ_k misses
  (a TYPICAL target met at N = 12) was reported `BUDGET` at native FP64 (1:64). The shared
  role's reuse is fixed at construction and keeps counting the DIRECT member's calls, so every
  phase's choice remains the `configs()` lookup bit for bit. In that phase the other members'
  pack term is then priced as if those calls read the pack: an optimistic [M] figure in a rare
  case (NOTES-C1 §5.5 S1-10). *Binding (re-review):* the reuse stays fixed. Counting only
  the phase's readers leaves each level's configuration unchanged in exact arithmetic (the
  shared term is one constant within a level), but in floating point it could reorder
  near-ties and pick a configuration whose plan was never created, a hard failure in the
  loop (graph capture, a frozen `PlanCache`). The optimism is at most pack_s × (the DIRECT
  members' summed calls / the summed calls) per iteration, in a phase that already runs a
  native FP64 (1:64) contraction, which dominates its time; numerics, budgets and footprints
  are unaffected. The same holds for the shared role staying packed when one member alone
  reads it in that phase. `Phase.reuse` reports the fixed figure (e.g. {B: 4} where one
  call reads the pack). To revisit only if a consumer runs share groups with DIRECT members
  in practice: then take each level's configuration from `configs()` and use the actual
  readers only for the level comparison and the reported time.
- **`configs(iterations)`** returns the best configuration at every level (status `OK`, or
  `BUDGET` with the least total_bytes; None where no plan exists, e.g. RN with κ ≥ T(N)). A
  fixed N's estimate is the same for every candidate, so it needs no target, and a phase's
  choice at a level is this one (for the same `iterations`). The caller creates every one of
  these plans before the iteration loop and freezes its `PlanCache` for capture (§6g.1):
  RESIDENT plan creation synchronizes, and with `k4_tune` each level's plans autotune.
- **`footprints()`**: per contraction and level, the maximum workspace and plan device bytes
  over the allowed pack modes × vec modes × every g that succeeds. For the workspace alone,
  the largest succeeding g would do by F1's downward statement (ABI, §6a), but at that g the
  plan device bytes would rest on I2, a certificate and not ABI (§6g.3); so every g is
  queried, a one-off sizing call (about N more lowerings per level × mode × vec mode). Per
  packed operand its `pack_bytes` and `imd_pack` workspace per level: a shared pack counts
  once, keyed by its group, while `Phase.pack_bytes` repeats it under every member that
  reads it. Each one's maximum (sizing once, C-ws and C-pack, NOTES-C1 Q2) and the per-level
  totals (no workspace shared). None marks a level without a configuration. No caller has
  to rely on I3 (§6g.3): the maxima come from the queries over the allowed modes.
- **Relaxed Krylov methods** (NOTES-C1 IR3) may loosen the matvec precision as the residual
  falls [literature: Bouras–Frayssé; Simoncini–Szyld; van den Eshof–Sleijpen]. They use
  `monotone=False` with a decreasing schedule (a matvec tolerance ∝ tol / ‖r_k‖, scaled by
  the expected iteration count). Fixed-point, Newton-type and residual-driven iterations,
  iterative refinement included, tighten: `monotone=True`, the default. What to do at a
  switch (reset DIIS, recompute a subspace's products, start Krylov tight) is the M-d1
  caller guidance in §6g.12.
- **The forcing rule, as caller code** (`bench/accuracy/schedule.py`): after iteration t,
  each contraction c gets an ABSOLUTE target η·r_t·w_c (Σ w_c = 1), with `norm_product_c`
  refreshed per phase; N rises only at iteration boundaries; BOUND for the final phase and
  TYPICAL, with p = P / C, for intermediate ones if the caller accepts it.
- **Cost.** Per contraction per phase switch: one `min_moduli` (about 1 µs), one rank over
  the levels ≥ the floor (C cost about one prediction per candidate), the memoized pack
  predictions, and the Python re-keying of every candidate. Measured on one core of the
  i9-13900K (three contractions: the CC ring of §6e.1 at v = 64, o = 32, the DF build
  acQ,bdQ→acbd at 64⁴ with Q = 2032, and 4096³; levels 6–14; NORMWISE targets 1e−8, 1e−10,
  1e−12; median of 7): `phase()` takes 0.31–0.34 ms in-line with one vec mode and 3.4–3.8 ms
  with every pack mode and both vec modes; `configs()` takes 0.48–6.1 ms. A re-run on
  51dba1e, after the stage-1 review fixes, reproduces these figures to within 0.04 ms
  (immerdink-eval `findings/c1-stage1-phase-cost.md`: the committed script, both records
  with their sha256, the product commits and the library's sha256). It runs at phase
  switches, off the hot path; ehrenfest's hot-path rank calls are untouched.
  It is not free per call, and a loop whose targets change every iteration should not call it
  every iteration (`bench/accuracy/schedule.py` does, as an example: 0.2–0.7 ms of host time
  per call on its two contractions, a new target each time, median 0.6 ms while the targets
  move the floors and 0.2 ms once they sit at the top level, one call in 144 at 1.4 ms;
  against its 5.8 ms [M] iteration; one core, c22ebcb, immerdink-eval
  `results/c1/final_review/phase_call_cost.txt`). A switch is due only when a floor moves:
  `phase()` starts from one `min_moduli` per contraction (steps 1–2; about 2.5 µs each
  through a `planner.Query`), and its configurations stay the same while those
  return the same status and `num_moduli` (with `"DIRECT"` listed, also the mode-DIRECT
  check at the top level), for the same `iterations` and `fail_budget` (TYPICAL targets
  carrying step 7's p); only the err fields follow the targets. So the caller can run those
  queries every iteration and `phase()` only when one of them changes.
- **Limits.** The choice is the performance model's [M], outside its hold-out at
  N ∉ {8, 10, 14, 16} until NOTES-C1's M-c passes (it failed on 2026-10-07, §6g.9; MC-3a ran once
  on 2026-10-09 without a declared pass, NOTES-C1 §5.11). Rank's footprint tie order
  (`TIES_BY_FOOTPRINT`) is not applied to the composed times. A phase names every
  contraction.

### 6g.11 The rigorous ρ from a previous output: `rho_from_output` **[cert: cert_rho_recipe]** **[test: host_precision]**

An ELEMENTWISE target needs `cond` = ρ ≥ max ‖a_i‖‖b_j‖ / |x_ij| over the outputs that must
meet it (§6f.1), and ρ needs |x|. The rigorous recipe takes it from a previous call's output
(NOTES-C1 IR7); it ships as a tested helper, not as documentation:

```python
rho_hat, certified = precision.rho_from_output(C, row_norm_ub, col_norm_ub, *, N, rounding,
                                               k, alpha=1.0, beta=0.0, l1_ratio=None,
                                               mode="EMULATE")
cond = rho_hat[certified].max()     # over the outputs that must meet the target
```

- **Inputs.** A call at (N, rounding, k, mode) with α = 2^e and β = 0 wrote C = RN(2^e D)
  (§3.7), D its product. `C` is a float64 host array (copy it, or the part needed, to the
  host: the helper moves no data) in the GEMM view (…, m, n) (§1.1). A tensor output whose
  modes interleave A's and B's free modes, such as the CC ring's abij of
  `"kjbc,kiac->abij"` (§6e.1, §6g.10's cost figures, the `precision.py` module example), is
  permuted to (batch modes, A's free modes, B's free modes) and reshaped first, its bounds
  in the same order: the helper pairs bounds with outputs by position only. `row_norm_ub`,
  `col_norm_ub` are upper bounds of the GEMM view's row and column 2-norms (§6f.1's
  obligation: really upper bounds), as FP values: float arrays (float16 and float32 widen to
  float64 exactly) or integers of magnitude ≤ 2⁵³. A bound that conversion to float64 would
  round to nearest, possibly below the caller's bound, is refused: an integer above 2⁵³
  (ValueError), a long double, an object array (Fractions, Decimals, Python ints beyond 64
  bits), complex or bool (TypeError). numpy converts a Python list before the check, so an
  integer above 2⁵³ mixed with floats in a list is already rounded: pass numpy float arrays.
  A row bound lies along axis −2 and a column bound along axis −1:
  - **C of shape (m, n):** a scalar; a 1-D `row_norm_ub` of length m and a 1-D
    `col_norm_ub` of length n, each on its own axis (another length, 1 included, is a
    ValueError); a row bound (m, 1) and a column bound (1, n).
  - **A batched C (…, m, n), C.ndim ≥ 3:** a scalar, or a bound with exactly C.ndim dims, a
    row bound (…, m, 1) and a column bound (…, 1, n) (numpy's `keepdims=True`), each batch
    dim C's or 1 (shared over that batch axis): per-batch norms
    `np.linalg.norm(A, axis=-1, keepdims=True)`, one value per batch (b, 1, 1), one vector
    shared by every batch `r[None, :, None]` / `c[None, None, :]` (at C.ndim = 3). A 1-D or
    2-D bound on a batched C is a ValueError, and so is a bound of any other ndim: the error
    names the keepdims spelling.
  - Anything else is a ValueError too: a transposed (1, m) row or (m, 1) column bound, an
    elementwise (m, n) one.
  For a 0-D or 1-D C the bounds broadcast elementwise. The rule is structural because numpy's
  trailing-axis broadcasting pairs a bound with the wrong axis whenever two extents are equal,
  and certified ρ̂ far below ρ each time: `row_norm_ub[j]` with column j at m = n (1-D bounds
  on a batched C, or a 1-D row bound with a (1, n) column bound; independent review of stage
  1, MF2: ρ̂ ≈ 1 where ρ ≥ 100); numpy's per-batch norms without keepdims, a (b, n) column
  bound at b = m and a (b, m) row bound at b = m = n (re-review, MF3: ρ̂/ρ ≈ 0.006); and
  per-batch scalars without keepdims, a (b,) row bound at b = m, a (b,) column bound at
  b = n and a (b, 1) row bound at b = m (second re-review, MF4: ρ̂/ρ = 0.0081–0.0113). On a
  batched C no 1-D or 2-D bound can be told apart from a per-batch one where b equals m or
  n, so only scalars and full-ndim bounds pass. Real norms, not σ-derived ones: a zero row
  has σ := 0, so its bound d·2⁰ is vacuous. `l1_ratio`: as `Accuracy`'s (ν = √k without
  it).
- **Restrictions.** β must be 0 and α = 2^e, else ValueError: D = ldexp(C, −e) is exact,
  while D cannot be recovered from RN(α·D) for another α (§3.7). For β = 1 accumulations (CC
  residual builders) take ρ from a probe call with β = 0 into a scratch buffer, once per
  phase.
- **U** is the NORMWISE `bound` of `imd_err_estimate` at (N, rounding, k) for the call's
  dispatch (EMULATE; DIRECT for mode DIRECT or k = 1), the library's own value: E·(1 + u) + u
  evaluated as E + E·u + u, times (1 + 2⁻⁴⁰) (§6f.3), so |D − x| ≤ U‖a_i‖‖b_j‖ for D finite
  and normal (PROOFS L3c). For DIRECT it is γ_k (§6f.2), which exceeds the EMULATE bound at
  large N, so the EMULATE U would be unsound there (hence `mode=`). `host_precision` checks
  U ≥ params' exact bound (the upper end of its enclosure) at N = 2–20 for both dispatches,
  and `cert_rho_recipe` checks the recipe on the native FMA chain as well as on emulated
  elements.
- **Claim.** For D_ij finite and normal with |D_ij| > U·n_ij, n_ij ≥ ‖a_i‖‖b_j‖:
  ρ_ij ≤ n_ij / (|D_ij| − U·n_ij), because |x| ≥ |D| − |D − x| ≥ |D| − U·n.
- **FP64 evaluation, ρ̂ rounded outward:** D = ldexp(C, −e); n = nextafter(row × col, +∞);
  t = nextafter(U·n, +∞); den = nextafter(|D| − t, 0); ρ̂ = nextafter(n / den, +∞). Each RN
  result lies within half an ulp, also in the subnormal range, so one ulp outward bounds it.
  An output is **certified** iff C and D are finite, D is normal (|D| ≥ 2⁻¹⁰²²), for e < 0
  also |C| > 2⁻¹⁰²² (a C of exactly 2⁻¹⁰²² can be a subnormal 2^e·D rounded up), den > 0 and
  ρ̂ is finite. The rest (zero, subnormal or non-finite outputs, |D| ≤ U·n, overflows) get
  ρ̂ = +∞ and `certified` False; the caller decides whether they must meet the target. With
  true norm bounds, ρ̂ ≥ 1. The evaluation costs 2 ulps: |D| at least 2 ulps above RN(U·n)
  is certified (`cert_rho_recipe`).
- **Scope.** ρ̂ is rigorous for that call's data and a prediction for the next iteration's;
  NOTES-C1 M-d1 measures how often the lagged ρ misses. Per output, ρ̂ is also an
  a-posteriori elementwise bound for that call, and it locates the rows for output-selective
  refinement (NOTES-C1 §3.1 B6, a caller recipe).
- **Cost** [E]: one pass over C, plus the caller's norms, a pass over A and B unless it has
  them (about 2× a pass over C at square shapes): about 9 % of a 4096³ N = 14 call when every
  norm is recomputed. The norms of an invariant operand are computed once per phase, and the
  ρ that sets N is needed once per phase decision. A stage-5 statistics hook in K5 (deferred,
  NOTES-C1 §4.6) would need no norms at all (L6 uses ‖a·2^σ‖ ≤ d).
- spec/params.py `rho_recipe` is the scalar reference; the helper is its numpy version, bit
  for bit (`host_precision`).

### 6g.12 Stage 1: tests, certificate, spec

- **`host_precision`** (CPU, `tests/test_precision.py`, in `tools/test.sh`'s CPU-only pass;
  needs numpy):
  - `phase()` equals a brute force from host.py's queries on random plans (levels, masks,
    reuse kinds, budgets, vec modes, share groups, three-phase target sequences in every
    metric and basis): every level, g, vec mode and allowed pack mode enumerated with
    `pack_desc_for`, `predict`, `predict_pack`, `workspace_size`, `lowering` and
    `pack_layout`, the targets checked with `err_estimate` at the lowering's (N, k, dispatch,
    b·m·n), the rule above applied from first principles. The chosen configuration, its
    status, its composed time (bit for bit), its per-role figures and its err fields match;
    each choice equals the `configs()` lookup; rank's `pack_s` equals the per-role sum; the
    band composes.
  - Scenarios: every reuse kind, the `"phase"` fallback and `assumed_reuse`; a shared role
    divided by the members' summed calls, members at one N above the group floor; N monotone
    until `reset()`, falling with `monotone=False`; DIRECT only when no emulated level meets
    and only if listed, and the monotone top; a share-group member at DIRECT leaves the
    group for the phase (both member forms; the others keep one N; it links no groups; the
    shared reuse and the `configs()` lookup kept); BUDGET (the floor and the least
    total_bytes) and MISSED's precedence, and `BUDGET` never on a missed target; TYPICAL
    p = P / (iterations × calls); RN κ ≥ T levels; argument errors. The brute force applies
    the same DIRECT rule from first principles. **[final review]** Share groups whose roles
    read different packs, or that no level can pack, are refused at construction (in the
    random suite too, from pack keys the test computes itself, after which the members run
    by name only), while one operand read by two contractions shares one PackSpec; equal
    step-1 `min_moduli` results give equal configurations (the "Cost" pattern).
  - `footprints()` equals the maximum of the C queries over the allowed pack modes × vec
    modes × every g, per level, with its totals.
  - **[final review]** The plans a phase names: every `configs()` and `phase()` choice lowers
    (`host.lowering` with its `options()` and `pack_specs`) with no PACK_MISMATCH and its
    `workspace_bytes`, and a share group's member lowers with another member's PackSpec of the
    shared role. `python_device`'s CPU part creates those Plans and Packs through its fake
    library (in-line, packed, DIRECT) and runs a share group's two plans on one
    `PackedBuffer`.
  - `rho_from_output` equals `params.rho_recipe` bit for bit (random and boundary values,
    every α = 2^e class, specials, broadcast and batched forms). On a 2-D C, 1-D bounds at
    m = n and mixed with 2-D bounds equal the explicit (m, 1) × (1, n) result, and a 1-D
    bound of another length raises. On a batched C (3-D and 4-D), a vector shared by every
    batch in the full-ndim spelling (also with the batch dims spelled out, or a scalar on one
    side) equals each batch's 2-D call bit for bit; per-batch (…, m, 1) × (…, 1, n) bounds
    and per-batch scalars (…, 1, 1) on unequal batches (one and two batch dims), and on a
    4-D C bounds shared over one batch axis only, (b0, 1, m, 1) × (1, b1, 1, n), equal each
    batch's own 2-D call, with no certified ρ̂ below row_i·col_j / |C_ij| in exact
    rationals. Every 1-D and 2-D bound on a batched C raises: per-batch scalars without
    keepdims, (b,) at b = m and at b = n and (b, 1) at b = m; the (b, m) / (b, n) norms at
    b = m and b = m = n; a shared (m,), (m, 1), (n,) or (1, n) bound on a 3-D and a 4-D C.
    So do bounds of another ndim and transposed, elementwise and partial-batch ones. A form
    the rule would wrongly accept fails with its count of certified outputs below the true ρ
    (exact rationals): the mutation that drops the ndim rule, and the pre-MF4 code, fail
    first on a (b,) row bound at b = m = n = 3, 9 of 27 certified outputs below ρ. U ≥
    params' exact bound (its upper enclosure) at N = 2–20, k = 1–40000, both roundings and
    dispatches, and with ℓ1 ratios from 1 to √k and 1e9 (k up to 2²⁰); ρ̂ ≥ ρ in exact rationals on emulated outputs (`params.emulate_element`);
    ValueError / TypeError cases. Norm bounds convert to float64 exactly (float16, float32
    and integers ≤ 2⁵³ equal the float64 call) or are refused (2⁵³ + 1 as an array and as a
    list, a uint64 above 2⁵³, Fractions, Decimals, 2⁷⁰, a long double wider than 64 bits,
    complex, bool).
- **[cert] `cert_rho_recipe`** (`params.py --check`), in exact rationals: on emulated elements
  (both roundings, every N = 2–20, random and cancelling rows whose |x| is 0.3–30 × U‖a‖‖b‖,
  α = 2^e, U the tightest double above the exact NORMWISE bound, with ν = √k and with the
  rows' ℓ1 ratios) and on native elements (the §3.7 FMA chain in exact arithmetic, RN at
  each step, k = 1, 2, 7, 40, U the tightest double above γ_k: the DIRECT and k = 1 case),
  certified implies x ≠ 0 and ρ̂ ≥ ρ; at the boundary (|D| from 3 ulps
  below to 6 above RN(U·n), α = 2^e), certified implies |D| > U·n exactly and
  ρ̂ ≥ n / (|D| − U·n), and the evaluation is tight (certified exactly from 2 ulps above);
  the uncertified classes (zero, subnormal, non-finite, C = 2⁻¹⁰²² with e < 0, overflows).
- **No TLA+ change** (the §6f.5 argument): the helper composes pure functions of descriptors
  and plain structs; its only state is the Python object's monotone floor, and it holds no C
  object with a lifetime. The ρ recipe is a [cert], not a protocol property.
- **No ABI change:** host API 13; Python only (NOTES-C1 §4.9).
- **Caller note: packing the invariant operand** [M, model-only, outside hold-out] (NOTES-C1
  §2.4, §5.10 MD2-4). On the PPL view (100 × 9216 × 9216), packing the invariant B (680 MB)
  per phase lowers the schedule's gain to 1.14× (P14/PF14 = 0.876), against 1.33× in-line,
  since each pack costs 1.72–2.41 ms.
  - Packing B is the larger lever: P14/S14 = 0.509 at the same schedule, and
    PF14/F14 = 0.436 at fixed N = 14.
  - DF and 4096³ barely change (P14/PF14 = 0.667, 0.678).
  - NOTES-C1's M-d2 measures P14/PF14, reported and not gated; its figures replace these
    once it runs.
- **Caller guidance: switching N inside an iterative method** (NOTES-C1 §1.3, M-d1; §5.6
  S2-9). CPU evidence on one fixture each, run bit-exact with the oracle's arithmetic: it is
  report-only guidance, not a C1 gate and not a library rule. Source: immerdink-eval
  `findings/c1-evidence.md` ("M-d1"), records `results/c1/m_d1/` (untracked), reproduced bit
  for bit on committed code (jacobi, davidson and cg at 928b49c; diis and ppl at 5925219).
  - *Fixed-point iterations (Jacobi, DIIS) and PPL + ring:* the forcing-rule ramps
    (`bench/accuracy/schedule.py`, `monotone=True`) match fixed N = 14: no more iterations
    to 1e−7 and 1e−10, the final residual within 2× (ratio 0.07–1.00), for η ∈ {0.1, 0.01},
    BOUND and TYPICAL. This is §4.3's success criterion, met on every system.
  - *DIIS:* reset the DIIS history at each precision switch. It costs no iteration and
    lowers the final residual (1.5e−16 against 2.1e−15 without the reset). The lower
    residual comes from restarting a DIIS that stalls: plain DIIS stalls near 1–2e−15 with
    or without precision switches (the ramps without the reset end at 1.18e−15 to 2.09e−15;
    fixed N = 14, with no switch, and the exact-arithmetic reference also end at 2.09e−15), so
    the stall is not caused by the switch.
  - *Subspace methods (block Davidson):* at each N switch, collapse the subspace to the Ritz
    block and recompute its σ = HV at the new N; otherwise the low-N products stay in the
    subspace and the ramp stalls (4.1e−8 attainable). Even with the recompute the ramp needs
    16 iterations to 1e−10 against 11 at fixed N = 14 (+5; 11 against 7 to 1e−7) on an easy,
    diagonally dominant case, so a ramp is not recommended for Davidson unless its
    per-iteration saving outweighs that.
  - *Krylov methods (CG): start tight and relax* (`monotone=False`). Rules that start low
    and tighten stall CG at N = 5–6 (cond 1e6). A relaxed matvec tolerance
    min(1e−2, tol/‖r_k‖) converges (1e−7 in 274 iterations against 251 at fixed N = 14, with
    N falling from 12 to 6) but leaves a residual gap (7.1e−8 attainable, against 3.3e−10
    at fixed N = 14 and 2.0e−10 with the FP64 matvec). Scale the relaxation by the expected
    iteration count, e.g. tol/(m_max·‖r_k‖), as inexact-Krylov theory does; M-d1's rule had
    no such factor and that scaled rule is not measured here.
  - *A lagged ρ for ELEMENTWISE targets* (`rho_from_output` of the previous iteration's
    output, §6g.11, choosing the next call's N): it missed the next call's target in 1 of
    228 MET calls (PPL at 1e−6, realized 1.013e−6, after an output passed near zero; the
    next call went to N = 16 and met it). The lagged ρ̂ predicts the next call's ρ and does
    not bound it; ρ̂ certifies only the output it is computed from. These records
    (`md1_rho.jsonl`, the IR7 re-run with the shipped helper) name d89988a with a dirty
    tree and carry the sha256 of every study file, each equal to the committed file at
    f97696a (`precision.py` also at 51dba1e; NOTES-C1 §5.6 S2-18): tied to committed code
    by content, not re-run on it.

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
| Accurate mode ([SC25]'s partner-aware σ/τ shifts) — **[v1.19]** re-examined for wide dynamic range (NOTES-C1 §2.5, §2.8, M-a) | Still rejected. It needs one more operand pass (KF/KL lose their group-1 fusion), and σ depends on the partner, which breaks operand packing (D1). It also converts headroom poorly: by an estimate from the §3.8 bound, confirmed by NOTES-C1 §2.5–§2.6's data (not a theorem), partner-aware shifts realize about G/7.8 moduli of a G-bit headroom where K-weights realize about G/3.9. Measured [C]: ≤ 0.3 moduli on dense data, ≤ 0.34 for the faithful 6-bit form even on decay_opp, about 3 with exact or envelope bounds on separable anti-correlated data. M-a at the ≥ 2048³ class (immerdink-eval `findings/c1-evidence.md`): the never-worse envelope rule (NOTES-C1 Appendix B, recorded as the proven partner-aware option) saves 1.5–3.5 moduli at equal elementwise error on decay_opp and opp_nonsep (K-equilibration 4–7), −0.5 to +1 on random-structure data, and none on the real psi4 operands. Its revival trigger (≥ 2 moduli beyond both K-equilibration and K-segmentation on a non-separable family of that class) is not met: 0 cases. |
| K-equilibration in the library: power-of-two K-weights with `imd_kweights` and caller pre-scaling, then library-internal weighted packs (NOTES-C1 stages 4, 6) | Closed by its pre-registered gate (NOTES-C1 §4.5). No real-operand contraction of the ≥ 2048³ class (psi4 DF and MP2 operands of benzene and water on the consumer's contraction list) passes the L5c guard: their K profiles are aligned (G ≤ 0), and with the caller's costs K-equilibration reaches 2–47 % lower TFLOPS-eq than fast mode there at the normwise targets (+2 % to +89 % time; N = 9–12) [M, model-only, outside hold-out] and 1–36 % lower at the ew8 targets (+1 % to +57 % time; N = 13–16) [M, model-only, outside hold-out at N = 13 and 15]. It saves about 6 moduli only on synthetic anti-correlated profiles. The exact rule stays as `oracle.kweights`, and `oracle.kweights` with M-a is C2's deliverable. **Stage 4** (caller-side, `imd_kweights`) **closed**, 2026-10-07; **revival trigger:** a real-operand family meeting gate (i) (guard = 1 on a real contraction of the ≥ 2048³ class, and a net TFLOPS-eq gain ≥ 5 % at equal target with every caller cost counted, or an elementwise target ≤ 1e−8 that fast mode cannot reach at N ≤ 20 becoming reachable), judged with the re-registered (ii) of NOTES-C1 §4.5, after the d*-deciding part of Tier S is re-run on committed code (NOTES-C1 §5.6 S2-21). **Stage 6** (library-internal weights and weighted packs) **deferred**, since its gate needs a shipped stage 4; **trigger:** a revived stage 4 that ships and passes M-e on its gate case, plus a real fixture on which library-internal weights remove ≥ 5 % of pre-scaling or scaled-copy cost on a recurring contraction (NOTES-C1 §4.7). |
| K-segmentation as a library mode (a SEGMENTS flag or a segment loop; NOTES-C1 Q15) | A documented caller recipe instead (§3.8 "Segmented accumulation", PROOFS L7): it works today with β = 1 and moved base pointers. M-a: on every guard-accepted Tier-L case it needs 1–4 more moduli than K-equilibration; on the real operands it saves at most one modulus (normwise 1e−15) at +4 % to +62 % of fast mode's time (R2b: 2 % faster, inside the model's error) [M, model-only, outside hold-out at N = 13], and it gives up the single rounding. An API only if a real workload shows value beyond the recipe. |
| More bits per modulus via larger moduli (e.g. Karatsuba-split residues) | More INT8 GEMMs per bit on sm_89 [I]: a residue of a modulus above 256 needs two int8 parts and three GEMMs per modulus, against one per 8-bit modulus. The FP8 analogue, Ozaki-II with Karatsuba-split FP8 residues (arXiv:2603.10634), needs 3 FP8 GEMMs per modulus and about 2.5× the multiplications of the INT8 scheme for 53 bits. |
| N-range plan: one plan serving every N′ ∈ [N_lo, N_hi] (NOTES-C1 §3.2) | Parked. A call at N′ is launch-identical to the single-N′ plan, so it gains nothing per call; precision phases use a plan per N (§6g). It would save only plan count, host bytes and creation time (0.011–0.15 ms per N, once). **Trigger:** a consumer that cannot re-plan per phase, or a measurement showing that plan count, plan creation or RESIDENT memory matters. |
| Multi-N packs; round-to-odd reference-integer packs; lower-N planes from a higher-N pack (NOTES-C1 §3.1 A3, A4, A6) | A multi-N pack has the same bytes as separate packs and saves one FP64 read per pack creation, for N-set key semantics. A round-to-odd pack takes 5 / 8 / 9 / 11 B per element at N_max = 8 / 14 / 16 / 20 (≥ FP64 from N_max = 14), each call still reads ≥ 8 B per element, and it saves only the S pass that KL already fuses. Lower-N planes cannot come from a higher-N pack: σ depends on N, rounding is not modular, and the CRT over a prefix gives z mod P(N′) (PackedOperand `neg_n`). |
| Choosing N on the device within a call | N fixes the launch sequence, the K5 instantiation, W, the workspace and σ itself, and kernel scope forbids the sync. N is chosen on the host per phase (§6g.10). |
| Ozaki-I slice mode: prefix slices, one pack for every precision (C1(c), NOTES-C1 §2.9, §3.3) | Parked. Faster than Ozaki-II only on benign data below about 1e−8, break-even near 1e−10, 1.25–1.75× slower at FP64 level [E], no win beyond S = 2 slices on heavy-tailed rows; and no operand is needed at two N at once. **Triggers:** a consumer needs one operand at two or more precisions at once under memory pressure, or M-d measures phase-switch re-packs at ≥ 5 % of a phase. A revival must beat the prefix-plane option and the best configuration without slices (a shared pack, separate packs or in-line). On the DF-CCSD contraction set a pack serving every N could save nothing in the 12 forcing-rule cases with real operands (the best of those configurations is already at or below the ideal: each member at its own floor reading one pack) and up to 9.1 % per iteration with one member's floor a level apart (16.0 % at 8 against 14) in synthetic floor patterns [M, model-only, outside hold-out; the slice pack priced as an ordinary pack at the group's top floor, no slice per-call penalty; B roles of one-factor contractions in-line, bare-factor norms; NOTES-C1 §3.3, M-b]. The share-group raise cost (shared against separate packs) is not that bar. A revival builds its TLA+ then. |
| **[v1.19]** KL B, the narrow operand's pack on the multi-pass launch (rows of three or more chunks, a grid of fewer CTAs than SMs): its time model (NOTES-C1 §5.9 MC-3a, §5.11; §6c.9) | **Open model defect** ([GitHub issue #7](https://github.com/Yurgenschmidt/immerdink/issues/7) (label backlog)). Shipped model: ×1.50–1.81 its REPEAT and ×1.31–1.65 its COLD prediction on Q24's three narrow-KL shapes, ×1.65–2.02 on M-c's Jacobi view; after the MC-3a candidate refit (Q24 in the fit) −18 % median signed on the Jacobi view's KL B marks (12 % median \|error\| on the allowed narrow-KL records), partly hidden at the call total by a KL A / K4 over-prediction of +9 %. The compute chain `nc · values · g` fits Q24 at 57 ns per step but not the calibration's wide launches (6–13 ns), and an underfilled-only feature breaks monotonicity (§6c.9). **Trigger:** a consumer whose planner decision depends on a narrow-n KL call below about 150 µs (the shipped model is 25–45 % low on its KL B phase and 26–29 % low on its total), or a calibration sweep that adds narrow multi-pass KL packs (which would identify the feature), or an explicit decision to adopt the candidate calibration (the independent review of 2026-10-09, ruling A, declined: MC-3a did not pass). |
| K5 statistics and a device a-posteriori certificate (lemma L6; NOTES-C1 stage 5) | Deferred: no consumer issues elementwise targets, and the host-side `rho_from_output` (§6g.11) gives a rigorous ρ now. **Triggers** (NOTES-C1 §4.6): T1, a consumer issues ELEMENTWISE requests per phase and the recipe's measured cost is ≥ 3 % of its run; T2, a consumer needs a rigorous per-call elementwise certificate on every call; T3, Eric asks for it. On a trigger it is built as NOTES-C1 §4.6 designs it, with its cost bar restated without `ncu` (compiler resource usage, the occupancy query and A/B timing; IR26). C1's definition of done does not need it (IR26, IR29). |
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
  - Every other sizing value (workspace, pack and plan layout, plan bytes apart from the
    add-mod shape keys) is unchanged.
  - `imd_perf_predict` / `imd_perf_rank` values change for every multi-chunk call
    (k̂ > 131008), not only in the windows: the chunk sizes and per-chunk round counts
    change; the format-2 slow-chunk set changes (the `c·K_c' mod 32` term is now always 0);
    and under format 1 the full chunks (131008 ≡ 0 mod 64) are no longer slow. Measured on
    θ fitted with 131056, the model error on 2048² × 196608 / 196640 goes from −38 / −32 %
    to −20 % (NOTES-K4ALIGN §6). θ is to be refit (with stage 2).
  - The "+1 chunk in the windows" statement holds while the windows `(j·131008, j·131056]`
    are disjoint, i.e. j·48 < 131008, j ≤ 2729 (k̂ ≲ 3.58·10⁸); beyond that the chunk counts
    can differ by more than one.
  - A pre-check before the change ran 11 long-k cases with `imd_test_set_limits(k_chunk =
    131008)` on the old code. All were bit-exact against the oracle at chunk 131008 and
    identical to the 131056 run.
- **Stage 0 (refactor, no behaviour change).** `k_hat_of(k, A)` (`src/host/k_hat.hpp`) is the
  one rounding of k to k̂ in libimmerdink_host, with A = 16 for every plan. It is mirrored
  by `params.k_hat_of`.
- **§6c.** A correction to the v1.10 per-chunk text: the criterion is the chunk end, not
  the start, and the 262144 effect was the residue-first phase, not a DRAM-channel effect.
  The format-2 criterion is now exactly "k̂ mod 32 ≠ 0". Review ruling: no calibration format
  bump; θ is to be refit (open item, done with stage 2).
- **Spec.** `spec/apalache/Epilogue.tla`: `Kp = 131008`, with the tight max 2146435327
  (new counterexample value). The comments in Pipeline, MC_Pipeline, OzakiCRT and
  Int32Chunk change; no TLC model changes. `params.cert_row_phase` (new) covers the boundary
  k list and the 131008 / 131056 windows. Lean `Chunk.lean` (≤ 131056) covers 131008
  a fortiori and was not re-instantiated.

## v1.13 stage 2 (implemented 2026-10-06: per-plan K alignment; host API 9, pack format 2; no semantic change)

- **§4b (final design, replacing the draft).** `k̂ = ⌈k⌉_A` is the row stride and the GEMM
  K extent of every residue row, with `A = 64` iff the logical k > `K_PAD64_MIN = 96`
  (frozen from the 2026-10-05 threshold sweep: mean call-time loss 0.30 % against the
  per-k best, +2.8 to +5.6 % per call against never padding), else 16. A pure function of
  k: no option, no calibration, no device. Every K4 load of an A = 64 plan is 64-aligned
  (with K_c' = 131008 and 256-aligned bases); launches, MMA work and results are unchanged;
  per residue row `δ = ⌈k⌉₆₄ − ⌈k⌉₁₆` more bytes (≤ 1/3 relative, at k = 129…144; ≤ 4.6 %
  for k > 1024). All review rulings of 2026-10-05 are folded in (§4b.3–§4b.10).
- **§4a, §6b.4.** The layout formulas with `k̂ = ⌈k⌉_A`; byte-identical for k ≤ 96. Format 2
  = the format-1 layout at k̂ = ⌈k⌉₆₄.
- **§6a / §6b.5 (host API 9, additive).** `imd_lowering_t` v6: `k_align`, `reserved6`
  (192 bytes); `k_hat` means ⌈k⌉_{k_align}. `IMD_PACK_FORMAT_A16 = 1`,
  `IMD_PACK_FORMAT_A64 = 2`, `IMD_PACK_FORMAT` = the newest format (2); a pack's format is
  AUTO(k); the key's word 0 is the format (the plan's for a slot key). No `imd_options_t`,
  `imd_pack_desc_t` or `imd_pack_layout_t` layout change. The test-only hook
  `imd_test_set_k_align` (libimmerdink_host) forces A. The integration contract: sizing only
  through libimmerdink_host; a caller persisting pack bytes persists `format` and
  `pack_bytes` with them and re-packs on a difference.
- **§6c.** No calibration format bump; the format-2 rule stays exact; θ to be refit and
  `k4_select` re-checked after stage 2.
- **Spec.** New module `K4Align` (CUTLASS's residue-first address sequence over absolute
  addresses; 8 positive and 8 negative models). Pipeline: `EveryRowByteWritten` and
  `RowPhaseAligned(T)`, four stage-2 producer bugs, 9 positive and 8 negative A = 64 models.
  PackedOperand: the format carries A (AUTO or hook), `AutoFormatsAgree`,
  `FormatMismatchRejected`, the reason order (`ReasonIffMismatch`,
  `FormatMeansSameTensor`), 12 positive and 4 negative models. `params.py`: `K_PAD64_MIN`,
  `k_align_of`, sizing by A, `cert_k_align` (new), `cert_workspace` / `cert_pack_layout`
  over both A with hand-computed A = 64 examples. `oracle.py`: `pack(fmt=)`, the format in
  `pack_lower` / `pack_layout` / `pack_key` / `lower_packed`; `test_oracle.test_k_align`.
  The implementation commit flips both to `K_ALIGN_POLICY = "AUTO"`.
- **Adversarial review of the stage-2 design (2026-10-05), must-fixes applied.**
  (1) "Never misread" was false: the format lives only in the handle key, pack bytes have
  no header; the hazard is caller-persisted v8 bytes in a v9 handle (silent wrong stride);
  contract: persist `format` and `pack_bytes`, re-pack on a difference (§4b.5, §6a, §6b.5).
  (2) A is resolved once per API call (`analyze`, `pack_lower`) and passed explicitly:
  `WorkspaceShape` / `pack_layout` take A, `ExecPlan.k_align`, libimmerdink never calls
  the resolver. (3) The hook state is one out-of-line atomic in libimmerdink_host behind
  an `IMD_HOST_INTERNAL` accessor; the §6a "pure" exception is documented. (4) Reason
  order: early FORMAT only for an unknown word, 1-vs-2 last against the plan's own format
  (k = 96 pack, k = 97 plan: EXTENTS); DESIGN §6b.5, the oracle and (stage 2) C++ agree;
  PackedOperand `Reason`, `ReasonIffMismatch`, `FormatMeansSameTensor`, 3 positive and 1
  negative models. (5) The policy switch is one switch: `k_hat_of` defaults to the policy,
  `PACK_FORMAT` is derived; the test call sites relying on a default are listed. (6)
  initcheck runs on unpoisoned fresh allocations; the padding check covers KL and K2/K3.
  (7) Stage-1 window tests pin A = 16; 196608 and 262017 added. (8) A = 64 can merge the
  last add shape key (k = 393008) and cross the 256 MiB `k4_tune` cap (`cert_k_align`,
  `params.k4_add_chunk_keys`, `k4_tune_bytes`). (9) `host.test_set_k_align` clears the
  Python plan cache. Later items folded in: `AutoFormatsAgree` is a sanity check;
  `pl.k_hat == e.k_hat` and the `call_of` ≡ 0 mod A debug check; `tile_macs` from
  Σ⌈kc⌉₆₄; A = 16-forced calibration cases and `--k-align`; the v5-size lowering test;
  `k_align = 16` for DIRECT / K-empty; `k_align` in `ExecPlan`, not `K4Plan`.
- **Implementation (2026-10-06, branch `k4-align-s2`).** As designed (§4b.3–§4b.8); the
  kernels needed no logic change (they index by the plan's k̂ and write the zeros [k, k̂)).
  Where the design left the code a choice, or the oracle and the model differed:
  - *Two packed operands: one reason rule.* The design and PackedOperand `Reason` compare
    the 1-vs-2 FORMAT after the other key words of **both** packs; the design-branch oracle
    compared each pack's full key in turn. `oracle.lower_packed` now uses the model's rule,
    and `analysis.cpp analyze_packed` the same: pack A differing only in its format and
    pack B in a stride is STRIDES (`test_oracle.test_k_align`, `tests/test_k_align.cpp`).
    No TLA+ change.
  - *The slot key of a DIRECT plan.* `imd_lowering_t.k_align` is 16 for a lowering without
    residue rows, but the slot key's format is that of the resolved A of the plan's k (as
    in the oracle), so a DIRECT plan accepts a matching pack of any k (§6b.2: the mode is
    not in the key); the `pl.k_hat` drift check compares with ⌈k⌉ of that resolved A.
  - *Where A lives.* `Analysis::k_align` (the one resolution in `analyze`),
    `WorkspaceShape::k_align` and `pack_layout(…, k_align)` (no default: a missing A is a
    layout failure), `ExecPlan::k_align` (in the padding after `trunc`: `sizeof(ExecPlan)`
    and the plan host bytes unchanged), `PackLowered::k_align / format` (the one resolution
    in `pack_lower`). A host query with pack descriptors resolves once in `analyze` and
    once per descriptor's `pack_lower` (each a pack of its own).
  - *The unknown-format branch* stays as a defensive check. It is tested only in the
    internal unit test `tests/test_k_align.cpp`, on a copy of the pack's lowering with key
    word 0 = 3 or 0 (FORMAT before MODULI), like the k̂ drift check.
  - *The `call_of` check* (`gemm_k4.cu`): every K4 operand pointer ≡ 0 mod A, for plans whose
    K-chunk is a multiple of A (the production K_c'; `imd_test_set_limits` may choose
    another). On by default in debug builds (a violation fails the call), and through the
    test hook `imd_test_k4_align_check` (libimmerdink) in any build (a violation is
    counted). 0 violations in `contract_k_align` and `packed_k_align`, each also asserting
    that the check ran (the hook's checked-launch count, mode 2, > 0) and that it fires: a
    test mutation (mode 3: the check sees the A pointer + 8 bytes, the launch unchanged)
    counts every checked launch as a violation (fix review 2026-10-06 must-fix 3).
  - *`k4_select`* keeps its estimate over the plan's k̂ (unchanged for every A = 16 plan);
    only the performance model's K4 MMA term and its ops count use Σ⌈kc⌉₆₄ (§4b.6). An
    A = 16-forced plan of k > 96 may therefore pick another K4 configuration than the
    A = 64 plan of the same k (results are identical). Re-checked with the refit: the rule
    stays (NOTES-B2 §8).
  - *Tests (§4b.8):* `host_k_align` (new, CPU); `host_c_api` (the hand-computed A = 64
    cases, the 180² case recomputed, a v5-size caller); `host_python` (`suite_k_align`: the
    boundary k list, every query at the hook's 16 and 64 against params / oracle at the
    same policy, the add-key merge at 393008 and 392961, the tune cap, a k = 96 pack against
    a k = 97 plan, accuracy and κ across A); `workspace_vs_params` (the boundary k);
    `contract_k_align` and `packed_k_align` (new, GPU: every producer path at A = 16 and 64
    on a 0xA5-poisoned workspace and pack buffer with the padding check, identical C
    across A, PACK_MISMATCH when the hook changes between pack and plan, both directions,
    and in `imd_contract_packed`). The stage-1 window tests pin A = 16. initcheck and
    memcheck run the two `k_align` suites with `IMD_TEST_POISON=0` (fresh, never-written
    allocations) and `IMD_TEST_QUICK=1`. The two-chunk cases run on skinny shapes only
    (the pure-Python oracle cannot do 512² × 131072 within the test's size guard).
  - *Bench and calibration:* `imd_bench --k-align auto|16|64` (or `IMD_BENCH_K_ALIGN`);
    every record carries `k_align_hook`, imd / duel records also `k_align` and `k_hat`;
    `tools/calibrate.sh --k-align`; `bench/perf_fit.py` models each record at its
    alignment (records without the field ran at A = 16: they predate stage 2).
  - *Refit (2026-10-06, branch `k4-align-s2-refit`; NOTES-B2 §8).* The shipped calibration
    was refit with no format change: 14 new calibration cases (A = 16 forced at k > 96 on
    every configuration, AUTO k ≤ 96) and 16 new hold-out cases across the AUTO range. The
    hold-out meets every §6c.5 target. `tools/calibrate.sh --sustained / --cache` reuse
    earlier sustained / cache records when those sweeps' plans are unchanged.
    `bench/perf_fit.py` now fits the K4 MAC costs per round, c_q = q·mac_q, as cumulative
    non-negative increments (c_1 ≤ c_2 ≤ c_3), so a round of q concurrent CTAs never costs
    less than a round of q − 1. This keeps the predictions monotone in the shape (host_perf).
    The `k4_select` re-check keeps the rule and its constants.

## v1.13 follow-up: hygiene (host API 10; no semantic, layout, launch, pack-format, calibration-format or spec change)

- **Release note (host API 9, restated): `IMD_PACK_FORMAT` changed meaning, 1 → 2.** Up to
  host API 8 it was *the* pack format (1). From host API 9 it is **the newest format the
  library writes** (2 = A64); a pack's actual format is a function of its k (1 for k ≤ 96,
  2 for k > 96) and is reported by `imd_pack_layout_t.format`. Code comparing
  `imd_pack_layout_t.format == IMD_PACK_FORMAT` to decide whether saved bytes are reusable
  is wrong for k ≤ 96 packs (format 1, still valid); compare with the `format` saved next
  to the bytes instead (§4b.5 contract, §6a integration contract, §6b.5). Also in
  NOTES-K4ALIGN §12.
- **§6c.3: explicit calibration parse errors (host API 10, additive).**
  `imd_perf_calib_error_t` (version 1, 272 bytes), `IMD_PERF_CALIB_ERR_*`,
  `imd_perf_calib_parse_ex`, `_load_ex`, `_default_ex`, `imd_perf_calib_error_name`. The
  existing functions are wrappers with unchanged results. The shipped calibration keeps its
  parse error; CMake rejects an embedded file whose string values do not fit at configure
  time; `bench/perf_fit.py` checks the metadata it writes and re-loads the file
  (`--check FILE...` for existing files); Python raises `host.CalibrationError` with the
  line, reason, key and message. **Python behaviour change:** with a broken shipped
  file (unreachable in a CMake build, §6c.3), the `calib=None` paths (`host.predict`,
  `host.predict_pack`, `host.rank`, `planner.Query` construction) raise `host.CalibrationError`
  (a `ValueError`) instead of `host.ImmerdinkError` (`INVALID_VALUE`, a `RuntimeError`);
  a caller catching `ImmerdinkError` there must also catch `CalibrationError`.
  `bench/perf_fit.py`: `--notes ""` / `--version ""` mean the generated value (as before
  host API 10); the generated version is cut to fit like `clocks` and `notes`. Cause: a
  calibration whose `notes` line exceeded the
  127-byte field failed to load, and as the shipped file it silently turned every NULL-
  calibration perf query into INVALID_VALUE.
- **§4b.5: the hook is read once per API call** (stage-2 code review): one `KAlignSel`
  snapshot at the entry of each API call, passed to `analyze`, `pack_lower`,
  `analyze_with`, the perf model's `predict_contract` / `predict_pack` and every
  candidate of `imd_perf_rank`; a host query with pack descriptors runs `analyze` once
  (`analyze` + `match_packs`, was `analyze` + `analyze_packed`).
- **§6b.4 / §6b.5: the slot key of a DIRECT plan** keeps the resolved A of the plan's k
  (not `imd_lowering_t.k_align` = 16); the reason is now stated in §6b.4 word 1.
- **§4b.5 device checks.** KF's `2^lg_tk ≥ k̂` moved to plan time (`plan_data`). The
  `call_of` ≡ 0 mod A check is a host pre-pass over every K4 launch before K0, so a
  debug-build violation fails the call with nothing queued (was: mid-call, after K0–K3
  and earlier groups' K4 launches had been queued).
- **Tests.** C and Python tests of every calibration error reason (an over-long `notes`
  line among them) and of the `perf_fit.py` generation check; the hook read once per call
  (`host_k_align`: `analyze`, `pack_lower`, `analyze_packed` and `analyze_with` follow the
  snapshot they are given, not the current hook, and `analyze_with` keeps analyze's,
  the descriptor's and `match_packs`' status order); the FORMAT reason of a handle-based
  mismatch asserted (`imd_test_pack_mismatch`, libimmerdink test export: plan creation's
  analysis of handles, returning the reason); `imd_lowering_packed` with descriptors under
  one hook value never reports FORMAT (one snapshot); `host.test_set_k_align` clears
  `device.default_plan_cache`; `planner.Memo` is caller-owned and **not** cleared by the
  hook (a hit after a hook change returns the result computed under the old A; tests use
  a fresh Memo or `clear()`), documented in `planner.py` and §4b.5's Python bullet; the
  **initcheck mutation test** for one-sided padding: the test export
  `imd_test_set_pad_mutation(x)` makes the K2/K3 pack of post-swap operand x store only
  up to ⌈k⌉₁₆ (Pipeline `neg_a64_pad_old_k3`'s bug), the call stays bit-exact (the other
  operand's zeros cancel it), and `compute-sanitizer --tool initcheck` on unpoisoned
  allocations must report it (`contract_initcheck_pad`, registered when
  `IMMERDINK_COMPUTE_SANITIZER` names the tool or it is found next to nvcc), while
  `contract_pad_mutation` (poisoned workspace) shows the mutation takes effect (the
  padding stays 0xA5 on that side only) and the call stays bit-exact. CMake's string check
  itself: `calib_strings_shipped` passes, `calib_strings_too_long` (the committed vector
  `tests/data/perf_calib_notes_too_long.calib`, notes 128 bytes) fails with the line.
- **Code hygiene.** `perf.cpp`'s header comment names the test-hook exception to "pure";
  `bench/summarize.py` sizes the phase bounds with a record's own `k_align` / `k_hat`
  (forced alignment) when present.
- **[hygiene-b] Review of the hygiene follow-up.** `tools/test.sh`: a caller's ctest
  selection option (`-R`, `-E`, `-L`, `-LE`, `-I`, `-U`, `--tests-from-file`,
  `--rerun-failed`, …, attached or `=` spellings too) replaced or widened the CPU-only
  pass's own `-R`, so GPU tests ran **without the GPU lock**; with any such option the
  whole selection now runs in one pass under the lock (the two-pass scheme is kept for a
  plain run). `tools/test.sh --self-test` checks the detection (ctest
  `tools_test_sh_selftest`, CPU-only). `tools/test.sh` warns at the start and the end of a
  run when `contract_initcheck_pad` is not registered and prints how to enable it. The
  pad-mutation hook state is a `std::atomic<int>` (relaxed). `imd_lowering_t.k_align`'s
  header comment states that it is not a DIRECT plan's pack key. Doc notes: the Python
  `calib=None` exception change and the broken-shipped-calibration path being unreachable
  in a CMake build (§6c.3, changelog above); `perf_fit.py` `--notes ""` / `--version ""`.

## v1.14 (branch-free ε in the pack emitter; no semantic, layout, launch-shape, pack-format, API, calibration-format or TLA+ change)

- **§3.4 / §3.4.1.** The pack emitter (K2/K3, KF, KL, `imd_pack`) holds |a'| = μ·2^ε as
  W(N) u32 words (W = 2 for N ≤ 16, 3 for N = 17–20) and reduces it with byte weights
  2^(8i) mod p (bytes 0..9) by W `dp4a`, one `umulhi` and one multiply-subtract; the
  `pow2_mod` multiply for ε > 0, its per-thread branch and its lane-divergent table load are
  gone. Same int8 bytes (same congruence class).
- **Kernels.** `PackEmitterT<E, W>` moved to src/kernels/pack_emitter.cuh; `k23_pack`,
  `kf_scale_pack` and `kl_scale_pack` take W as a template parameter, chosen per plan from
  N (`pack_words`). `PackWeights` / `c_pack_weights` (derived from `pow2_mod`) and
  `mod_p_words32<W>` (device/modp.hpp) are new; `ByteWeights` / `mod_p_bytes32` stay for the
  exhaustive test of A5's reduction.
- **Certificates.** New `cert_pack_words`; generated constants `kPackW2MaxN`, `kPackXBits2`,
  `kPackXBits3`, `kPackWeightBytes`.
- **Tests.** `mod_exhaustive_cpu` widened to [0, 12·255²] and `mod_p_words32`; new GPU unit
  test `pack_emitter`; new contract suite `eps` and packed cases with ε-heavy data.
- **Performance model.** Unchanged in v1.14; its eps data dependence is retired in v1.15
  (§6c.1 "The eps path — retired").
- **Measured** (RTX 4090; eval workspace `results/followups2/bf-eps/`, bench/sweep_bf_eps.txt,
  two interleaved before/after rounds, C bit-identical by hash in every case):
  - Registers (ptxas): `k23_pack` 114 → 84 (W = 2) / 117 (W = 3); KF and KL stay at their
    launch-bound caps (128 / 64) and their spills drop (KF E = 16: 8 B → 0; KL E = 8:
    28 / 52 B stores / loads → 4–8 / 4–12 B).
  - Pack phases at N ≥ 15: 1024³ −38 to −41 % (call TFLOPS-eq 13.65 → 15.65 at N = 15,
    9.33 → 11.19 at N = 20); 1024² × 16384 −21 to −32 % at N = 16–20 (call 15.81 → 18.07
    at N = 16); `imd_pack` 2048² −18 to −25 % at N = 16–20; DF-like acQ,bdQ −4 to −17 %;
    4096³ −5 to −15 % at N = 16–20 (call +1–5 %); r_mm-like 16384×256×4096 −5 to −7 % (call +3–5 %).
    Dominant-entry data now costs what typical data costs (it was the slow case at N = 14
    for k = 1024: pack −19 to −22 %).
  - N ≤ 13: no regression beyond 1 % (large and medium shapes 0–6 % faster; tiny calls
    of 12–16 µs, three rounds of 3000 calls: median ratio 1.000–1.008).

## v1.15 (§6c.7 long hot runs and clock-scaled byte energy; host API 11, calibration format 2 with optional keys, the model's eps path retired; no semantic, layout, launch, pack-format or spec change)

- **§6c.7 (new, draft).** Long sustained windows (2–4 min, NVML traces, energy counter never
  read) on 16384³, 8192³ and a DF 96⁴ × 3000 at N = 14 settle at 74–75 °C and 2240–2310
  MHz. The settled time is 0.6–3.9 % above the shipped SUSTAINED prediction, not ~20 %; the
  thermal part is ≈ 1–1.8 % (half of the gap at 16384³, most of it at 8192³ and DF); the
  rest is a fit bias of the unscaled form (e_tc 0.650 vs 0.691 pJ/op: solved clock too high,
  optimistic). Proposed: a clock-scaled byte energy (`power.byte_f_pj`, SM/L2-side energy per
  modelled DRAM byte; host API 11, optional key, folded v9/v10 view, fit with byte_pj fixed;
  settled hold-out median 1.58 → 0.72 %), no THERMAL regime (recipe: `clock_mhz`, +1–2 %
  documented drift; calibration pre-heat 60 s). Q1–Q3 decided by the independent reviewer;
  the refit needs new sustained windows (GPU).
- **§6c.1** The "~1950 MHz at 16384³" statement and the "~20 % optimistic for long hot runs"
  bias are corrected to the measured values (most likely a single post-loop NVML sample).
- Bench only: `imd_bench --sustain-trace-s T` (time series in method sustained's record)
  and `temp_c_start` / `temp_c_end` in every sustained record.
- **[implemented 2026-10-06, branch `hotclock`] host API 11.** `imd_perf_calib_t.energy_byte_f_pj`
  (appended; 3672 → 3680 bytes); format-2 key `power.byte_f_pj`, optional (absent = 0, ≥ 0,
  rejected in format 1, written by `imd_perf_calib_format` only when non-zero);
  `call_power` adds `e_byte^f·D·(f/f_ref)^γ` (+0 at e_byte^f = 0: every existing file
  predicts bit-identically, tested over shapes × regimes × cache states); a v7–v10-sized
  struct gets the folded view `energy_byte_pj = byte + byte_f` from default / parse / load and
  is read with byte_f = 0 (§6c.2, §6c.3). `bench/perf_fit.py --power-form scaled` (default;
  `--byte-pj` fixed term, default 0; γ = 1, `--power-gammas`) / `unscaled` (v1.10).
  `bench/sweep_sustained.txt`: 60 s pre-heat and a 2 s trace; `tools/calibrate.sh
  --sustained-only`. Python `host.API_VERSION` 11, `Calibration.power["power.byte_f_pj"]`.
  Tests: §6c.4 [v1.15]. New 60 s pre-heat windows and the redone form comparison are in
  §6c.7 "Implementation". **Shipped calibration:** the combined refit (below, NOTES-B2 §10)
  writes the key.
- **Combined refit** (`followups-combined`, NOTES-B2 §10): `src/host/calib/rtx4090_sm89.calib`
  refit on the v1.14 kernels with the eps path retired (no `op_eps` keys) and the clock-scaled
  byte energy (`power.byte_f_pj` 301 pJ/B, e_tc 0.708 pJ/op, γ = 1, f_ref 2480 MHz); hold-out
  targets met in every size class (as shipped before: < 1 ms classes missed on the v1.14
  kernels, N = 15–17 medians 25–29 %). [review 2026-10-06] That gain is the eps retirement,
  not the refit: the old θ with eps ignored already met every target, and against it the
  refit was worse on < 100 µs calls (p90 12.5 → 14.5 %, max 22.9 → 28.8 %), neutral on ≥ 1 ms
  and better at 100 µs – 1 ms (NOTES-B2 §10).
- **§6c.1 / §6c.2 / §6c.3 / §6c.4: the eps path retired** (combined branch
  `followups-combined`, part of host API 11's change set; no struct layout change). The
  v1.14 pack emitter has no eps path, so: `op_eps` features 0 and the chains without the
  (1 + f) factor; request `eps_fraction` validated as before, then ignored; `eps_s` /
  `eps_fraction` out always 0 (predict, predict_pack, rank candidates); the `op_eps` keys
  optional (absent = 0, formats 1 and 2), written only when non-zero; calibration format 2
  kept (the other parameters keep their meaning; a calibration is refit with the kernels, as
  after v1.13 stage 2). `bench/perf_fit.py` drops `record_eps`; `bench/sweep_calib.txt` drops
  its six `--data dominant` cases; the hold-out keeps its three as a check. Python
  docstrings and `host.Prediction` / `host.Candidate` comments updated. Tests: §6c.4 [v1.15,
  eps retired].
- **Review fixes (2026-10-06, `followups-final`; docs and tests only).** §6c.2: the folded
  v7–v10 view is a degraded, unvalidated view (never fitted; not an API-10 prediction; pass
  NULL), and only the calibration functions fold (the predictors read byte_f = 0); §6c.7
  item 1 corrected accordingly (it said the predictors fold). §6c.2: format 2 is **not
  backward-readable**: API-11 files omit zero `op_eps` keys and may carry `power.byte_f_pj`,
  so they do not load in API ≤ 10 (pinned by a `host_perf` check on the shipped file).
  NOTES-B2 §10: the hold-out gain is attributed to the eps retirement (the refit vs old θ
  with eps ignored: worse on < 100 µs, neutral ≥ 1 ms, better at 100 µs – 1 ms; KF at N = 8
  and K3 worse), the sustained figure is stated with u_sq2048 −1.4 %, the f_ref selection on
  the same leave-one-out metric and the 2–4 % high capped clock, and all 14 hotclock windows
  are to be recollected. `test_contract` suite eps: the eps-reach check counts only
  emulated (k ≥ 2) cases.

## v1.16 (pack emitter: defined shift, exact s maxima, W-aware occupancy, W = 1 evaluated; no semantic, layout, launch-shape, pack-format, API, calibration-format or TLA+ change)

- **§3.4.1 defined shift.** The W = 2 shift `μ << ε` is PTX `shl.b64` (count clamped at 64:
  defined for any ε, identity in the contract); W = 3 was already defined; the debug
  assert stays. Masks (`ε & 63`) were measured and rejected (+8 to +40 SASS instructions
  per W = 2 kernel). Cost of PTX shl, measured (RTX 4090, three interleaved A/B rounds at
  N = 13 and 16, where it is the only change; eval workspace
  `results/openitems/emitter/`): C bit-identical by hash; per-case median call ratios
  0.990–1.002 on 4096³ / 1024² × 16384 / DF / `imd_pack` 2048², pack phases 0.992–1.002;
  registers and spills unchanged.
- **§3.4.1 the s bound.** 10·255·255 noted as loose; exact maxima certified per W
  (`cert_pack_words`: 210,472 / 264,156; 258,500 over X ≤ T).
- **§3.4.1 / §6c W-aware occupancy.** `kernel_occupancy` reports the most constrained W
  instantiation of each pack kernel (fewest CTAs per SM, then most registers; the W = 3 row,
  as before: shipped calibration unchanged); `kernel_occupancy_w` per W; test hook
  `imd_test_kernel_occupancy_w`, benchmark hook `imd_prof_kernel_occupancy_w`, imd_bench
  `--methods occupancy` record field `kernels_w`. Test: `contract_api` asserts equal CTAs
  per SM over W for every pack kernel and that the reported row is the most constrained.
- **§3.4.1 W = 1 for N ≤ 8: evaluated, not adopted** (above; NOTES-BFEPS §4). The
  certificate keeps the fact that N = 8 is the largest N with T < 2³².
- **Tests.** contract suite `eps` adds N = 2, 5, 6, 7; `packed_packed` adds two ε-heavy
  `imd_pack` cases at N = 6, 8.

## v1.17 (§6c performance model: KF element chain, PACK L2-hit re-reads, f_ref rule; host API 12, calibration format 2 with optional keys; no semantic, layout, launch, pack-format or TLA+ change)

- **§6c.1 two additive features** (from the open-items model analysis, evaluation workspace
  `findings/openitems-model-analysis.md`): KF x₄ = `elem`, the busiest SM's per-element chain
  (`kf.chain` with u = E; one launch-shape helper for both), not clock-bound; PACK x₃ =
  `dram` = x₁ − the operand re-reads that hit L2 (call working set S ≤ L_hit =
  `warm.pack_l2_bytes`), not clock-bound. Both in the slots the eps path freed; every
  existing feature keeps its meaning. The analysis's KF read / write split is not adopted
  (needs a seventh slot or a format bump).
- **§6c.1 f_ref rule:** the vertex of a parabola through the LOO scan within ±100 MHz of its
  argmin (was: the argmin); the nested leave-one-out figure is printed and reported.
- **§6c.1 cache:** the per-phase times of `--l2 flush` records are REPEAT state; `perf_fit.py`
  raises them to COLD (latent: no record used it).
- **§6c.2 keys:** `kf.elem`, `pack.dram`, `warm.pack_l2_bytes` optional (absent = 0, written
  only when non-zero, format-2 keys); the `op_eps` keys retired (accepted, discarded, never
  written); format 2 kept, with the reason (no file is read with features it was not fitted
  with; §6c.5 amended). `perf_fit.py`: L_hit = 0.8 × the nominal L2 in the seed, CLOCK_BOUND
  kf (2, 3, 5) / pack (2, 4), `--drop CLASS.FEATURE` for feature A/Bs. Sweeps: the L2-step
  cases in the calibration sweep, N = 18–20 in the hold-out. `host_perf`: the synthetic
  calibration iterates its keys sorted (it depended on the string hash seed) and the
  capped solve is checked as a bracket at the bisection's 0.05 MHz (§6c.4).
- **§6c.3 host API 12:** `imd_perf_calib_t.pack_l2_hit_bytes` appended (3688 bytes);
  `imd_perf_param_name` KF 4 `"elem"`, PACK 3 `"dram"`, KL 4 NULL; a v7–v11-sized struct
  reads L_hit = `warm_l2_bytes` when `pack.dram` > 0, else 0 ([review 2026-10-06]; was 0:
  §6c.3). Python `host.API_VERSION` 12.
- **§6c.4 tests:** see §6c.4 [v1.17].
- **Shipped calibration** (2026-10-06, branch `model-fix`, NOTES-B2 §11): one full refit on
  the current library (all 14 sustained windows recollected, 60 s pre-heat; the cache sweep;
  the hold-out with N = 18–20): `kf.elem` 534 ns, `pack.dram` 0.844 ps/B, L_hit 60.4 MB,
  f_ref 2491 MHz (nested leave-one-out 1.19 %), e_tc 0.721 pJ/op, e_byte^f 289 pJ/B. Hold-out
  (189 calls) median / p90 / max: < 100 µs 3.8 / 12.9 / 23.2 %, 100 µs – 1 ms 2.6 / 9.6 /
  13.9 %, ≥ 1 ms 2.2 / 6.6 / 16.8 % (old θ with eps ignored: 5.8 / 13.4 / 21.7, 3.5 / 14.1 /
  21.3, 2.1 / 7.7 / 15.8 %); K3 phase 13.3 / 22.4 → 4.9 / 14.7 %.
- **Review of `model-fix` (2026-10-06, branch `openitems-final`):** §6c.1 says S is a proxy
  (no K4 planes, no pre-packed residues; L_hit fitted on squares only), the clock-bound list
  no longer names `op_eps`, §6c.7 states the clock (0.8–3.4 % high) and power (every small
  high-clock window low, not only r_mm) gaps, NOTES-B2 §11 the hold-out caveats; §6c.3 the
  v7–v11 view reads L_hit = `warm_l2_bytes` when `pack.dram` > 0 (exact for the shipped
  file). §3.4.1 [v1.16]: `shl_clamped` at counts ≥ 64 is tested.
- **Docs caveats of the 2026-10-06 review** carried into DESIGN: §6c.7's shipped sustained
  figure (u_sq2048 −1.4 %, f_ref chosen on the same metric, capped clock 2–4 % high) and the
  v1.15 changelog's hold-out claim (the gain is the eps retirement, not the refit).

## v1.18 (§6c.8 performance model: cache-state terms, PACK set with K4's planes, busy power and boost clock; host API 13, calibration format 2 with optional keys; no semantic, layout, launch, pack-format or TLA+ change)

- **§6c.8 time model** (from `findings/perfminor-time.md`): the cache term gains a per-call
  REPEAT saving `warm.call_ns` (c, when res > 0) and a clean-L2 credit `warm.clean_byte` (ω,
  `−ω·min(F − I, L_w)` on the call phase) in a new state `IMD_PERF_CACHE_CLEAN` (the bench's
  flushed records and method pair's COLD); COLD keeps its meaning (inputs evicted by other,
  dirty work) and stays the default. The PACK group working set S counts one group's W planes
  when `warm.pack_l2_planes = 1`, with L_hit = 1.1 × the nominal L2. Not adopted: the KF read /
  write split (`kf.row` is non-zero in format-1 files: a format bump for ≤ 0.8 points), a KF /
  PACK / KL roofline (refuted), a per-modulus E = 4 KF term (fits ≈ 0), a fixed L2-hit price.
- **§6c.8 power model** (from `findings/perfminor-power.md`): busy-GPU power `power.act_w` with
  the busy share b = 1 − launches · `power.act_gap_us` / t_B, DRAM energy `power.dram_pj` for
  calls whose footprint exceeds `warm.l2_bytes`, the boost clock `power.clock_boost_mhz` (BURST
  at f_top, SUSTAINED solved on [f_min, f_top], time flat above f_ref, `clock_mhz` out = the
  clock used), γ = 2.5 as a prior; `perf_fit.py --power-form busy` (default), f_top from the
  uncapped windows, f_ref scan down to 2255 MHz.
- **§6c.2 keys:** the seven keys optional (absent = 0, written only when non-zero, format-2
  keys); `warm.pack_l2_planes` 0 or 1, `power.clock_boost_mhz` 0 or ≥ `power.clock_eff_mhz`.
- **§6c.3 host API 13:** seven appended `imd_perf_calib_t` fields (3744 bytes),
  `IMD_PERF_CACHE_CLEAN = 3`; a v7–v12-sized struct reads the fields as 0 (degraded view of a
  v1.18 file). Python `host.API_VERSION` 13.
- **Bench:** `imd_bench --methods sustained --mode direct` writes " mode=direct" into the
  variant. Sweeps: the four PACK-set cases in `sweep_calib.txt`, 1024³ N = 8–16, 1536³ and 768³
  pairs in `sweep_cache.txt`, 12 sustained windows in `sweep_sustained.txt`, three S-rule shapes,
  three cache pairs and the 352³ N = 14–20 series in `sweep_perf_validate.txt`;
  `perf_validate.py` reports the pairs.
- **§6c.4 tests:** see §6c.8 "Tests".
- **Shipped calibration** (2026-10-06, branch `perf-minor`, NOTES-B2 §12): one full refit on the
  v1.18 library (calibration and cache sweeps and hold-out measured; 26 sustained windows, 24 of
  them reused from the model-fix and perfminor runs on the same kernels): L_w 0.90 × L2, Δ 1.29
  ps/B, c 1.23 µs, ω 0.169 ps/B; PACK S with planes, L_hit 83 MB; busy power e_tc 0.577 pJ/op,
  e_alu 92 pJ/op, P_act 154 W, e_dram 96 pJ/B, γ 2.5, boost 2788 MHz, f_ref 2484 MHz (nested
  leave-one-out 1.62 %, v1.15 form on the same 26 windows 2.34 %; clock 1.3 vs 6.6 %, power max
  9.9 vs 80.6 %; after the review amendments below: f_ref 2483 MHz, nested 1.40 % vs 2.70 %,
  clock 0.7 vs 6.6 %, power max 9.8 vs 80.6 %). Hold-out (199 calls) against the v1.17 calibration on the same records:
  < 100 µs 3.9 / 10.8 / 15.8 → 3.7 / 12.2 / 19.0 %, 100 µs – 1 ms 2.7 / 10.3 / 15.7 → 2.7 / 6.9 /
  13.4 %, ≥ 1 ms 2.4 / 6.6 / 16.0 → 2.6 / 6.4 / 16.4 % (no class worse beyond run-to-run noise against that calibration; [review] against a v1.17
  refit on the same records ≥ 1 ms median 1.6 → 2.6 and < 100 µs p90 9.8 → 12.2, a trade-off;
  KF phase 8.9 / 24.4 → 6.6 / 21.1, K3 5.2 / 16.3 → 4.2 / 14.0; 1024³ N = 14 +14.9 → +6.3 %).
  Reviewer questions: COLD's meaning, `clock_mhz` on the NVML scale, and whether to keep c
  (it costs the small REPEAT calls ≈ 1 µs; NOTES-B2 §12 ablations). **Superseded by
  perf-minor-noc below (c dropped).**
- **Review amendments** (two independent reviews of `perf-minor`, 2026-10-06; branch
  `perf-minor-final`, NOTES-B2 §12 "Review"): **the power and the SUSTAINED clock divide by
  the request's cache-state time** t(f) (they used the COLD time in every state while the fit
  uses the windows' measured REPEAT time; a COLD request is unchanged); `perf_fit.py`
  recomputes the boost clock inside every leave-one-out / nested fold and warns on a busy fit
  without an uncapped window; the power keys refit offline on the same records (time
  parameters unchanged). Text: the S-rule hold-out shapes and the 3072 × 1024² pair as built,
  `pack.byte` identifiability not borne out, the dirty-write-back reading of ω marked as a
  hypothesis (the WARM data argue against it), COLD unvalidated for res > 0, the clean credit's
  early saturation, the D_dram step and the uncapped BURST power, and the v7…v12-sized struct
  as a stated exception to §6c.2 (the v11 L_hit fallback is no longer exact for the shipped
  file).
- **c dropped** (perf-minor-noc, the independent reviewer's binding verdict, CPU only;
  NOTES-B2 §12 "Decision: c dropped"): the shipped calibration is the refit of the same records
  with `perf_fit --cache-terms delta,omega` (now the default): `warm.call_ns` absent (= 0),
  L_w 0.85 × L2, Δ 1.31 ps/B, ω 0.104 ps/B; f_ref 2479 MHz, nested leave-one-out 1.42 % (with c
  1.40), clock 0.7 %, power max 9.2 %; hold-out < 100 µs 3.7 / 12.2 / 19.0 → 3.8 / 10.7 / 17.1 %,
  100 µs – 1 ms 2.7 / 6.9 / 13.4 → 2.7 / 8.2 / 13.3 %, ≥ 1 ms 2.6 / 6.4 / 16.4 → 2.5 / 6.4 /
  16.4 %; 1024³ N = 14 +6.3 → +10.8 %; small COLD calls ≈ 1.2 µs faster (64³ N = 14 16.68 →
  15.45 µs). The per-call saving is not identifiable from these records: an open item tied to
  the launch-timing question (power reviewer item 5). The key stays supported (optional,
  0 = off); the struct and format are unchanged. Text: COLD described as a modelling convention,
  the write-back reading of ω as a hypothesis.

## v1.19 (§6g adaptive precision, stages 0, 1 and 3: precision phases on host API 13, N chosen per phase on the host, the K-segmentation recipe; stage 4 closed and stages 5 and 6 deferred by their gates; no semantic, layout, launch, pack-format, API, calibration-format or TLA+ change)

Design: docs/NOTES-C1.md §4.1 (stage 0), as amended by the independent reviewer's binding
decisions (IR1–IR29, NOTES-C1 §5.3).
- **§6g (new):** C1(a), per-call N, as precision phases: a plan and a pack per N; the N-range
  plan and the Ozaki-I slice mode parked with their triggers; C1(b) on the host (§6f, and
  NOTES-C1 stage 1's `PrecisionPlan`).
- **§6a contract text (from API 13 on):** F1 (`imd_workspace_size[_packed]` nondecreasing in
  N and g over the (N, g) where the query succeeds, and the downward statement: success at
  (N_max, g_max) implies success with no more bytes at every smaller (N, g), or the packs' κ
  rejection), F2 (`pack_bytes` nondecreasing in N), F3 (`imd_pack`'s write extent, a packed
  contract's read extent) and their consequences C-ws (one workspace for every phase) and
  C-pack (one packed buffer for every N′ ≤ N_max). They describe the current formulas: no
  API version change. An earlier text of this version also promised that the N where the
  query succeeds form an up-set. That is false: a size past uint64 (INVALID_VALUE) grows with
  N and g, so the succeeding N form an interval. The up-set was withdrawn before merge
  (independent re-review of stage 0; NOTES-C1 §5.4 S0-2), and `host_footprint_monotone`
  now covers two shapes that pass uint64. F3's second half ("`imd_pack` writes no byte beyond `pack_bytes`")
  became ABI only after `pack_write_extent` and `pack_write_extent_memcheck` passed on every
  `imd_pack` launch shape (NOTES-C1 IR17), the radix > 1 row mapping included, with the
  coverage asserted by the test (independent review of stage 0, must-fix 1). `imd_pack`'s
  own workspace is outside F1–F3 (its independence of N is I1). The header comments of
  `imd_pack`, `imd_workspace_size[_packed]` and `imd_pack_layout` say so.
- **§6g.6 measured** (NOTES-C1 IR10): the first call at a never-launched N takes at most 9 µs
  more than a steady call, end to end (5–7 µs at 1024³, under 0.3 % at 4096³), less than one
  re-pack of an operand (0.018–0.42 ms of GPU time). The first plan creation at that N
  (0.011–0.150 ms) is a one-time host cost per N per process, paid before the iteration loop.
  Cold capture works and its first replay is 0.98–1.10× the steady one, so no plan-time
  preload is added. (This corrects the summary in commit 8b58101's message, which said "at
  most 8 µs" and compared plan creation with a re-pack.)
- **Certificates, not ABI:** I1 (pack offsets independent of N, res(N′) a prefix of
  res(N_max)), I2 (plan device bytes independent of N and g), I3 (packing never grows the
  workspace): `params.py cert_footprint_monotone` (also F1, F2, C-ws, C-pack).
- **§6b.7:** the re-pack-in-place rule (the last pack's key only; no graph replay after a
  re-pack with another key; WAR ordering). §6b.10 #15 notes that C1(a) keeps N in the key.
- **Python:** `PackSpec.with_moduli`, `device.PackedBuffer` and `PackRecord` (host-side
  record of the last pack, refusal of a stale plan with `PackMismatchError`), exported from
  `immerdink`; `bench/python_overhead.py` measures the guard (methods `packed`,
  `packed_guard`); `bench/first_call.py` measures the first call at a new N (§6g.6).
- **Tests:** `host_footprint_monotone` (CPU, in `tools/test.sh`'s CPU-only pass, outside the
  GPU lock), the `python_device` CPU part (a fake libimmerdink), and on the GPU
  `packed_repack_n`, `workspace_shared_n`, `graph_phases` (cold capture), `pack_write_extent`
  and `pack_write_extent_memcheck` (`tests/test_phases.py`, `tests/memcheck_pack_extent.sh`).
- **Provenance** (NOTES-C1 IR9, IR14): this worktree's libraries rebuilt first; host API 13,
  `imd_perf_rank`, `imd_perf_predict_pack` and `imd_err_min_moduli` present;
  `libimmerdink_host.so` byte-identical to the library that produced NOTES-C1's [M] figures.
  The design-phase scripts and records are in immerdink-eval (`scripts/c1/design-phase/`,
  `results/c1/design-phase/`), listed with their sha256 in
  `findings/c1-design-phase-manifest.md`. **M-b** (NOTES-C1 §4.3, IR9, IR15, IR23; CPU) re-ran
  them on this worktree's library: every stored [M] record reproduces byte for byte, as do
  stage 2's, and no cited [M] figure here changed. §3.8 adds the [M] composition of the
  K-segmentation cost next to the [E]; §6g.10 and §7 add the share-group raise cost
  (immerdink-eval `findings/c1-m-b.md`). *M-b review fixes* (approve-with-fixes; NOTES-C1
  §5.8): §7's slice row gives the most a pack serving every N could save, from the M-b records,
  instead of the raise cost; §6g.10 states the raise figures' level grid, base and modelling
  limits; §3.8 names the [M] composition as the K-segmentation reference.
- **Stage 1** (NOTES-C1 §4.2 with IR2, IR3, IR7, IR8, IR15, IR24; Python only, host API 13):
  - **§6g.10:** `immerdink.precision.PrecisionPlan` (new module; planner.py unchanged):
    per phase, `min_moduli`, then one rank per contraction over the levels ≥ the floor with
    reuse = 1, re-keyed by contract_s + Σ pack_s(role, N) / reuse(role) (per-role reuse
    'call' / 'iteration' / 'phase' / int, the 'phase' fallback recorded); pack-mode masks per
    contraction and a BUDGET status (rank's budget meaning kept); share groups ranked jointly
    over every level ≥ the group floor; an optional final `"DIRECT"` level; a monotone floor
    with `reset()` (`monotone=False` for relaxed Krylov methods); p = P / C for TYPICAL
    targets; `configs()` (every level's plan, created before the loop) and `footprints()`
    (per-level and maximum bytes from the queries).
  - **§6g.11:** `precision.rho_from_output` (β = 0, α = 2^e only), the rigorous ρ of an
    ELEMENTWISE target from a previous output; spec/params.py `rho_recipe` is its reference,
    and `cert_rho_recipe` checks it in exact rationals.
  - **§6d.1:** the module row. **Tests:** `host_precision` (CPU, in the CPU-only pass).
    **Example:** `bench/accuracy/schedule.py`, the forcing rule as caller code.
  - **Stage-1 review fixes** (independent reviewer, approve-with-fixes; verdicts in NOTES-C1
    §5.5): a share-group member at DIRECT leaves the group's N coupling for the phase (it
    had dragged every member to DIRECT and mislabelled a missed target `BUDGET`), and
    `BUDGET` is reported only for a candidate that meets its target (§6g.10); a 1-D norm
    bound of `rho_from_output` is always axis −2 (rows) or −1 (columns), since numpy alone
    certified ρ̂ ≈ 1 where ρ ≥ 100 on a batched C or mixed bounds with m = n (§6g.11);
    `footprints()` takes the maximum over every succeeding g, so the plan device bytes do not
    rest on I2; `cert_rho_recipe` covers every N = 2–20 and the native FMA chain;
    `PrecisionPlan` lowers at the lowest level for flops_eq; §6d's carve-out for
    precision.py.
  - **Stage-1 re-review fixes** (approve-with-fixes; NOTES-C1 §5.5): a `rho_from_output`
    bound of ndim ≥ 2 must lie along its own axis (a row bound shape[−1] = 1, a column bound
    shape[−2] = 1) and carry none or all of C's batch dims, since numpy's per-batch (b, m) /
    (b, n) norms without keepdims were paired with C's rows or columns at b = m and certified
    ρ̂ ≈ 0.006 ρ (§6g.11, MF3); S1-10 (a shared role's reuse stays fixed while a member is at
    DIRECT) is binding and documented in §6g.10 and the `reuse` and `Phase` docstrings.
  - **Stage-1 second re-review fixes** (approve-with-fixes; NOTES-C1 §5.5 S1-7, MF4): on a
    batched C (C.ndim ≥ 3) a `rho_from_output` bound is a scalar or has exactly C.ndim dims
    (each batch dim C's or 1, a row bound shape[−1] = 1, a column bound shape[−2] = 1), and
    every 1-D or 2-D bound raises a ValueError that names the keepdims spelling, since
    per-batch scalars without keepdims, (b,) or (b, 1), were paired with C's rows at b = m
    or columns at b = n and certified ρ̂/ρ = 0.0081–0.0113 (§6g.11). This narrows MF2's "for
    any batch dims" to C.ndim = 2, where the rule is unchanged, and subsumes the
    partial-batch check. `host_precision` checks the batched forms against each batch's 2-D
    call and the rejected ones in exact rationals (§6g.12). The S1-10 optimism bound reads
    "the DIRECT members' summed share of the calls, per iteration" (§6g.10, the `reuse` and
    `Phase` docstrings).
  - **Stage-1 third review fixes** (approve-with-fixes, should-fix items only; NOTES-C1 §5.5
    S1-7, S1-11): `rho_from_output` converts a norm bound to float64 exactly or refuses it
    (an integer above 2⁵³: ValueError; a long double, an object array of Fractions, Decimals
    or big ints, complex or bool: TypeError), since rounding to nearest could fall below the
    caller's bound; §6g.11 and the docstring say that C and its bounds are in the GEMM view
    (a tensor output with interleaved modes is permuted and reshaped first);
    `host_precision` adds bounds shared over one batch axis only on a 4-D C and the dtype
    cases (§6g.12).
- **Stage 2, CPU evidence** (NOTES-C1 §4.3 with IR1, IR3, IR6, IR7, IR11, IR16, IR19, IR27;
  M-a and M-d1; no library, ABI or §6 change; host API 13):
  - **spec/oracle.py** `kweights`, the exact K-weight rule with its multi-partner mode
    (union extrema, pairwise guard G = min_p G_p), spec/test_oracle.py (12). The study:
    `bench/accuracy/kw_engine.py`, `c2_fixtures.py`, `c2_gate.py`, `c2_summary.py` and
    `md1_converge.py`; ctest `c2_engine_selftest` (CPU). Summary immerdink-eval
    `findings/c1-evidence.md`, records `results/c1/m_a/`, `m_d1/` and `real/` (untracked).
  - **The gates, applied by the pre-registered rules** (NOTES-C1 §4.5–§4.7): stage 4
    (caller-side K-equilibration) fails its gate and is **closed** (no real psi4 contraction
    of the ≥ 2048³ class has guard = 1); stage 6 is **deferred**, since it needs a shipped
    stage 4; stage 5 was already deferred (IR26). Their triggers are in §7. The accurate-mode
    row of §7 cites M-a, and its Appendix B trigger is not met.
  - **M-d1** meets its success criterion on every system (the forcing-rule ramps against
    fixed N = 14), the lagged ρ̂ of §6g.11 missed in 1 of 228 MET calls (IR7 closed), and
    the report-only Davidson and CG cases give the caller guidance of §6g.12.
  - Reviewed four times (approve-with-fixes; NOTES-C1 §5.6). Provenance (NOTES-C1 Status):
    the records gates (i) and (ii) are evaluated on, and those behind M-d1's criterion, are
    reproduced bit for bit on committed code (121 records at 5925219, M-d1 jacobi, davidson
    and cg at 928b49c; the ρ-miss records are tied to it by sha256, NOTES-C1 §5.6 S2-18).
    Tier S, which sets the deadband d* = 3, is not (8b58101-era, 4 of 245 runs
    smoke-checked), nor is Tier S's share of gate (iii); the gate verdict holds at d = 0 as
    well.
- **Stage 3** (NOTES-C1 §4.4 with IR5, IR12, IR15; C2; no ABI change, host API 13):
  - **§3.8 "Segmented accumulation":** the K-segmentation caller recipe (S calls, β = 1
    after the first, moved base pointers; per-band σ, flags, RN acceptance and packs), its
    error bound with the per-metric forms and caveats (whole-row ℓ1 ratios do not transfer
    to a band; ELEMENTWISE loses the single rounding), the rigorous request transform
    `t′ = RN(RN(min(t_eff, 1)(1 − 2⁻⁵⁰)) − 4S·2⁻⁵³)`, and M-a's evidence on when it pays.
  - **PROOFS L7** (L7a recursive summation γ_{S−1}, L7b the segmented bound, L7c the metric
    forms, L7d the request); L5 and L6 stay reserved (stage 4 closed, stage 5 deferred).
    **Lean:** L3c (ν form, compositions, normal-range final rounding), the summation lemma
    and L7 `kseg_compose` (proofs/, NOTES-C1 IR5).
  - **spec/params.py** `kseg_gamma`, `kseg_unit`, `kseg_request` and `cert_kseg_composition`
    (L7 against exact rationals over every M5 family and constant rows, S = 1–4,
    N ∈ {3, 4, 5, 8, 14, 20}, RN and TRUNC, per-band ν, every metric, underflowing products;
    the request transform; the whole-row-ν counterexample, the lost single rounding and the
    necessity of γ). **spec/oracle.py** `segment_bounds`, `contract_segments` (a literal
    composition of the band calls); spec/test_oracle.py (13).
  - **§7:** accurate mode (re-examined with M-a's figures), K-equilibration in the library
    (closed by its gate, with the trigger), K-segmentation as a library mode, larger moduli
    (with arXiv:2603.10634), the N-range plan, multi-N / round-to-odd packs, device-chosen N,
    Ozaki-I slices and the K5 statistics, each with its trigger where deferred.
  - **Stage-3 review fixes** (independent reviewer, approve-with-fixes; verdicts in NOTES-C1
    §5.7): a band of k_s = 1 is K-empty and runs DIRECT, as the library dispatches it
    (`contract_segments` emulated it, so it was not the bit-exact reference there; test (13)
    now also runs the recipe's calls through `contract_tensor` on moved sub-views); a
    rigorous ELEMENTWISE ρ_S from a previous segmented output, N_S/(|C_S| − W̄·N_S) rounded
    outward with `params.kseg_wbar_up` (PROOFS L7c (7), `cert_kseg_composition` (7)), in
    place of the unsound N_S/|C|; the transform scoped to α = 1 or 2^e and β = 0, with r and
    ρ_S rounded upward, its guarantee stated as W ≤ t̄ and "t′ ≤ 0" as "no N certified"; the
    underflow hypothesis of NORMWISE and ELEMENTWISE; the reference of a non-contiguous
    partition; `contract_segments` without A and B at α = 0.
- **Round 2, the stage-0 GPU tests** (NOTES-C1 §4.11; 2026-10-07; no change): pass on
  committed code at 4ba78b1 (host API 13; `libimmerdink.so` sha256 `c4bdcc4f…`,
  `libimmerdink_host.so` `466acdb4…`): `packed_repack_n`, `workspace_shared_n`,
  `graph_phases` (cold captures at N = 10 and, packed, N = 12), `pack_write_extent` (113
  packs, 612 packed contracts, all 42 KL launch shapes, 12 at radix > 1) and
  `pack_write_extent_memcheck`, with `tools/test.sh`'s 56 ctest tests, `params.py --check`
  (33 of 33) and `spec/test_oracle.py` (immerdink-eval
  `findings/c1-round2-stage0-gpu-tests.md`).
- **M-c, the small-N hold-out** (NOTES-C1 §4.3, IR4, Q24; 2026-10-07; no library,
  calibration or ABI change):
  - Tooling: `bench/sweep_mc.txt` (160 calls), `bench/sweep_disjoint.py` (absence from the
    calibration set by GEMM-view extents), `imd_bench --data decay_opp`, and the Q24 cases
    `bench/sweep_calib_n.txt`.
  - Outcome: **failed**. The < 100 µs median is missed (15.8 % against 15 %) through two
    effects, each necessary: a small-N optimism on the KF path (sq1024 at N = 4) and the
    KL path with a narrow n (the Jacobi view), where KL A keeps only part of its REPEAT
    credit and the narrow pack KL B is slow in every cache state. The Q24 refit was not
    adopted, so the shipped calibration is unchanged and the [M, model-only, outside
    hold-out] marks stay (§6g.9; NOTES-C1 Status, §5.9).
  - Review (approve-with-fixes; NOTES-C1 §5.9, binding): M-c failed, no partial pass; the
    MC-3 feature under §6c.5's feature rule, its §6c text first, is authorized as the only
    path to a pass; M-d2 waits for it. `bench/sweep_disjoint.py --self-test` is ctest
    `sweep_disjoint_selftest`.
  - Second review (approve-with-fixes; NOTES-C1 §5.9 MC-3a, MC-6, binding): the per-phase
    account above replaces "the REPEAT credit" alone. MC-3 is amended as MC-3a: §6c text
    from a per-phase decomposition of the allowed records (never M-c's), at most one
    optional key per physically distinct effect (two in all: the REPEAT credit; the KL
    narrow-pack time), the cache key fitted against the KL A and KF phase marks, three more
    pass conditions (KL A's per-phase error improves; no phase class worsens in-sample by
    more than 1 point; a fresh narrow-KL hold-out, committed before the refit, meets every
    class target). The v1.18 hold-out is judged on all its cases and on the 69 whose GEMM
    view matches no fit case.
- **M-d2, pre-registered 2026-10-07 and run 2026-10-09: PASS** (NOTES-C1 §4.3, §5.12, §5.10; 2026-10-07; no library,
  calibration or ABI change). `bench/accuracy/md2_time.py` (3254b15; re-registered after
  review at daf3583, after the re-review at 35eeb46, and at fce29e9 with the NaN-filled
  bit-exact check) has four modes:
  - `predict` (CPU) checks that the shipped calibration reproduces the registered bars, and
    that the library's embedded calibration equals the registered file;
  - `run` (GPU) times the 16-call schedules end to end, in-line (gated) and with the
    invariant operand re-packed per phase (reported). It refuses the M-d2 systems until
    the ordering of §5.9 is met, and any record the protocol's provenance checks or re-run
    rules would reject;
  - `judge` (CPU) applies the bands per system and data set by the pre-registered rules
    (a)–(d) of NOTES-C1 §4.3, and refuses records outside the protocol;
  - `selftest` (CPU) checks those rules on synthetic records (ctest `md2_time_selftest`).

  A smoke on two non-M-d2 shapes checked the tooling, at 3254b15, daf3583 and 35eeb46.
  - Review (approve-with-fixes; NOTES-C1 §5.10, binding): MD2-0 to MD2-7 upheld. M-d2 waits
    for MC-3a. The judge's verdict was keyed by system only, so one data set recorded twice
    passed; it is now keyed per data set, with the admissibility and provenance rules
    above. A caller note on packing the invariant operand is in §6g.12.
  - Re-review (approve-with-fixes; NOTES-C1 §5.10 "Re-review", binding): one normative
    records file, the driver frozen from the first M-d2 record on (MD2-8 amended), no
    automatic continuation after two inadmissible attempts, STOPPED or a refused file;
    library identity and `--md2-cleared`'s citation checked; re-registered at 35eeb46.
- **C1/C2 close-out** (NOTES-C1 §4.11, IR29; 2026-10-07; docs and docstrings only). No
  library, calibration or ABI *layout* change in v1.19: host API 13 (the later MC-3a key
  `warm.kl_drop` uses the previously unused `param[95]` of `imd_perf_calib_t`, same layout and
  size; a hand-built struct must zero `param`, §6c.9). Since the C1 design draft
  (39972b7) `src/` has not changed, `include/` only in comments (§6a F1–F3), and Python
  gained `precision.py`, `PackSpec.with_moduli` and `PackedBuffer` (§6g.5, §6g.10).
  - §6g.12: the M-d1 caller guidance (DIIS reset at a switch, a subspace's recompute and its
    +5 iterations for Davidson, Krylov started tight and relaxed by a tolerance scaled with
    the iteration count, the lagged ρ̂'s miss rate), carried over from NOTES-C1 §1.3; §6g.10, the
    `PrecisionPlan` docstring and `bench/accuracy/schedule.py` point to it.
  - §7: the K-equilibration row states stage 4's revival trigger in full and stage 6's
    trigger; the K5-statistics row states stage 5's cost bar without `ncu` (IR26).
  - Close-out review (approve-with-fixes, 2026-10-07): §6g.9 states `PrecisionPlan`'s
    shared-pack limit, the candidate member role `"A=B"` and its trigger; §6g.12's DIIS
    bullet says plain DIIS stalls near 1–2e−15 with or without switches; the header says
    stage 5 was deferred by IR26, not by the CPU study. IR26's "counters are admin-only"
    premise was wrong (GPU counters are open to all users). Stage 5's cost bar keeps its
    content (A/B timing at stock clocks under the GPU lock, the compiler's register count,
    the occupancy query) because it is a time bar and these measure time directly; `ncu`'s
    serialized replay does not reproduce production timing. `ncu` is not part of the bar but
    not forbidden: as a diagnostic, only with `--clock-control none` (its default, `boost`,
    locks the clocks), under the GPU lock (NOTES-C1 §4.6, "Correction"; the §7 row's
    "restated without `ncu`" stands).
  - The definition of done (NOTES-C1 §4.11): **C2 is done** (M-a published; stage 3 shipped
    with L3c and L7 in Lean; stage 4 closed and stage 6 deferred, each with its trigger;
    ROADMAP C2 retitled). **C1 was not at the close-out:** stages 0 and 1, the round-2 stage-0
    GPU tests and M-d1 were done and stage 5 deferred (T1–T3), but M-c failed. MC-3a was
    executed on 2026-10-09 and recorded (NOTES-C1 §5.11), with no pass (independent review,
    ruling A, binding, 2026-10-09). M-d2 then ran and passed (NOTES-C1 §5.12), and **Eric
    amended the done condition "M-c has passed" (decision, 2026-10-09; M-d2 passed on the
    shipped calibration): C1 is done as amended.** M-c stays FAILED and the marks stay.
- **Final review of C1/C2** (three independent reviewers, approve-with-fixes, 2026-10-07;
  NOTES-C1 Status, "Final review"). No library, calibration, ABI or TLA+ change: host API 13.
  - §3.8: `params.kseg_request` returns None, not 0.0, when no N is certified (rank reads a
    target of 0 as "no target"; normwise targets near 1e−15 are out of reach at S = 2–4),
    and step 3 requests t′ > 0 with basis BOUND; the cheap r states which norms must be upper
    and which lower bounds, skips zero rows, takes min(1, ·) and encloses RN-computed norms;
    the rigorous ρ_S carries the absolute underflow term `params.kseg_eta(S)` =
    S·2⁻¹⁰⁷⁴ ≥ (1+γ)·S·2⁻¹⁰⁷⁵, so it no longer assumes every D_s normal (a caller cannot
    check that); `params.rho_recipe` takes it as `eta` (PROOFS L7c (7)).
    `cert_kseg_composition` adds `kseg_wbar_up` exactly on a grid of U and S up to 2²⁶, a
    two-band K-empty witness of γ's size (L7 met to 0.996), ρ_S with the term on outputs
    whose last band underflows, and the None case; spec/test_oracle.py (13) adds a K-empty
    first band with β not a power of two.
  - §6g.10: `PrecisionPlan` refuses a share group whose shared roles read different packs, or
    that no level can pack (ValueError at construction; it had priced and sized two operands
    as one pack); the per-call cost of `phase()` and the cheaper pattern (re-run it only when
    a `min_moduli` result changes). §6g.5: `PackedBuffer` assumes one host thread per buffer.
    `host_precision` adds the refusals, the equal-`min_moduli` claim, ℓ1 ratios in its U
    check and the plans a phase names (`host.lowering` with each choice's `options()` and
    `pack_specs`, a share group's members with one PackSpec); `python_device`'s CPU part
    creates those Plans and Packs through its fake library.
  - §6g.6: the first call's extra enqueue time "much of it" overlaps the GPU (it said
    "most"), and the first-call records' provenance limit is stated. §7: K-equilibration's
    cost on the real operands is TFLOPS-eq first with time alongside, tagged [M, model-only,
    outside hold-out]; the K-segmentation row names R2b. `bench/first_call.py` records the
    host API.
