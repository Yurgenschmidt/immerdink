# Lean formalization of PROOFS.md: notes

Lean 4 `v4.34.1` with Mathlib tag `v4.34.1` (commit `d13f23b7`, pinned in `lake-manifest.json`).
Build and audit with `tools/check_proofs.sh`. The script does three things:
- it greps for `sorry`, `admit` and `axiom`;
- it runs `lake build`;
- it runs `#print axioms` on every headline theorem in `AxiomCheck.lean`, which must report
  only `propext`, `Classical.choice` and `Quot.sound`.

Setup: `cd proofs && lake exe cache get && lake build`. The prebuilt Mathlib oleans are
downloaded, never built from source.

## Map: PROOFS.md → Lean

| PROOFS.md | Lean (namespace `Immerdink`) | File |
|---|---|---|
| L1a (u exists) | `Scaling.exists_max_rule`, `exists_u_rn`, `exists_u_trunc`, `not_rule_succ`, `rule_mono`; boundary `no_rule_of_R_zero` | `Proofs/Scaling.lean` |
| L1b (‖a'‖ ≤ T) | `Scaling.budget_rn`, `budget_trunc` (+ `_int` forms over ℤ) | `Proofs/Scaling.lean` |
| L1c (CRT precondition) | `Scaling.crt_precondition` | `Proofs/Scaling.lean` |
| L1d (σ bound) | `Scaling.sigma_bound_rn`, `sigma_bound_trunc` (core: `sigma_bound_core`) | `Proofs/Scaling.lean` |
| L1 in the exact setting | `Scaling.row_rn`, `row_trunc` (`RowExp`, `κ = kappa k = ⌈√k⌉`, maximal `u`) | `Proofs/Scaling.lean` |
| quantizers | `Scaling.rne` (ties-to-even) and `abs_rne_sub`; `truncZ`, `abs_truncZ_le`, `abs_truncZ_sub_lt` | `Proofs/Scaling.lean` |
| L2a (bounds) | `CRT.x_lt_P`, `xp_lt_two_P`, `xp_lt_word` | `Proofs/CRT.lean` |
| L2b (congruence) | `CRT.x_N_modEq` (and `exists_q`: the `q_l` exist) | `Proofs/CRT.lean` |
| L2c (exact recovery) | `CRT.z_eq` (any `H` with `2H+1 ≤ P`), `z_eq_of_even` (PROOFS form), `z_eq_floor` (DESIGN form) | `Proofs/CRT.lean` |
| L3a (ℓ1 form) | `Error.l1_form` | `Proofs/Error.lean` |
| L3b (normwise) | `Error.normwise_rn`, `normwise_trunc` (core: `normwise_core`) | `Proofs/Error.lean` |
| L1 + L3 end to end | `Error.end_to_end_rn`, `end_to_end_trunc`; `zero_row`, `q_zero_rn` | `Proofs/Error.lean` |
| L4a (no saturation) | `Chunk.partial_sum_bound`, `satFold_chunk_eq_sum`, `tree_chunk_eq_sum` (generic: `satFold_eq`, `AccTree.eval_eq_sum`) | `Proofs/Chunk.lean` |
| L4b (carry) | `Chunk.epi_modEq_total`, `epi_mem_range`, end to end `carry_correct` | `Proofs/Chunk.lean` |
| L4c (int32 epilogue) | `Chunk.epi_input_range`, `epi_input_int32`, `design_constants` | `Proofs/Chunk.lean` |
| "L1c + L2c ⇒ exact" | `Exact.emulated_product_exact` (L1c + L4 + L2c chained) | `Proofs/Exact.lean` |
| L3c (1), the ν form | `Error.l3c_unit` (E = `unitE δ ν_a ν_b s d k`) | `Proofs/Error.lean` |
| L3c (2), the compositions | `Error.round_step`, `l3c_normwise`, `l3c_elementwise`, `l3c_absolute` (with the subnormal term η) | `Proofs/Error.lean` |
| L3c (3), the native FMA chain | `Summation.fma_chain` (γ_k) | `Proofs/Summation.lean` |
| L7a (recursive summation, γ_k) | `Summation.recsum` (power form `recsum_pow`; `pow_sub_one_le_gam`, `gam_nonneg`) | `Proofs/Summation.lean` |
| L7b (segmented bound) | `Segments.kseg_compose` (`kseg_compose_sub` with gradual underflow; per band `band_bound`); from the band rows through L3c (1): `kseg_rows` | `Proofs/Segments.lean` |
| L7c (metric forms) | `Segments.bands_norm_le` (N_S ≤ ‖a‖‖b‖), `kseg_normwise` (NORMWISE); (3) the underflow term: `kseg_compose_sub`; E monotone (`E_s ≤ E(k)`): `Error.unitE_mono`; (7) ρ_S from the output: `kseg_wbar_le`, `kseg_rho_le`. The ELEMENTWISE and ABSOLUTE forms of (2): see the issue #12 row below | `Proofs/Segments.lean`, `Proofs/Error.lean` |
| L7d (the recipe's request) | `Segments.gam_mul_le`, `kseg_request_core` (the exact half) | `Proofs/Segments.lean` |
| L7c (2) ELEMENTWISE / ABSOLUTE, (7) without the normal range | `Segments.kseg_wbar`, `kseg_elementwise`, `kseg_absolute`; `kseg_wbar_sub` (W̄·N_S + (1 + γ)·S·η from `kseg_compose_sub`) and `kseg_rho_le_sub` (ρ_S from it; DESIGN §6g.11 `rho_from_output` is `kseg_rho_le`'s shape) | `Proofs/Segments.lean` |
| L5a (cancellation) | `KWeights.kweight_cancel` | `Proofs/KWeights.lean` |
| L5c (the guard) | `KWeights.norm_le_of_rule` (L1b, norm form), `kweight_ratio` (core), `kweight_ratio_rn`, `kweight_ratio_trunc`, `kweight_guard` (any G), `kweight_guard_lt` (G ≥ 3), `sF_sq_le` (s² ≤ 1 + 2⁻²⁰ for k ≤ 2⁴⁰) | `Proofs/KWeights.lean` |
| L5d (transfer, exact half) | `KWeights.kweight_transfer_elementwise`, `kweight_transfer_absolute`, `kweight_transfer_normwise` | `Proofs/KWeights.lean` |
| L6a–c (a-posteriori certificate) | `Apost.apost_bound` (a), `apost_scaled`, `apost_rho`, `apost_elementwise` (b), `apost_pow2` (c) | `Proofs/Apost.lean` |
| R1 (κ in integers) | `Scaling.kappa_isqrt` | `Proofs/ShiftRule.lean` |
| R2 (u by one compare) | `Scaling.shift_estimate`, `u_eq_estimate` | `Proofs/ShiftRule.lean` |
| R3 (S width) | `Scaling.S_le`, `S_lt_two_pow_128`, `S_shift_lt` (the device's `L = 4·S < 2¹²⁷` for k < 2⁶³) | `Proofs/ShiftRule.lean` |
| R4 (SMod, int8, table, sign, byte weights) | `Residue.smod`, `smod_modEq`, `smod_congr`, `smod_range`, `smod_int8`, `smod_mul_abs_le`, `int8_mul_ge`, `smod_mul_ge`, `smod_table`, `smod_neg_odd`, `smod_neg_byte`, `byte_weights_modEq`, `byte_weights_le` | `Proofs/Residue.lean` |
| R5 (multiply-high reductions) | `Residue.mulhi_div_eq` (generic), `mod_p_bytes32` (packs), `crt_y_small32` (CRT digit, `mod_p_small32`), `barrett_epilogue` (epilogue), `residue_mul_reduction` (`residue()`'s `r·(2^ε mod p)`); `Moduli.barrett_epilogue_v113` (the epilogue hypothesis holds on every reachable input) | `Proofs/Residue.lean`, `Proofs/Moduli.lean` |
| R6 (the CRT word) | `CRT.R_dvd_x`, `x_le_P_sub_R`, `xp_le_peak`, `xp_le_crtPeak`, `peak_attained`, `crtPeak_attained` (the exact peak); `xp_ne_P`, `le_xp_iff_lt` (≥ vs >); `xp_mod_word` (no wrap); `center_word`, `z_word` (the DESIGN / TLA+ word form: sign bit set iff the subtraction is taken); `center_signmag`, `z_signmag` (the device's sign-magnitude form, `crt_finish`) | `Proofs/CRTWord.lean` |
| R6 (the design instance) | `Moduli.pm_pos`, `pm_le_256`, `pm_coprime`, `P_even`, `exists_q_design`, `T_ge_180`, `L32_exact`, `peak_needs_every_limb`, `chunk_constants`, `chunk_sum_range`, `epilogue_range_v113` (tight at p = 256), `qd`, `qd_inv` (the multipliers) | `Proofs/Moduli.lean` |
| capstone with SMod residues | `Residue.emulated_product_exact_smod` | `Proofs/Residue.lean` |
| capstone at the real moduli | `Moduli.emulated_product_exact_design` (any `q_l`), `emulated_product_exact_shipped` (the checked `qd`) | `Proofs/Moduli.lean` |

## Modelling choices

These choices are faithful to PROOFS.md, but they are not literal transcriptions.

- **Numbers.**
  - L1 and L3 work over ℝ, which contains the dyadic rationals.
  - L2 and L4 work over ℕ and ℤ. The CRT word is unsigned, so it is modelled in ℕ.
  - There is no IEEE model. The final binary64 rounding in L3 is the hypothesis
    `|D − z·2^{−σ−τ}| ≤ u·|D|`.
- **Indices.** Moduli are 0-based (`p 0 … p (N−1)`), and L2's `x l` is the word after
  `l` steps. L1 and L3 index a row by `Fin k`. L4 indexes `[0,k)` by `range k`.
- **Quantizers are abstract.**
  - RN means any `q : ℝ → ℤ` with `|q x − x| ≤ 1/2`.
  - TRUNC means any `q` with `|q x| ≤ |x|`. L3 additionally uses `|q x − x| ≤ 1`.
  - `rne` (ties-to-even) and `truncZ` are concrete instances that satisfy these
    hypotheses.
  - So L1 and L3 also hold for ties-away rounding. The ties-to-even requirement in DESIGN
    §3.3 is about bit-exactness with the oracle, not soundness. This matches the comment in
    `spec/MC_OzakiScaling__neg_quant_ties_away.cfg`.
- **"u is the largest integer with …"** is stated as `Rule u ∧ ∀ v, Rule v → v ≤ u`.
  - L1b uses only `Rule u`.
  - L1d uses only `¬ Rule (u+1)` (`not_rule_succ`).
- **"Any accumulation order" (L4)** is modelled two ways:
  1. `satFold`: a sequential fold of clamped additions over any list that enumerates the
     chunk (`Enumerates`, a multiset equality).
  2. `AccTree.eval`: an arbitrary binary tree of clamped additions. This covers MMA and
     split-K reduction shapes.

  The clamp is `max (−A−1) (min A v)`. The hypothesis actually used is
  `Σ_{h∈C}|r_h s_h| ≤ A`, which follows from `|r_h s_h| ≤ Q`, `|C| ≤ K` and `Q·K ≤ A`.
- **Partition (L4b).** The chunks are any family `C 0 … C (n−1)` that is pairwise disjoint
  with union `range k`. They need not be contiguous.
- **L3c and L7 (NOTES-C1 IR5, stage 3).**
  - The final rounding is the hypothesis `|D − y| ≤ u|y|` (normal range), or
    `|D − y| ≤ u|y| + η` where D may be subnormal; L3b's `|D − y| ≤ u|D|` is the other
    standard form and is kept for L3b.
  - Floating-point addition enters only through the per-step hypothesis
    `|fl(a + b) − (a + b)| ≤ u|a + b|`. There is still no IEEE model.
  - L7's bands are indexed by `ℕ` over `range (m + 1)`, so `S = m + 1` and `γ = γ_m`.
    `kseg_compose` takes each band's statistics abstractly (`|x_s| ≤ n_s`,
    `|y_s − x_s| ≤ E_s n_s`); `kseg_rows` instantiates them from band rows of lengths
    `ks s` (a dependent family `(s : ℕ) → Fin (ks s) → ℝ`) through `l3c_unit`. That the
    contraction's sum is the sum of the band sums is the partition itself and is not
    restated. `kseg_rows`' hypothesis `hσb` (`2^-σ ≤ 2‖a^s‖·sf/d`) cannot hold for a band
    whose row is zero (its right side is 0), so such a band enters only through
    `kseg_compose`, with `x_s = y_s = 0` and `n_s = 0` given directly.
  - L7d's FP evaluation of `t′` (two roundings, `params.kseg_request`) is not modelled;
    `kseg_request_core` proves the exact half, and `cert_kseg_composition` checks the
    rounding. Likewise L7c (7): `kseg_wbar_le` and `kseg_rho_le` are the exact statements;
    their outward-rounded evaluation (`params.kseg_wbar_up`, `params.rho_recipe`) is checked
    by `cert_kseg_composition` (7). The ρ_S without the normal-range hypothesis (DESIGN
    §3.8: the absolute term `(1 + γ)·S·η ≤ S·2⁻¹⁰⁷⁴`, `params.kseg_eta`) is the same
    argument as `kseg_rho_le` with `W̄·N_S + (1 + γ)·S·η` from `kseg_compose_sub`. Issue #12
    restates it in Lean: `kseg_wbar_sub` (the bound) and `kseg_rho_le_sub` (ρ_S).

- **Issue #12 additions.**
  - `smod` is defined from the non-negative mod (`Int.emod`), as DESIGN §3.4 requires.
  - The multiply-high reductions are stated on natural numbers, with `umulhi(s, m)` modelled
    as `⌊s·m/2^W⌋`. The u32 / u64 machine types are the bounds `s < 2²⁴`, `n < 2³²` and
    `n·e < 2⁶⁴`.
  - **Centring on the word.**
    - The unsigned `L_w`-bit word is a natural number below `2^{L_w}`.
    - The wrapped subtraction is `(x + 2^{L_w} − P) mod 2^{L_w}`.
    - Two's complement is `toSigned`, and the sign bit is `w ≥ 2^{L_w−1}`.
    - This is the word form of DESIGN §3.6 and `CRTRealization`. Borrows are not modelled; the
      sign bit is shown set exactly when the subtraction (whose borrow-out is 1) is taken.
    - The device computes the sign-magnitude form instead (`crt_finish`: `neg = x > H`,
      `|z| = P − x`). That form is `center_signmag` / `z_signmag`.
    - The limb carry and borrow chains are not modelled (TLA+ `CRTRealization`
      `RefinesWord`).
  - **The design instance.**
    - The 20 moduli are a `List ℕ`. `pm l` is `1` past the table.
    - Every concrete fact is proved by kernel evaluation (`decide`), never `native_decide`.
    - `tools/check_proofs.sh` compares the list with `spec/params_out.json`.
  - **L5 and L6.**
    - The weighted row `ã` enters L5c only through its own shift rule (L1b in norm form,
      `norm_le_of_rule`), so the guard holds for any `ã` with the same `(T, κ)`. Its relation
      `ã = a·2^w` matters only for L5a.
    - L6 takes `B̄ ≥ 2δκd + kδ²` and `X ≥ d²` as bounds, which covers DESIGN's ceilings.

## Discrepancies with PROOFS.md and proposed text

None of these invalidates a design claim. Each one is a missing or implicit hypothesis, or
a statement that the formalization proves in a more general form.

1. **L1, TRUNC needs `T ≥ 1`.** The setting assumes `2T − κ ≥ 1` only for RN. For TRUNC
   with `T = 0`, `R = 0`: no `u` exists (L1a), and L1d would divide by 0. L1a's proof
   already says "because R ≥ 1".
   Proposed setting line: *"TRUNC: `R = T²`, `c = 1`, and we assume `T ≥ 1`."*
   In Lean: `exists_u_trunc` and `sigma_bound_trunc` take `1 ≤ T`.

2. **L1d, and hence L3b, needs `a ≠ 0` and `b ≠ 0`.** For an all-zero row, the right-hand
   side of L1d is 0 and the claim is false. L1's setting does say "not all zero", but L3
   uses L1d for both operands without restating it. The L3b bound still holds for zero rows
   under DESIGN §3.2's convention `σ := 0`: then `a' = q(0) = 0`, `z = 0`, and only `u|D|`
   remains.
   Proposed L3 setting line: *"a and b are nonzero. For a zero row, σ := 0 gives a' = 0,
   z = 0 and |D − Σab| ≤ u|D|."*
   In Lean: `end_to_end_*` take `RowExp` for both operands; `zero_row` and `q_zero_rn`
   cover the zero case.

3. **L1 uses only the lower half of the definition of γ.** L1a–d need only
   `2^γ ≤ max_h |a_h|`, for `‖x‖ ≥ 1` and `S ≥ 2^F`. The upper half `max < 2^{γ+1}` is
   never used.
   Consequence: the norm budget and the σ bound stay sound if γ is underestimated, for
   example by an off-by-one `ilogb`. Accuracy is unaffected too, because `2^{−σ}` is bounded
   in terms of `‖a‖`, not γ.
   Proposed remark under L1: *"Only `2^γ ≤ max|a_h|` is used."* This is optional.

4. **L1's κ conditions are weaker than stated.**
   - L1b (RN) needs only `κ ≤ 2T` and `√k ≤ κ`.
   - L1d (RN) needs `κ < 2T`.
   - Only L1a needs `2T − κ ≥ 1`, so that R > 0.

   All three follow from the stated `2T − κ ≥ 1`. No change is needed. Note that `κ = ⌈√k⌉`
   is used only through `√k ≤ κ`.

5. **L2 holds for every P, not just even P.** `CRT.z_eq` proves L2c for any H with
   `2H + 1 ≤ P`.
   - PROOFS' `H = (P−2)/2` (P even) is `z_eq_of_even`.
   - DESIGN's `H = ⌊(P−1)/2⌋` (any P) is `z_eq_floor`.

   The implicit hypothesis `p_l ≥ 1` is explicit in Lean. L2b uses `q_l` only through
   `M_l·q_l ≡ 1 (mod p_l)`. Coprimality is needed only for CRT in L2c and for the existence
   of the `q_l` (`exists_q`).
   Proposed setting line: *"P need not be even: L2c holds for any H with 2H + 1 ≤ P,
   including H = ⌊(P−1)/2⌋."* This is optional.

6. **L4b needs at least one chunk** (`n ≥ 1`). Otherwise `W_{n−1}` is undefined, which is
   the `k = 0` case that DESIGN §3.7 routes around the GEMM.

7. **L4c's int32 inclusion is a separate numeric fact.** It does not follow from
   `Q·K ≤ A_max`: it needs `(p − 1) + Q·K ≤ 2³¹ − 1`.
   - `design_constants` checks this for `Q = 128²` and `p ≤ 256`. The maximum is
     2147221759 for `K = 131056`, the chunk length before DESIGN v1.13.
   - Since v1.13 the chunk length is `K_c' = 131008`, with maximum 2146435327
     (`spec/params.py` cert_kc, `spec/apalache/Epilogue.tla`). The bound is monotone in
     `K`, so `epi_input_int32` (chunks of length `≤ 131056`) covers it unchanged. Issue
     #12 re-instantiates it at 131008 with DESIGN §3.5's exact range
     (`Moduli.epilogue_range_v113`).
   - It also holds for `K = K_c = 131071`.

   PROOFS states the concrete numbers, so its claim is correct as written.

## Remaining work / not formalized

- The IEEE side is a hypothesis by design, in particular `fl` satisfying
  `|fl(x) − x| ≤ u|fl(x)|` in the normal range, and the per-addition bound.
- The floating-point evaluation of the requests stays with `cert_kseg_composition` and
  `cert_rho_recipe`: L5d's multiply by `1 ± 2^{−19}`, L7d's `t′`, and ρ̂_S's outward
  rounding.
- The limb-by-limb carry and borrow chains of the CRT word are TLA+ only (`CRTRealization`
  `RefinesWord`). Lean proves the word-level algorithm they refine.
- The σ int16 range (`cert_shift_range`) and the 32-bit Granlund–Montgomery insufficiency for 7
  moduli (`cert_barrett`) are certificate-only. Both are facts about the IEEE exponent range
  and about a rejected alternative, not about soundness.
- DESIGN §6f's *typical* estimate (Hoeffding plus a union bound) is probabilistic and has no
  formal proof. It is labelled an estimate, not a bound.
- The capstones are at the algorithm level: `Chunk.epi` is the mathematical mod, and the CRT
  word is a natural number. R5–R6 prove the machine-word steps separately, and they are not
  composed into one kernel statement. Not modelled:
  - the 32-bit decomposition of `umulhi64` (`src/device/modp.hpp`);
  - the pack path as one "emitted byte = SMod(a', p)" theorem.
- The multipliers are `qd`, the unique inverses in `[0, p_l)` found by search. These are the
  same numbers as `spec/params.py`'s `pow(M_l mod p_l, −1, p_l)`, which
  `cert_crt_roundtrip` checks.
