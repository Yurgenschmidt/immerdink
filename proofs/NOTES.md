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
     `K`, so `epi_input_int32` (chunks of length `≤ 131056`) covers it unchanged. The
     theorem was not re-instantiated at 131008.
   - It also holds for `K = K_c = 131071`.

   PROOFS states the concrete numbers, so its claim is correct as written.

## Remaining work / not formalized

- The IEEE side, in particular `fl` satisfying `|fl(x) − x| ≤ u|fl(x)|` in the normal range,
  is a hypothesis by design.
- DESIGN's residue map `SMod` and the int8 range `|r| ≤ 128` for `p ≤ 256` are not
  formalized. `Exact.emulated_product_exact` takes `r ≡ a'`, `s ≡ b' (mod p)` and
  `|r s| ≤ Q` as hypotheses.
- Instance facts about the 20 concrete moduli (pairwise coprimality, `2P ≤ 2^{L_w}`) are
  left to `spec/params.py`. `decide` on 20 moduli would work but adds nothing parametric.
