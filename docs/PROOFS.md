# Proofs of the parametric lemmas (Lean blueprint)

DESIGN.md's correctness rests on four lemmas that must hold **for every k, every N and
every input**. The other checks only cover finitely many cases:
- TLA+ checks them exhaustively at toy sizes;
- `params.py` checks real-width instances;
- the oracle tests random and adversarial inputs.

The lemmas are proved by hand here. Each statement uses only integers, rationals (dyadic
values `m·2^e`) and finite sums, with no IEEE library, so it transcribes directly into Lean 4 +
Mathlib. The Mathlib pieces needed are standard: Chinese remainder / `Int.ModEq`,
Cauchy–Schwarz over `Finset` sums, `Int.ceil` / `Int.floor`, and `Nat.sqrt`. Exact lemma
names will be pinned in `proofs/` once the Lean toolchain is set up.

Notation:
- `‖v‖ = ‖v‖₂` and `‖v‖₁` are norms over a finite index set of size k.
- `RN` is round-to-nearest-even to an integer. `TRUNC` truncates toward zero.
- `ρ = 1/2` for RN and `ρ = 1` for TRUNC, so `|q(x) − x| ≤ ρ` for either quantizer q.
  (Strictly `< 1` for TRUNC; `≤` is enough here.)

---

## L1. Shift rule ⇒ norm budget (DESIGN §3.2–§3.3)

**Setting.**
- `a ∈ ℚ^k` is a row, not all zero. γ is the integer with `2^γ ≤ max_h |a_h| < 2^{γ+1}`.
- `x_h = a_h·2^{−γ}`, so `1 ≤ max|x_h| < 2` and `‖x‖ ≥ 1`.
- `F ∈ ℕ`, and `S = Σ_h ⌈x_h²·2^F⌉ ∈ ℕ`.
- `T ∈ ℕ`, and `κ = ⌈√k⌉ ∈ ℕ`.
- RN: `R = (2T − κ)²`, `c = 4`, and we assume `2T − κ ≥ 1`. TRUNC: `R = T²`, `c = 1`, and
  we assume `T ≥ 1`.
- Only `2^γ ≤ max_h |a_h|` is used, so all results stay sound if γ is underestimated.
- `u` is the largest integer with `c·S·4^u ≤ R·2^F`, and `σ = u − γ`.
- `a'_h = q(a_h·2^σ)`.

**Claim L1a (u exists).** The set `{u : c·S·4^u ≤ R·2^F}` is non-empty and bounded above.

*Proof.* It is bounded above because `S ≥ 2^F > 0`, so `c·S·4^u → ∞`. It is non-empty
because `R ≥ 1`, so the inequality holds for u small enough. ∎

(At `2T − κ = 0`, R = 0 and no u exists. This is the soundness boundary, checked by
`spec: OzakiScaling neg_kappa_2T`.)

**Claim L1b (budget).** `‖a'‖ ≤ T`.

*Proof.*
1. `‖a·2^σ‖² = 4^u·‖x‖² ≤ 4^u·S·2^{−F} ≤ R/c`, using `S ≥ ‖x‖²2^F`.
2. TRUNC: `|a'_h| ≤ |a_h 2^σ|`, so `‖a'‖ ≤ ‖a 2^σ‖ ≤ √R = T`.
3. RN: `‖a 2^σ‖ ≤ (2T − κ)/2 = T − κ/2`. Write `a' = a2^σ + δ` with `|δ_h| ≤ 1/2`. Then
   `‖δ‖ ≤ √k/2 ≤ κ/2`, and the triangle inequality gives `‖a'‖ ≤ T`. ∎

**Corollary L1c (CRT precondition).** For a column b treated the same way,
`Σ_h |a'_h||b'_h| ≤ ‖a'‖‖b'‖ ≤ T² ≤ H`, by Cauchy–Schwarz and `T = ⌊√H⌋`.

**Claim L1d (σ bound, used by L3).** With `s = √(1 + k·2^{−F})`:
- RN: `2^{−σ} < 2‖a‖·s/(T − κ/2)`;
- TRUNC: `2^{−σ} < 2‖a‖·s/T`.

*Proof.*
1. Maximality of u means `c·S·4^{u+1} > R·2^F`.
2. Each ceiling adds less than 1, so `S < ‖x‖²2^F + k ≤ ‖x‖²2^F·s²`, using `‖x‖ ≥ 1`.
3. Hence `4^u > R/(4c·‖x‖²s²)`, i.e. `2^u > √R/(2√c·‖x‖·s)`.
4. Then `2^{−σ} = 2^γ·2^{−u} < 2√c·‖x‖2^γ·s/√R = 2√c·‖a‖·s/√R`.
5. RN (c = 4, `√R = 2T − κ`) gives `2‖a‖s/(T − κ/2)`. TRUNC gives `2‖a‖s/T`. ∎

The factor `s` is necessary: without it the inequality is false, as shown by
`OzakiScaling neg_sigmabound_literal` and by counterexamples at N = 14 and N = 20 in
`params.py`.

---

## L2. Division-free running CRT (DESIGN §3.6)

**Setting.**
- `p_1..p_N` are pairwise coprime, `P = ∏ p_l`, and P is even.
- `H = (P − 2)/2`, `M_l = P/p_l`, and `q_l = M_l⁻¹ mod p_l`.
- `W_l ∈ [0, p_l)`.
- The algorithm:
  - `x_0 = 0`;
  - `y_l = (q_l·W_l) mod p_l`;
  - `x'_l = x_{l−1} + M_l·y_l`;
  - `x_l = x'_l − P·[x'_l ≥ P]`;
  - `z = (x_N ≤ H) ? x_N : x_N − P`.

**Claim L2a (bounds).** `0 ≤ x_l < P` and `x'_l < 2P` for all l. Hence an unsigned word
of `L_w ≥ ⌈log2(2P)⌉` bits never overflows.

*Proof.* By induction. `M_l·y_l ≤ M_l(p_l − 1) < P`, so `x'_l < 2P`, and one conditional
subtraction restores `x_l < P`. ∎

**Claim L2b (congruence).** `x_N ≡ W_j (mod p_j)` for every j.

*Proof.* `x_N ≡ Σ_l M_l y_l (mod P)`, because each subtraction removes a multiple of P.
Modulo p_j: `M_l ≡ 0` for l ≠ j, and `M_j y_j ≡ M_j q_j W_j ≡ W_j`. ∎

P need not be even: L2c holds for any H with `2H + 1 ≤ P`, including `H = ⌊(P−1)/2⌋`.

**Claim L2c (exact recovery).** If `Z ∈ ℤ`, `Z ≡ W_j (mod p_j)` for all j, and `|Z| ≤ H`,
then `z = Z`.

*Proof.*
1. By L2b and pairwise coprimality (Chinese remainder theorem), `z ≡ x_N ≡ Z (mod P)`.
2. `z ∈ [H+1−P, H]`, a window of exactly P consecutive integers that contains `[−H, H]`.
3. Two integers in such a window that are congruent mod P are equal. ∎

Combining L1c and L2c with `Z = Σ a'b'` gives that the emulated product is exact.

---

## L3. Error contract (DESIGN §3.8)

**Setting.**
- a and b are nonzero. For a zero row, σ := 0 gives `a' = 0` and `z = 0`, so
  `|D − Σab| ≤ u|D|`.
- a and b are finite, with quantized forms `a'_h = a_h2^σ + δ_h` and `b'_h = b_h2^τ + ε_h`,
  where `|δ_h|, |ε_h| ≤ ρ`.
- `z = Σ_h a'_h b'_h`, which is exact by L1 + L2 + L4.
- `D = RN_double(z·2^{−σ−τ})`, with no overflow or underflow.

**Claim L3a (ℓ1 form).**
`|z·2^{−σ−τ} − Σ a_h b_h| ≤ ρ(2^{−τ}‖a‖₁ + 2^{−σ}‖b‖₁) + k·ρ²·2^{−σ−τ}`.

*Proof.* Expand: `a'b'2^{−σ−τ} − ab = a_h ε_h 2^{−τ} + b_h δ_h 2^{−σ} + δ_h ε_h 2^{−σ−τ}`.
Then take absolute values and sum. ∎

**Claim L3b (normwise form).**
- RN: `|D − Σab| ≤ 2√k‖a‖‖b‖·s/(T − κ/2) + k·2^{−σ−τ−2} + u|D|`.
- TRUNC: `|D − Σab| ≤ 4√k‖a‖‖b‖·s/T + k·2^{−σ−τ} + u|D|`.

*Proof.*
1. Use `‖a‖₁ ≤ √k‖a‖` and L1d for `2^{−σ}` and `2^{−τ}` in L3a.
2. RN: `ρ·2·(√k‖a‖·2‖b‖s/(T − κ/2)) = 2√k‖a‖‖b‖s/(T − κ/2)`. TRUNC: `4√k‖a‖‖b‖s/T`.
3. The final rounding adds `|D − z2^{−σ−τ}| ≤ u|D|`, because RN to binary64 in the normal
   range satisfies `|fl(x) − x| ≤ u·|fl(x)|`. ∎

**Claim L3c (accuracy-aware planning, DESIGN §6f.2).**

**Notation.**
- δ = ½ (RN) or 1 (TRUNC), and d = T − κ/2 (RN) or T (TRUNC). L1d gives 2^{−σ} < 2‖a‖s/d.
- y = z·2^{−σ−τ}.
- E = 2δ(ν_a + ν_b)s/d + 4δ²k s²/d².
- 0 < ν_a, ν_b, with ‖a_i‖₁ ≤ ν_a‖a_i‖₂ for **every** row i and ‖b_j‖₁ ≤ ν_b‖b_j‖₂ for every
  column j.

Then for every (i, j), with x = Σ_h a_h b_h:

1. |y − x| ≤ E‖a_i‖‖b_j‖.
2. With |D − y| ≤ u|y|:
   - NORMWISE: |D − x| ≤ (E(1+u) + u)‖a_i‖‖b_j‖;
   - ELEMENTWISE, if ‖a_i‖‖b_j‖ ≤ ρ|x|: |D − x| ≤ (Eρ(1+u) + u)|x|;
   - ABSOLUTE, if ‖a_i‖‖b_j‖ ≤ M̄: |D − x| ≤ (E(1+u) + u)M̄.

   If D may be subnormal, the hypothesis becomes |D − y| ≤ u|y| + 2⁻¹⁰⁷⁵, and the ABSOLUTE
   form gains + 2⁻¹⁰⁷⁵.
3. DIRECT (`contract.cu`'s native path: acc = +0, acc = fma(a_h, b_h, acc), k steps) without
   underflow: |D − x| ≤ γ_k Σ_h|a_h b_h| ≤ γ_k‖a‖‖b‖, with γ_k = ku/(1 − ku) and ku < 1. With
   underflow, each step adds at most 2⁻¹⁰⁷⁵, and the bound gains k·2⁻¹⁰⁷⁵·(1 + γ_k) ≤ 2k·2⁻¹⁰⁷⁵
   for ku ≤ ½.

*Proof.*
1. In L3a, ‖a‖₁ ≤ ν_a‖a‖ and ‖b‖₁ ≤ ν_b‖b‖ hold for the row and column at hand, because the
   hypothesis is per row and column, so the scalar maximum is valid. Substitute 2^{−σ} and
   2^{−τ} from L1d: the first term is ≤ δ·2s/d·(ν_a + ν_b)‖a‖‖b‖. The second is
   ≤ kδ²·4‖a‖‖b‖s²/d². With ν = √k this is L3b.
2. |D − x| ≤ |D − y| + |y − x| ≤ u|y| + |y − x| ≤ u|x| + (1+u)|y − x|.
   - NORMWISE: |x| ≤ ‖a‖‖b‖ (Cauchy–Schwarz).
   - ELEMENTWISE: divide by |x| and use ‖a‖‖b‖ ≤ ρ|x|.
   - ABSOLUTE: |x| ≤ ‖a‖‖b‖ ≤ M̄.
3. Standard recursive-summation analysis (Higham, Thm 3.1, with one rounding per fma): each
   partial sum carries one factor (1 + θ), |θ| ≤ u, so
   |fl − x| ≤ Σ_h |a_h b_h|·((1+u)^{k−h+1} − 1) ≤ γ_k Σ|a_h b_h|. Then Σ|a_h b_h| ≤ ‖a‖‖b‖ by
   Cauchy–Schwarz. An underflowing step adds an absolute η ≤ 2⁻¹⁰⁷⁵, which is propagated by at
   most (1 + γ_k). ∎

Checked by `cert_error_estimate` (params.py) on emulated and native elements, including
constant rows.

---

## L4. K-chunked accumulation with modular carry (DESIGN §3.5)

**Setting.**
- Integers `r_h, s_h` with `|r_h s_h| ≤ Q`.
- Chunks `C_0, …, C_{n−1}` partition `[0, k)`, with `n ≥ 1` and `|C_c| ≤ K`.
- `Q·K ≤ A_max`, where `A_max = 2³¹ − 1`.
- The accumulator saturates: any partial sum in any order is clamped to `[−A_max−1, A_max]`.
- Epilogue: `W_0 = acc_0 mod p` and `W_c = (W_{c−1} + acc_c) mod p`, with mod in `[0, p)`.

**Claim L4a (no saturation, any order).** For every chunk and every accumulation order,
each partial sum `Σ_{h∈S} r_h s_h` (for `S ⊆ C_c`) has absolute value `≤ Q·K ≤ A_max`.
So saturation never triggers, and `acc_c = Σ_{h∈C_c} r_h s_h` exactly. ∎

**Claim L4b (carry).** `W_{n−1} ≡ Σ_{h<k} r_h s_h (mod p)`, and `W_{n−1} ∈ [0, p)`.

*Proof.* By induction on c, using that the chunks partition `[0,k)`. ∎

**Claim L4c (int32 epilogue).** This is a separate numeric fact, not a consequence of
`Q·K ≤ A_max`; it needs `(p − 1) + Q·K ≤ 2³¹ − 1`. `W_{c−1} + acc_c ∈ [−Q·K, (p − 1) + Q·K] ⊆ int32`. For
`Q = 128²` and `K = 131056` the maximum is 2147221759, which is ≤ 2³¹ − 1. ∎

The hypothesis `Q·K ≤ A_max` must hold for **every** order (a bound on `Σ|products|`), not
just for the final sum. `spec: OzakiCRT neg_order` shows a data-sized chunk failing under
saturation.

---

## Formalization status

**Done.** All lemmas are machine-checked in `proofs/` (Lean 4.34.1, Mathlib v4.34.1), with no
`sorry`, `admit` or new axioms. The capstone `emulated_product_exact` chains L1c + L4 + L2c.
Run `tools/check_proofs.sh`. The lemma-to-theorem map and modelling choices are in
`proofs/NOTES.md`. Still taken as hypotheses:
- the IEEE rounding property;
- the residue map's int8 range;
- facts about the 20 concrete moduli, which `spec/params.py` certifies.

## Formalization plan (as written before the proofs)

| Lemma | Lean statement sketch | Main Mathlib ingredients |
|---|---|---|
| L1a/b/c | `∀ a : Fin k → ℚ, a ≠ 0 → ‖q(a·2^σ)‖² ≤ T²` (and the CRT corollary) | `Int.ceil`, `Nat.sqrt`, Cauchy–Schwarz on `Finset` sums, triangle inequality |
| L1d | `2^{-σ} < 2‖a‖s/(T−κ/2)` | real `Real.sqrt`, maximality of u (`Nat.find`/`Int` greatest) |
| L2 | running-sum bounds, congruence, exact recovery | `Int.ModEq`, Chinese remainder for pairwise coprime moduli |
| L3 | ℓ1 and normwise bounds, final rounding as a hypothesis `|fl x − x| ≤ u|fl x|` | `Finset.sum` algebra, `abs_sub` lemmas |
| L4 | chunk sums, carry by induction | `Finset.sum_partition`/`sum_biUnion`, `Int.emod` lemmas |

The floating-point rounding of the final result appears only as a stated hypothesis in L3.
Everything else is exact arithmetic over ℤ, ℚ or ℝ.
