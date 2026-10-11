# Proofs of the parametric lemmas (Lean blueprint)

DESIGN.md's correctness rests on four lemmas that must hold **for every k, every N and
every input**; L7 (v1.19) adds the bound of the K-segmentation caller recipe, L5 and L6 the
lemmas NOTES-C1 reserved for its K-weights and device certificate, and R1–R6 the integer facts
of the realization (DESIGN §2, §3.2, §3.4–§3.6). The other checks
only cover finitely many cases:
- TLA+ checks them exhaustively at toy sizes;
- `params.py` checks real-width instances;
- the oracle tests random and adversarial inputs.

The lemmas are proved by hand here. Each statement uses only integers, rationals (dyadic
values `m·2^e`) and finite sums, with no IEEE library, so it transcribes directly into Lean 4 +
Mathlib. The Mathlib pieces needed are standard: Chinese remainder / `Int.ModEq`,
Cauchy–Schwarz over `Finset` sums, `Int.ceil` / `Int.floor`, and `Nat.sqrt`. The Lean
theorem names are mapped in `proofs/NOTES.md`.

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
`Q = 128²` and the chunk length `K = 131008` used since DESIGN v1.13, the maximum is
2146435327, which is ≤ 2³¹ − 1. (With the earlier `K = 131056` it was 2147221759. The bound
is monotone in K, and Lean's `epi_input_int32` is stated for chunks of length ≤ 131056.) ∎

The hypothesis `Q·K ≤ A_max` must hold for **every** order (a bound on `Σ|products|`), not
just for the final sum. `spec: OzakiCRT neg_order` shows a data-sized chunk failing under
saturation.

---

## L5. K-weight cancellation and the guard (NOTES-C1 §4.4, stage 4)

NOTES-C1 §4.4 states L5 and gives the hand proof. Stage 4 was closed by its gate, so no API,
TLA+ `GuardSound` or `cert_kw_guard` exists. The lemma itself is proved here (issue #12, for
the paper track).

**Setting.**
- Rows a, columns b, and weights w ∈ ℤ^k.
- The weighted operands are `ã_h = a_h·2^{w_h}` and `b̃_h = b_h·2^{−w_h}`.
- σ, σ̃ are the fast shifts of a and ã at the same (N, k, rounding); τ, τ̃ are the same for
  the columns.
- `s² = 1 + k·2^{−F}`.

**Claim L5a.** `Σ_h ã_h b̃_h = Σ_h a_h b_h` exactly.

*Proof.* `2^{w_h}·2^{−w_h} = 1` term by term. ∎

**Claim L5b.** L1–L4 and L3c hold verbatim for (ã, b̃), using the weighted rows' norms and ℓ1
ratios.

*Proof.* They quantify over all rows. ∎

**Claim L5c (the guard).**
- `‖ã‖ < 2s·2^{σ−σ̃}·‖a‖`, and the same for the columns.
- For `G ≤ (σ̃ − σ) + (τ̃ − τ)`: `‖ã‖‖b̃‖ < 4s²·2^{−G}·‖a‖‖b‖`, for any integer G.
- If `G ≥ 3` and `s² ≤ 1 + 2^{−20}` (k ≤ 2⁴⁰ at F = 60): `‖ã‖‖b̃‖ < ‖a‖‖b‖`.

*Proof.*
1. L1b in norm form gives `‖ã‖ ≤ d·2^{−σ̃}`, since the rule bounds the scaled row before
   quantization.
2. L1d gives `d·2^{−σ} < 2s‖a‖`.
3. Divide, then multiply the two sides.
4. At G ≥ 3 the factor is `≤ 4(1 + 2^{−20})/8 < 1`. ∎

**Claim L5d (transfer to the request, exact half).** Write `g = 4s²·2^{−G}`, so
`‖ã‖‖b̃‖ ≤ g‖a‖‖b‖`. Then:
- ELEMENTWISE: `‖a‖‖b‖ ≤ cond·|x|` gives `‖ã‖‖b̃‖ ≤ max(1, g·cond)·|x|`.
- ABSOLUTE: `‖a‖‖b‖ ≤ M̄` gives `‖ã‖‖b̃‖ ≤ g·M̄`.
- NORMWISE: if the weighted bound β satisfies `β ≤ t′ = t/g`, then `|D − x| ≤ t‖a‖‖b‖`.

The FP evaluation of g (an exact ldexp, then one multiply by `1 ± 2^{−19}`) is not modelled;
NOTES-C1 §4.4 (d) argues it.

## L6. A-posteriori certificate (NOTES-C1 §4.4, stage 5, deferred)

NOTES-C1 §4.4 states L6 and gives the hand proof. Stage 5 is deferred (trigger T3, "e.g. for
the paper track"). Only the lemma is proved here; the device statistics, `CertSound` and the
certificates are not built.

**Setting.** As in L3, with:
- `‖a·2^σ‖, ‖b·2^τ‖ ≤ d` (L1b, before quantization);
- `√k ≤ κ`;
- `B̄ ≥ 2δκd + kδ²` and `X ≥ d²`.

**Claim L6a.** `|z − x·2^{σ+τ}| ≤ B̄`.

*Proof.* Apply L3a to the scaled rows (σ = τ = 0), then use `‖v‖₁ ≤ √k‖v‖ ≤ κd`. ∎

**Claim L6b.** If `|z| > B̄`, let `Y = |z| − B̄`. Then:
- `ρ = ‖a‖‖b‖/|x| ≤ X/Y`;
- with `|D − y| ≤ u|y|` (normal range), `|D − x| ≤ (u + (1+u)·B̄/Y)·|x|`.

*Proof.* `|x|·2^{σ+τ} ≥ Y` and `‖a‖‖b‖·2^{σ+τ} ≤ d² ≤ X`, so `|y − x| ≤ B̄·2^{−σ−τ} ≤ B̄|x|/Y`.
Then apply L3c step 2. ∎

**Claim L6c.** If `Y·2^r ≥ X`, then `ρ ≤ 2^r` and `B̄/Y ≤ B̄·2^r/X`. ∎

---

## L7. Segmented accumulation (DESIGN §3.8 "Segmented accumulation", the K-segmentation recipe)

**Setting.**
- K is partitioned into S ≥ 1 bands K_1 … K_S (any partition; the recipe uses contiguous
  bands of the canonical order). For a row a and a column b, `a^s` and `b^s` are their
  restrictions to K_s, `k_s = |K_s|`, `x_s = Σ_{h∈K_s} a_h b_h`, so `x = Σ_s x_s`.
- D_s is the emulated product of band s: one call at (N_s, k_s, rounding) with α = 1 and
  β = 0. Each band has its own σ_s, τ_s and N_s. A band of k_s = 1 is K-empty and runs
  DIRECT (one fma, `D_s = RN(a_h b_h)`, so `|D_s − x_s| ≤ u|x_s|`): it satisfies everything
  below with E_s = 0.
- `C_1 = D_1` and `C_s = fl(C_{s−1} + D_s)`: the calls after the first use β = 1 at α = 1, and
  §3.7's `fma(1, C, E)` is one rounding of the exact sum.
- `n_s = ‖a^s‖‖b^s‖`. E_s is L3c's E at (N_s, k_s, rounding) with the band's ν:
  ‖a^s‖₁ ≤ ν_a^s‖a^s‖ and ‖b^s‖₁ ≤ ν_b^s‖b^s‖ (ν = √k_s always qualifies).
- `γ_j = j·u/(1 − j·u)` for `j·u < 1`, and `γ = γ_{S−1}`.
- Hypotheses: every D_s satisfies L3c's final rounding `|D_s − y_s| ≤ u|y_s|` (normal range);
  every addition satisfies `|C_s − (C_{s−1} + D_s)| ≤ u|C_{s−1} + D_s|`. RN satisfies the
  latter absent overflow: an addition whose result underflows is exact.

**Claim L7a (recursive summation, γ_k).** Let `d_1 … d_S` be reals, `c_1 = d_1`, and
`|c_s − (c_{s−1} + d_s)| ≤ u|c_{s−1} + d_s|` for s ≥ 2. If `(S − 1)u < 1` then
`|c_S − Σ_s d_s| ≤ γ_{S−1}·Σ_s |d_s|`.

*Proof.* By induction on s, with `e_s = c_s − Σ_{t≤s} d_t` and `A_s = Σ_{t≤s} |d_t|`, show
`|e_s| ≤ γ_{s−1} A_s`. At s = 1, `e_1 = 0`. For the step, `c_{s+1} = c_s + d_{s+1} + θ`
with `|θ| ≤ u(|c_s| + |d_{s+1}|) ≤ u((1 + γ_{s−1})A_s + |d_{s+1}|)`, using
`|c_s| ≤ A_s + |e_s|`. So `|e_{s+1}| ≤ ((1 + γ_{s−1})(1 + u) − 1)A_s + u|d_{s+1}|`. Now
`(1 + γ_j)(1 + u) ≤ 1 + γ_{j+1}`, because `1 + γ_j = 1/(1 − ju)` and
`(1 + u)(1 − (j+1)u) = 1 − ju − (j+1)u² ≤ 1 − ju`; and `u ≤ γ_s` for s ≥ 1. Hence
`|e_{s+1}| ≤ γ_s A_{s+1}`. ∎ (Higham, Accuracy and Stability, §4.2, for recursive summation.)

**Claim L7b (segmented bound).** `|C_S − x| ≤ Σ_s n_s·W_s`, with
`W_s = u + (1 + u)E_s + γ(1 + u)(1 + E_s)`.

*Proof.* L3c (2) for band s: `|D_s − x_s| ≤ u|x_s| + (1 + u)|y_s − x_s| ≤ (u + (1 + u)E_s)n_s`,
using `|x_s| ≤ n_s` (Cauchy–Schwarz). Hence `|D_s| ≤ |x_s| + |D_s − x_s| ≤ (1 + u)(1 + E_s)n_s`.
L7a with `d_s = D_s` gives `|C_S − Σ D_s| ≤ γ Σ|D_s|`. The triangle inequality,
`|C_S − x| ≤ |C_S − Σ D_s| + Σ|D_s − x_s|`, finishes it. ∎

**Claim L7c (metric forms).** Let `W̄ = max_s W_s` and `N_S = Σ_s n_s`.
1. `N_S ≤ ‖a‖‖b‖`: Cauchy–Schwarz on the vectors `(‖a^s‖)_s`, `(‖b^s‖)_s`, with
   `Σ_s ‖a^s‖² = ‖a‖²`.
2. NORMWISE: `|C_S − x| ≤ W̄·‖a‖‖b‖`. ELEMENTWISE, if `N_S ≤ ρ_S|x|`:
   `|C_S − x| ≤ W̄ρ_S·|x|`, and ρ_S ≤ ρ. ABSOLUTE, if `N_S ≤ M̄`: `|C_S − x| ≤ W̄·M̄`.
3. With gradual underflow of some D_s (L3c's hypothesis `|D_s − y_s| ≤ u|y_s| + η`,
   η = 2⁻¹⁰⁷⁵), each band adds η to `|D_s − x_s|` and to `|D_s|`, so the bound gains
   `(1 + γ)·S·η`. Only the ABSOLUTE form keeps a meaning there.
4. Against L3c's unsegmented bound **with ν = √k** at the same N: `E_s ≤ E(k)`, since E grows
   with ν, s and k and falls with d, and `ν_s ≤ √k_s ≤ √k`, `s_s ≤ s`, `k_s ≤ k`,
   `κ_s ≤ κ` (so `d_s ≥ d`). So `W̄ ≤ E(1 + u) + u + γ(1 + u)(1 + E)`: the NORMWISE and
   ABSOLUTE bounds exceed L3c's by `γ(1 + u)(1 + E)·‖a‖‖b‖` only.
5. A whole-row ν < √k does **not** bound a band's ν_s. For `a = (1, ε, …, ε)` the whole-row
   ratio is near 1, while a band without the first entry has ratio √k_s. So compare only with
   the ν = √k bound, or measure ν per band.
6. ELEMENTWISE is **worse** than L3c's `u + Eρ(1 + u)` at high N: W̄ρ_S contains
   `(u + γ(1 + u))ρ_S` where L3c has u. Segmentation gives up the single final rounding.
7. ρ_S from the output. With `U_s = E_s(1 + u) + u` and `U ≥ max_s U_s`:
   `W̄ ≤ U + Su(1 + U)`, since `W_s = U_s + γ(1 + u)(1 + E_s)`, `γ(1 + u) ≤ Su` (L7d step 1,
   S ≤ 2²⁶) and `E_s ≤ U_s`. If `|C_S| > W̄·N_S` then
   `|x| ≥ |C_S| − |C_S − x| ≥ |C_S| − W̄·N_S > 0`, so `N_S/|x| ≤ N_S/(|C_S| − W̄·N_S)`, which
   increases with N_S and W̄: upper bounds of both, rounded outward as in DESIGN §6g.11, give
   a valid ρ_S. Without the normal-range hypothesis on the D_s (which a caller cannot check:
   D_s for s ≥ 2 is never materialized), (3) gives `|C_S − x| ≤ W̄·N_S + (1 + γ)·S·η`, and
   the same argument holds with the denominator `|C_S| − W̄·N_S − (1 + γ)·S·η`. Since γ ≤ 1
   for S ≤ 2²⁶, `(1 + γ)·S·η ≤ S·2⁻¹⁰⁷⁴`, a double (`params.kseg_eta`), which DESIGN §3.8's
   evaluation adds to U·n rounding upward. ∎

**Claim L7d (the recipe's request).** Let `t_eff > 0` be RN of a positive rational t̄, let
`S ≤ 2²⁶`, and let `t′ = RN(RN(min(t_eff, 1)·(1 − 2⁻⁵⁰)) − 4S·2⁻⁵³)` (`params.kseg_request`).
If `E(1 + u) + u ≤ t′` then `u + (1 + u)E + γ(1 + u)(1 + E) ≤ t̄`.

*Proof.*
1. `γ_{S−1}(1 + u) ≤ Su`: with j = S − 1 it is `j(1 + u) ≤ (j + 1)(1 − ju)`, i.e.
   `j(j + 2)u ≤ 1`, true for S ≤ 2²⁶.
2. `E ≤ E(1 + u) + u ≤ t′ ≤ 1`, so the left side is `≤ t′ + Su(1 + t′) ≤ t′ + 2Su`.
3. If `t_eff ≤ 1`: everything is in the normal range, since `t′ ≥ u` (step 2) forces
   `RN(min(t_eff, 1)(1 − 2⁻⁵⁰)) > 4Su ≥ 2⁻⁵¹`, so t_eff, t̄, the product and the difference
   are all normal. Hence `t_eff ≤ t̄(1 + u)`, the product rounds up by at most a factor
   `1 + u`, and `t′ > 0` rounds up by at most `1 + u` too, so
   `t′ ≤ t̄(1 − 2⁻⁵⁰)(1 + u)³ − 4Su < t̄(1 − 4u) − 4Su`, and `t′ + 2Su < t̄`.
4. If `t_eff > 1` then `t̄ > 1` (RN is monotone and 1 is a double), and
   `t′ ≤ (1 − 2⁻⁵⁰)(1 + u) − 4Su`, so `t′ + 2Su < 1 < t̄`. ∎

Checked by `cert_kseg_composition` (params.py) against exact rationals; `oracle.contract_segments`
is the recipe's bit-exact reference.

---

## R. Realization lemmas (DESIGN §2, §3.2, §3.4–§3.6)

These are the integer facts between L1–L4 and the device. `spec/params.py` certifies them at
the real parameters and the TLA+ models check them at toy widths. They are proved here for all
widths (issue #12).

**R1 (κ, DESIGN §3.2).** `c = isqrt(k)`, `κ = c + [c² < k]` equals `⌈√k⌉`.

**R2 (u by one compare, DESIGN §3.2).** Let `A = c·S` and `B = R·2^F` be positive integers, and
`ĉ = ⌊(bitlen B − bitlen A + 1)/2⌋` (an arithmetic shift).
- Then `A·4^{ĉ−1} ≤ B < A·4^{ĉ+1}`, so the largest u is `Rule(ĉ) ? ĉ : ĉ − 1`.
- *Proof.* `2^{a−1} ≤ A < 2^a`, `2^{b−1} ≤ B < 2^b`, and `b − a ≤ 2ĉ ≤ b − a + 1`. ∎

**R3 (S width, DESIGN §3.2).** With F = 60 and `max|x_h| < 2` (the upper half of the row
exponent), each term is `≤ 2⁶²`, so `S < 2¹²⁸` for `k < 2⁶⁶`.

**R4 (SMod, DESIGN §3.4).** `SMod(x, p) = r` if `r < ⌈p/2⌉`, else `r − p`, where
`r = x mod p ∈ [0, p)`.
- `SMod(x, p) ≡ x`, and it lies in `[−⌊p/2⌋, ⌈p/2⌉ − 1] ⊆ [−128, 127]` for p ≤ 256.
- So `−128·127 ≤ r·s ≤ 128²`.
- `SMod((μ mod p)(2^ε mod p), p) = SMod(μ·2^ε, p)`.
- For odd p, `SMod(−x, p) = −SMod(x, p)`; for p = 256 the two agree as bytes.
- The byte-weight sum `Σ b_i (2^{8i} mod p)` is `≡ X` and `≤ n·255²`.

**R5 (multiply-high reductions, DESIGN §3.4.1, §3.5, §3.6).** Let `m = ⌈D/p⌉` and
`e = mp − D < p`.
- If `s·e < D`, then `⌊s·m/D⌋ = ⌊s/p⌋`.
  - *Proof.* Write `s = pq + r`. Multiplying by p: `qD ≤ sm` and
    `p·sm = sD + se < sD + D ≤ p(q+1)D`. ∎
- Instances:
  - `mod_p_bytes32`: `D = 2³²`, `s < 2²⁴`, `p ≤ 256`.
  - The epilogue's `umulhi64`: `D = 2⁶⁴`, `n = v + p⌈2³¹/p⌉ < 2³²`, giving `v mod p`, because
    p divides the bias.
  - The CRT digit `(q·W) mod p`: `q, W < p ≤ 256`.

**R6 (the CRT word, DESIGN §2, §3.6).** Let `R_l = ∏_{j≥l} p_j`.
- **Peak bound.** `R_l | x_l`, so `x_l ≤ P − R_l` and `x'_l ≤ (P − R_l) + M_l(p_l − 1)`.
  - This is `crt_peak_exact`. It is attained: choose `W_j = (P − R_l) mod p_j` for j < l and
    `W_l = (p_l − 1)M_l mod p_l`.
- **No exact hit.** `x'_l = P` never occurs: it would force `p_l | M_l y_l`, hence `y_l = 0`
  and `x_l = P`. So `≥` and `>` agree.
- **Centring on the word.** On an unsigned `L_w`-bit word with `2P ≤ 2^{L_w}`, centring by a
  wrapped subtraction gives a word that, read as two's complement, is z. Its sign bit equals
  the borrow-out of `x − P`.
- **The design instance (the 20 moduli of DESIGN §2).** Pairwise coprime, P even,
  `T ≥ 180` for N ≥ 2.
  - L32 is the least limb count with `2P ≤ 2^{32·L32}`.
  - The exact peak is `≥ 2^{32(L32−1)}`, so every limb is needed.
  - The exact epilogue range at `K_c' = 131008` is
    `[−2129666048, (p − 1) + 2146435072] ⊆ int32`.

---

## Formalization status

**Done.** All lemmas are machine-checked in `proofs/` (Lean 4.34.1, Mathlib v4.34.1), with no
`sorry`, `admit` or new axioms. Run `tools/check_proofs.sh`. The lemma-to-theorem map and
modelling choices are in `proofs/NOTES.md`.
- **Capstones.**
  - `emulated_product_exact` chains L1c + L4 + L2c.
  - `Residue.emulated_product_exact_smod` feeds it the SMod residues the device stores.
  - `Moduli.emulated_product_exact_design` instantiates it at the 20 moduli, K_c' = 131008
    and int32 saturation.
- **[v1.19]** L3c (the ν form, its NORMWISE / ELEMENTWISE / ABSOLUTE compositions with the
  final rounding, and the native chain's γ_k) and L7 were added for NOTES-C1 stage 3 (IR5).
  - L7 includes L7a, `kseg_compose`, `kseg_rows`, L7c (1)–(4) and (7), and the exact half of
    L7d.
- **[issue #12]** Added:
  - the realization lemmas R1–R6;
  - L7c (2)'s ELEMENTWISE and ABSOLUTE forms and (7) without the normal-range hypothesis;
  - L5 (a, c, d's exact half);
  - L6 (a–c).

Still taken as hypotheses:
- the IEEE rounding properties (of the final conversion, `|fl(y) − y| ≤ u|fl(y)|` or
  `≤ u|y|`, and of an addition, `|fl(a + b) − (a + b)| ≤ u|a + b|`);
- the floating-point evaluation of the requests (L5d's `1 ± 2^{−19}`, L7d's `t′`, ρ̂_S), which
  `params.py` certifies;
- the choice of the multipliers `q_l` (any `M_l·q_l ≡ 1 (mod p_l)`; they exist, `exists_q`).

The residue map's int8 range and the facts about the 20 concrete moduli are no longer
hypotheses (R4, R6).

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
