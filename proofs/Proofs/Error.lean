import Proofs.Scaling

/-!
# L3. Error contract (PROOFS.md §L3, DESIGN.md §3.8)

Setting:
* rows/columns `a, b : Fin k → ℝ` and integer quantized forms `a', b' : Fin k → ℤ` with
  `a'_h = a_h·2^σ + δ_h`, `b'_h = b_h·2^τ + ε_h`, `|δ_h|, |ε_h| ≤ ρ`;
* `z = Σ_h a'_h b'_h` (exact, by L1 + L2 + L4);
* the final rounding is a **hypothesis**: `|D − z·2^{−σ−τ}| ≤ u·|D|` (binary64 RN in the
  normal range).

Results: L3a `l1_form`; L3b `normwise_rn`, `normwise_trunc` (from σ/τ bounds), the
generic `normwise_core`, the end-to-end `end_to_end_rn` / `end_to_end_trunc` (L1 + L3 from the
shift rule itself), and `zero_row` for the all-zero-row convention `σ := 0`.
-/

namespace Immerdink.Error

open Finset Real Immerdink.Scaling

variable {k : ℕ}

/-- One term of the expansion: with `A' = A·P + d`, `B' = B·Q + e` (`P, Q > 0`, `|d|, |e| ≤ ρ`),
`|A'B'/(PQ) − AB| ≤ ρ|A|/Q + ρ|B|/P + ρ²/(PQ)`. -/
private theorem term_bound {A B A' B' P Q ρ : ℝ} (hP : 0 < P) (hQ : 0 < Q)
    (hd : |A' - A * P| ≤ ρ) (he : |B' - B * Q| ≤ ρ) :
    |A' * B' * (P⁻¹ * Q⁻¹) - A * B| ≤ ρ * (Q⁻¹ * |A|) + ρ * (P⁻¹ * |B|) + ρ ^ 2 * (P⁻¹ * Q⁻¹) := by
  set d := A' - A * P
  set e := B' - B * Q
  have hA' : A' = A * P + d := by simp [d]
  have hB' : B' = B * Q + e := by simp [e]
  have expand : A' * B' * (P⁻¹ * Q⁻¹) - A * B
      = A * e * Q⁻¹ + B * d * P⁻¹ + d * e * (P⁻¹ * Q⁻¹) := by
    rw [hA', hB']; field_simp; ring
  have hP' : 0 < P⁻¹ := inv_pos.2 hP
  have hQ' : 0 < Q⁻¹ := inv_pos.2 hQ
  have hρ : 0 ≤ ρ := (abs_nonneg _).trans hd
  rw [expand]
  refine (abs_add_le _ _).trans ?_
  refine add_le_add ((abs_add_le _ _).trans (add_le_add ?_ ?_)) ?_
  · rw [abs_mul, abs_mul, abs_of_pos hQ']
    have := mul_le_mul_of_nonneg_left he (abs_nonneg A)
    nlinarith
  · rw [abs_mul, abs_mul, abs_of_pos hP']
    have := mul_le_mul_of_nonneg_left hd (abs_nonneg B)
    nlinarith
  · rw [abs_mul, abs_mul, abs_of_pos (mul_pos hP' hQ')]
    have : |d| * |e| ≤ ρ * ρ := mul_le_mul hd he (abs_nonneg _) hρ
    have h0 : 0 < P⁻¹ * Q⁻¹ := mul_pos hP' hQ'
    nlinarith

/-- **L3a (ℓ1 form).**
`|z·2^{−σ−τ} − Σ a_h b_h| ≤ ρ(2^{−τ}‖a‖₁ + 2^{−σ}‖b‖₁) + k·ρ²·2^{−σ−τ}`. -/
theorem l1_form {a b : Fin k → ℝ} {a' b' : Fin k → ℤ} {σ τ : ℤ} {ρ : ℝ}
    (hδ : ∀ h, |(a' h : ℝ) - a h * 2 ^ σ| ≤ ρ) (hε : ∀ h, |(b' h : ℝ) - b h * 2 ^ τ| ≤ ρ) :
    |((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ) - ∑ h, a h * b h|
      ≤ ρ * (2 ^ (-τ) * l1 a + 2 ^ (-σ) * l1 b) + k * ρ ^ 2 * 2 ^ (-σ - τ) := by
  have hP : (0 : ℝ) < 2 ^ σ := zpow_pos two_pos _
  have hQ : (0 : ℝ) < 2 ^ τ := zpow_pos two_pos _
  have hst : (2 : ℝ) ^ (-σ - τ) = (2 ^ σ)⁻¹ * (2 ^ τ)⁻¹ := by
    rw [sub_eq_add_neg, zpow_add₀ two_ne_zero, zpow_neg, zpow_neg]
  have e1 : ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ) - ∑ h, a h * b h
      = ∑ h, ((a' h : ℝ) * b' h * ((2 ^ σ)⁻¹ * (2 ^ τ)⁻¹) - a h * b h) := by
    push_cast; rw [sum_sub_distrib, sum_mul, hst]
  rw [e1]
  refine (abs_sum_le_sum_abs _ _).trans ?_
  refine (sum_le_sum fun h _ => term_bound hP hQ (hδ h) (hε h)).trans (le_of_eq ?_)
  simp only [sum_add_distrib, ← mul_sum, l1, sum_const, card_univ, Fintype.card_fin,
    nsmul_eq_mul, hst, zpow_neg]
  ring

/-- **L3b, generic form.** Given `‖a‖₁ ≤ √k‖a‖`, L3a and σ/τ bounds
`2^{−σ} ≤ 2‖a‖s/V`, `2^{−τ} ≤ 2‖b‖s/V` (`V > 0`), and the final rounding hypothesis:
`|D − Σab| ≤ ρ·4√k‖a‖‖b‖s/V + k·ρ²·2^{−σ−τ} + u|D|`. -/
theorem normwise_core {a b : Fin k → ℝ} {a' b' : Fin k → ℤ} {σ τ : ℤ} {ρ : ℝ}
    (hδ : ∀ h, |(a' h : ℝ) - a h * 2 ^ σ| ≤ ρ) (hε : ∀ h, |(b' h : ℝ) - b h * 2 ^ τ| ≤ ρ)
    {s V : ℝ} (hV : 0 < V)
    (hσb : (2 : ℝ) ^ (-σ) ≤ 2 * l2 a * s / V) (hτb : (2 : ℝ) ^ (-τ) ≤ 2 * l2 b * s / V)
    {D u : ℝ} (hD : |D - ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ u * |D|) :
    |D - ∑ h, a h * b h|
      ≤ ρ * (4 * √k * l2 a * l2 b * s / V) + k * ρ ^ 2 * 2 ^ (-σ - τ) + u * |D| := by
  have hρ : 0 ≤ ρ := by
    rcases isEmpty_or_nonempty (Fin k) with hk | ⟨⟨h0⟩⟩
    · -- k = 0 is impossible here: ‖a‖ = 0 would force 2^{−σ} ≤ 0 in `hσb`
      have ha : l2 a = 0 := by simp [l2]
      have hb : l2 b = 0 := by simp [l2]
      exfalso
      rw [ha] at hσb; simp at hσb
      linarith [zpow_pos (two_pos : (0 : ℝ) < 2) σ, zpow_pos (two_pos : (0 : ℝ) < 2) (-σ)]
    · exact (abs_nonneg _).trans (hδ h0)
  have L3a := l1_form hδ hε
  have ha1 := l1_le_sqrt_mul_l2 a
  have hb1 := l1_le_sqrt_mul_l2 b
  have hsq : 0 ≤ √(k : ℝ) := Real.sqrt_nonneg _
  have hla := l2_nonneg a
  have hlb := l2_nonneg b
  have hτ0 : (0 : ℝ) < 2 ^ (-τ) := zpow_pos two_pos _
  have hσ0 : (0 : ℝ) < 2 ^ (-σ) := zpow_pos two_pos _
  -- 2^{−τ}‖a‖₁ + 2^{−σ}‖b‖₁ ≤ 4√k‖a‖‖b‖s/V
  have key : 2 ^ (-τ) * l1 a + 2 ^ (-σ) * l1 b ≤ 4 * √k * l2 a * l2 b * s / V := by
    have t1 : 2 ^ (-τ) * l1 a ≤ (2 * l2 b * s / V) * (√k * l2 a) :=
      mul_le_mul hτb ha1 (sum_nonneg fun _ _ => abs_nonneg _) (hτ0.le.trans hτb)
    have t2 : 2 ^ (-σ) * l1 b ≤ (2 * l2 a * s / V) * (√k * l2 b) :=
      mul_le_mul hσb hb1 (sum_nonneg fun _ _ => abs_nonneg _) (hσ0.le.trans hσb)
    have : (2 * l2 b * s / V) * (√k * l2 a) + (2 * l2 a * s / V) * (√k * l2 b)
        = 4 * √k * l2 a * l2 b * s / V := by field_simp; ring
    linarith
  have := mul_le_mul_of_nonneg_left key hρ
  calc |D - ∑ h, a h * b h|
      ≤ |D - ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ)|
        + |((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ) - ∑ h, a h * b h| := abs_sub_le _ _ _
    _ ≤ ρ * (4 * √k * l2 a * l2 b * s / V) + k * ρ ^ 2 * 2 ^ (-σ - τ) + u * |D| := by
        linarith

/-- **L3b (RN).** `|D − Σab| ≤ 2√k‖a‖‖b‖s/(T − κ/2) + k·2^{−σ−τ−2} + u|D|`. -/
theorem normwise_rn {a b : Fin k → ℝ} {a' b' : Fin k → ℤ} {σ τ : ℤ}
    (hδ : ∀ h, |(a' h : ℝ) - a h * 2 ^ σ| ≤ 1 / 2) (hε : ∀ h, |(b' h : ℝ) - b h * 2 ^ τ| ≤ 1 / 2)
    {F T κ : ℕ} (hTκ : (κ : ℝ) < 2 * T)
    (hσb : (2 : ℝ) ^ (-σ) ≤ 2 * l2 a * sF k F / (T - κ / 2))
    (hτb : (2 : ℝ) ^ (-τ) ≤ 2 * l2 b * sF k F / (T - κ / 2))
    {D u : ℝ} (hD : |D - ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ u * |D|) :
    |D - ∑ h, a h * b h|
      ≤ 2 * √k * l2 a * l2 b * sF k F / (T - κ / 2) + k * 2 ^ (-σ - τ - 2) + u * |D| := by
  have := normwise_core hδ hε (by linarith) hσb hτb hD
  have e : (2 : ℝ) ^ (-σ - τ - 2) = 2 ^ (-σ - τ) / 4 := by
    rw [zpow_sub₀ two_ne_zero]; norm_num
  rw [e]
  have e2 : (1 / 2 : ℝ) * (4 * √k * l2 a * l2 b * sF k F / (T - κ / 2))
      = 2 * √k * l2 a * l2 b * sF k F / (T - κ / 2) := by ring
  linarith

/-- **L3b (TRUNC).** `|D − Σab| ≤ 4√k‖a‖‖b‖s/T + k·2^{−σ−τ} + u|D|`. -/
theorem normwise_trunc {a b : Fin k → ℝ} {a' b' : Fin k → ℤ} {σ τ : ℤ}
    (hδ : ∀ h, |(a' h : ℝ) - a h * 2 ^ σ| ≤ 1) (hε : ∀ h, |(b' h : ℝ) - b h * 2 ^ τ| ≤ 1)
    {F T : ℕ} (hT : 1 ≤ T)
    (hσb : (2 : ℝ) ^ (-σ) ≤ 2 * l2 a * sF k F / T) (hτb : (2 : ℝ) ^ (-τ) ≤ 2 * l2 b * sF k F / T)
    {D u : ℝ} (hD : |D - ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ u * |D|) :
    |D - ∑ h, a h * b h|
      ≤ 4 * √k * l2 a * l2 b * sF k F / T + k * 2 ^ (-σ - τ) + u * |D| := by
  have hT' : (0 : ℝ) < T := by exact_mod_cast hT
  have := normwise_core hδ hε hT' hσb hτb hD
  simpa using this

/-! ### End to end: L1 (shift rule) + L3 -/

/-- **L1 + L3 (RN).** Both operands scaled by the RN shift rule (largest `u`, `u'`), quantized
with any round-to-nearest `q`, product exact (`z = Σ a'b'`), final rounding as hypothesis. -/
theorem end_to_end_rn {a b : Fin k → ℝ} {γ γ' : ℤ} (hγ : RowExp a γ) (hγ' : RowExp b γ')
    {F T : ℕ} (hTκ : kappa k + 1 ≤ 2 * T) {u u' σ τ : ℤ}
    (hσ : σ = u - γ) (hτ : τ = u' - γ')
    (hmax : ∀ v, Rule 4 ((2 * T - kappa k : ℝ) ^ 2) (S a γ F) F v → v ≤ u)
    (hmax' : ∀ v, Rule 4 ((2 * T - kappa k : ℝ) ^ 2) (S b γ' F) F v → v ≤ u')
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1 / 2)
    {D eps : ℝ}
    (hD : |D - ((∑ h, q (a h * 2 ^ σ) * q (b h * 2 ^ τ) : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ eps * |D|) :
    |D - ∑ h, a h * b h|
      ≤ 2 * √k * l2 a * l2 b * sF k F / (T - kappa k / 2) + k * 2 ^ (-σ - τ - 2)
        + eps * |D| := by
  have : ((kappa k + 1 : ℕ) : ℝ) ≤ ((2 * T : ℕ) : ℝ) := by exact_mod_cast hTκ
  push_cast at this
  exact normwise_rn (fun _ => hq _) (fun _ => hq _) (F := F) (by linarith)
    (sigma_bound_rn hγ.1 (by linarith) hσ (not_rule_succ hmax)).le
    (sigma_bound_rn hγ'.1 (by linarith) hτ (not_rule_succ hmax')).le hD

/-- **L1 + L3 (TRUNC).** As `end_to_end_rn`, for truncation (`|q x| ≤ |x|`, `|q x − x| ≤ 1`). -/
theorem end_to_end_trunc {a b : Fin k → ℝ} {γ γ' : ℤ} (hγ : RowExp a γ) (hγ' : RowExp b γ')
    {F T : ℕ} (hT : 1 ≤ T) {u u' σ τ : ℤ} (hσ : σ = u - γ) (hτ : τ = u' - γ')
    (hmax : ∀ v, Rule 1 ((T : ℝ) ^ 2) (S a γ F) F v → v ≤ u)
    (hmax' : ∀ v, Rule 1 ((T : ℝ) ^ 2) (S b γ' F) F v → v ≤ u')
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1)
    {D eps : ℝ}
    (hD : |D - ((∑ h, q (a h * 2 ^ σ) * q (b h * 2 ^ τ) : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ eps * |D|) :
    |D - ∑ h, a h * b h| ≤ 4 * √k * l2 a * l2 b * sF k F / T + k * 2 ^ (-σ - τ) + eps * |D| :=
  normwise_trunc (fun _ => hq _) (fun _ => hq _) hT
    (sigma_bound_trunc hγ.1 hT hσ (not_rule_succ hmax)).le
    (sigma_bound_trunc hγ'.1 hT hτ (not_rule_succ hmax')).le hD

/-- **Zero rows.** L1d (hence L3b) needs `a ≠ 0`. For an all-zero row the design sets `σ := 0`;
then `a' = q(0) = 0` (any RN or TRUNC `q` maps 0 to 0), `z = 0`, and only the final
rounding term remains. -/
theorem zero_row {a b : Fin k → ℝ} (ha : ∀ h, a h = 0) {a' b' : Fin k → ℤ} (ha' : ∀ h, a' h = 0)
    {σ τ : ℤ} {D eps : ℝ}
    (hD : |D - ((∑ h, a' h * b' h : ℤ) : ℝ) * 2 ^ (-σ - τ)| ≤ eps * |D|) :
    |D - ∑ h, a h * b h| ≤ eps * |D| := by
  simpa [ha, ha'] using hD

/-- Any round-to-nearest `q` (`|q x − x| ≤ 1/2`) maps 0 to 0, as does any `|q x| ≤ |x|`. -/
theorem q_zero_rn {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1 / 2) : q 0 = 0 := by
  have h := hq 0
  rw [sub_zero] at h
  have : |q 0| ≤ 0 := by
    have : (|q 0| : ℝ) < 1 := by linarith
    have : |q 0| < 1 := by exact_mod_cast this
    omega
  exact abs_nonpos_iff.1 this

end Immerdink.Error
