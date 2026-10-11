import Proofs.Scaling

/-!
# L5. K-weight cancellation and the guard (PROOFS.md §L5, NOTES-C1 §4.4)

Setting: rows `a`, columns `b`, and power-of-two K-weights `w ∈ ℤ^k`, with the weighted operands
`ã_h = a_h·2^{w_h}`, `b̃_h = b_h·2^{−w_h}`.

* **L5a** `kweight_cancel`: `Σ_h ã_h·b̃_h = Σ_h a_h·b_h` exactly.
* **L5b** needs no new proof: L1–L4 and L3c quantify over all rows, so they hold verbatim for
  the weighted operands (with the weighted rows' norms and ℓ1 ratios).
* **L5c** (the guard): with `σ, σ̃` the fast shifts of a row and of its weighted row at the
  same `(N, k, rounding)`, L1b gives `‖ã‖ ≤ d·2^{−σ̃}` (`norm_le_of_rule`) and L1d gives
  `‖a‖ > d·2^{−σ}/(2s)`, so `‖ã‖ < 2s·2^{σ−σ̃}·‖a‖` (`kweight_ratio`, `kweight_ratio_rn`,
  `kweight_ratio_trunc`). For a pair with `G ≤ (σ̃ − σ) + (τ̃ − τ)`:
  `‖ã‖‖b̃‖ < 4s²·2^{−G}·‖a‖‖b‖` for any integer `G` (`kweight_guard`), and `< ‖a‖‖b‖` when
  `G ≥ 3` and `s² ≤ 1 + 2^{−20}` (`kweight_guard_lt`; `k ≤ 2⁴⁰` at `F = 60`, `sF_sq_le`).
* **L5d** (the transfer to the request, exact half): ELEMENTWISE `cond′ = max(1, g·cond)`,
  ABSOLUTE `M̄′ = g·M̄`, NORMWISE `t′ = t/g`, with `g = 4s²·2^{−G}` (`kweight_transfer_*`).
  The floating-point evaluation of `g` (an exact ldexp and one multiply by `1 ± 2^{−19}`) is not
  modelled, as for L7d.

These are the lemmas NOTES-C1 reserves for stage 4, which its gate closed; the lemmas are
proved here for the paper track without reopening the stage (no API, TLA+ `GuardSound` or
`cert_kw_guard` is added).
-/

namespace Immerdink.KWeights

open Finset Real Immerdink.Scaling

variable {k : ℕ}

/-- **L5a.** Power-of-two K-weights cancel exactly in the dot product. -/
theorem kweight_cancel (a b : Fin k → ℝ) (w : Fin k → ℤ) :
    ∑ h, (a h * 2 ^ w h) * (b h * 2 ^ (-w h)) = ∑ h, a h * b h := by
  refine sum_congr rfl fun h _ => ?_
  have : (2 : ℝ) ^ w h * 2 ^ (-w h) = 1 := by
    rw [← zpow_add₀ two_ne_zero, add_neg_cancel, zpow_zero]
  calc (a h * 2 ^ w h) * (b h * 2 ^ (-w h)) = a h * b h * (2 ^ w h * 2 ^ (-w h)) := by ring
    _ = a h * b h := by rw [this, mul_one]

/-- `‖a·t‖₂ = ‖a‖₂·t` for `t ≥ 0`. -/
theorem l2_mul_const (a : Fin k → ℝ) {t : ℝ} (ht : 0 ≤ t) :
    l2 (fun h => a h * t) = l2 a * t := by
  unfold l2
  have e : ∑ h, (a h * t) ^ 2 = t ^ 2 * ∑ h, a h ^ 2 := by
    rw [mul_sum]; exact sum_congr rfl fun h _ => by ring
  rw [e, Real.sqrt_mul (sq_nonneg t), Real.sqrt_sq ht, mul_comm]

/-- **L1b, norm form.** If `u` satisfies the rule with `R = c·d²` (`c > 0`, `d ≥ 0`), then the
scaled row satisfies `‖a‖ ≤ d·2^{−σ}` (before quantization). -/
theorem norm_le_of_rule {a : Fin k → ℝ} {γ : ℤ} {F : ℕ} {c R d : ℝ} (hc : 0 < c) (hd : 0 ≤ d)
    (hR : R = c * d ^ 2) {u σ : ℤ} (hσ : σ = u - γ) (hu : Rule c R (S a γ F) F u) :
    l2 a ≤ d * 2 ^ (-σ) := by
  have h1 := rule_norm_sq hc.le hσ hu
  rw [hR] at h1
  have h2 : ∑ h, (a h * 2 ^ σ) ^ 2 ≤ d ^ 2 := le_of_mul_le_mul_left h1 hc
  have hpos : (0 : ℝ) < 2 ^ σ := zpow_pos two_pos _
  have e : ∑ h, (a h * 2 ^ σ) ^ 2 = (l2 a * 2 ^ σ) ^ 2 := by
    rw [← l2_mul_const a hpos.le, l2_sq]
  rw [e] at h2
  have h3 : l2 a * 2 ^ σ ≤ d := by
    have := mul_nonneg (l2_nonneg a) hpos.le
    nlinarith
  have e2 : l2 a = l2 a * 2 ^ σ * 2 ^ (-σ) := by
    rw [mul_assoc, ← zpow_add₀ two_ne_zero, add_neg_cancel, zpow_zero, mul_one]
  rw [e2]
  exact mul_le_mul_of_nonneg_right h3 (zpow_pos two_pos _).le

/-- **L5c, one side, abstract.** From `‖ã‖ ≤ d·2^{−σ̃}` (L1b for the weighted row) and
`2^{−σ} < 2‖a‖s/d` (L1d for the plain row): `‖ã‖ < 2s·2^{σ−σ̃}·‖a‖`. -/
theorem kweight_ratio {na nw d s : ℝ} {σ σw : ℤ} (hd : 0 < d)
    (h1 : nw ≤ d * 2 ^ (-σw)) (h2 : (2 : ℝ) ^ (-σ) < 2 * na * s / d) :
    nw < 2 * s * 2 ^ (σ - σw) * na := by
  have h2' : d * 2 ^ (-σ) < 2 * na * s := by
    rw [lt_div_iff₀ hd] at h2; linarith
  have e : d * (2 : ℝ) ^ (-σw) = d * 2 ^ (-σ) * 2 ^ (σ - σw) := by
    rw [mul_assoc, ← zpow_add₀ two_ne_zero]; congr 2; ring
  have hp : (0 : ℝ) < 2 ^ (σ - σw) := zpow_pos two_pos _
  calc nw ≤ d * 2 ^ (-σ) * 2 ^ (σ - σw) := e ▸ h1
    _ < 2 * na * s * 2 ^ (σ - σw) := mul_lt_mul_of_pos_right h2' hp
    _ = 2 * s * 2 ^ (σ - σw) * na := by ring

/-- **L5c, one side, RN.** Plain row `a` with its largest RN `u` (so `u + 1` fails) and weighted
row `ã` with any `ũ` satisfying the RN rule, at the same `T, κ` (`κ < 2T`):
`‖ã‖ < 2s·2^{σ−σ̃}·‖a‖`. -/
theorem kweight_ratio_rn {a aw : Fin k → ℝ} {γ γw : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|)
    {F T κ : ℕ} (hTκ : (κ : ℝ) < 2 * T) {u σ uw σw : ℤ} (hσ : σ = u - γ) (hσw : σw = uw - γw)
    (hmax : ¬ Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F (u + 1))
    (huw : Rule 4 ((2 * T - κ : ℝ) ^ 2) (S aw γw F) F uw) :
    l2 aw < 2 * sF k F * 2 ^ (σ - σw) * l2 a :=
  kweight_ratio (by linarith)
    (norm_le_of_rule (by norm_num) (by linarith) (by ring) hσw huw)
    (sigma_bound_rn hγ hTκ hσ hmax)

/-- **L5c, one side, TRUNC** (`T ≥ 1`). -/
theorem kweight_ratio_trunc {a aw : Fin k → ℝ} {γ γw : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|)
    {F T : ℕ} (hT : 1 ≤ T) {u σ uw σw : ℤ} (hσ : σ = u - γ) (hσw : σw = uw - γw)
    (hmax : ¬ Rule 1 ((T : ℝ) ^ 2) (S a γ F) F (u + 1))
    (huw : Rule 1 ((T : ℝ) ^ 2) (S aw γw F) F uw) :
    l2 aw < 2 * sF k F * 2 ^ (σ - σw) * l2 a :=
  kweight_ratio (by exact_mod_cast hT)
    (norm_le_of_rule one_pos (Nat.cast_nonneg _) (by ring) hσw huw)
    (sigma_bound_trunc hγ hT hσ hmax)

/-- **L5c, the pair.** If both sides satisfy `kweight_ratio`'s conclusion and
`G ≤ (σ̃ − σ) + (τ̃ − τ)`, then `‖ã‖‖b̃‖ < 4s²·2^{−G}·‖a‖‖b‖`, for any integer `G`. -/
theorem kweight_guard {na nb nwa nwb s : ℝ} {σ σw τ τw G : ℤ} (hs : 0 ≤ s) (hna : 0 ≤ na)
    (hnb : 0 ≤ nb) (hnwa : 0 ≤ nwa) (hnwb : 0 ≤ nwb)
    (ha : nwa < 2 * s * 2 ^ (σ - σw) * na) (hb : nwb < 2 * s * 2 ^ (τ - τw) * nb)
    (hG : G ≤ (σw - σ) + (τw - τ)) :
    nwa * nwb < 4 * s ^ 2 * 2 ^ (-G) * (na * nb) := by
  have h1 : nwa * nwb < (2 * s * 2 ^ (σ - σw) * na) * (2 * s * 2 ^ (τ - τw) * nb) :=
    mul_lt_mul'' ha hb hnwa hnwb
  have e : (2 * s * 2 ^ (σ - σw) * na) * (2 * s * 2 ^ (τ - τw) * nb)
      = 4 * s ^ 2 * 2 ^ ((σ - σw) + (τ - τw)) * (na * nb) := by
    rw [zpow_add₀ two_ne_zero]; ring
  have h2 : (2 : ℝ) ^ ((σ - σw) + (τ - τw)) ≤ 2 ^ (-G) :=
    zpow_le_zpow_right₀ (by norm_num) (by omega)
  have h3 : 0 ≤ 4 * s ^ 2 * (na * nb) := by positivity
  rw [e] at h1
  nlinarith

/-- `s² = 1 + k·2^{−60} ≤ 1 + 2^{−20}` for `k ≤ 2⁴⁰` (F = 60). -/
theorem sF_sq_le {k : ℕ} (hk : k ≤ 2 ^ 40) : sF k 60 ^ 2 ≤ 1 + 2 ^ (-20 : ℤ) := by
  unfold sF
  rw [Real.sq_sqrt (by positivity)]
  have hk' : (k : ℝ) ≤ 2 ^ 40 := by exact_mod_cast hk
  have : (k : ℝ) * 2 ^ (-((60 : ℕ) : ℤ)) ≤ 2 ^ 40 * 2 ^ (-((60 : ℕ) : ℤ)) :=
    mul_le_mul_of_nonneg_right hk' (zpow_pos two_pos _).le
  have e : (2 : ℝ) ^ 40 * 2 ^ (-((60 : ℕ) : ℤ)) = 2 ^ (-20 : ℤ) := by
    rw [show (2 : ℝ) ^ 40 = 2 ^ (40 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; norm_num
  linarith

/-- **L5c, the guard.** With `G ≥ 3` and `s² ≤ 1 + 2^{−20}`, the weighted norm product is
strictly below the plain one (when the plain one is positive): weighting never worsens the
§3.8 bound. -/
theorem kweight_guard_lt {na nb nwa nwb s : ℝ} {σ σw τ τw G : ℤ} (hs : 0 ≤ s)
    (hs2 : s ^ 2 ≤ 1 + 2 ^ (-20 : ℤ)) (hna : 0 ≤ na) (hnb : 0 ≤ nb) (hnwa : 0 ≤ nwa)
    (hnwb : 0 ≤ nwb) (ha : nwa < 2 * s * 2 ^ (σ - σw) * na) (hb : nwb < 2 * s * 2 ^ (τ - τw) * nb)
    (hG : G ≤ (σw - σ) + (τw - τ)) (hG3 : 3 ≤ G) : nwa * nwb < na * nb := by
  have h := kweight_guard hs hna hnb hnwa hnwb ha hb hG
  have hg : (2 : ℝ) ^ (-G) ≤ 2 ^ (-3 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have h20 : (2 : ℝ) ^ (-20 : ℤ) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)
  have h3 : (2 : ℝ) ^ (-3 : ℤ) = 1 / 8 := by norm_num
  have hcoef : 4 * s ^ 2 * 2 ^ (-G) ≤ 4 * (1 + 2 ^ (-20 : ℤ)) * (1 / 8) := by
    have := zpow_pos (two_pos : (0 : ℝ) < 2) (-G)
    rw [← h3]
    exact mul_le_mul (mul_le_mul_of_nonneg_left hs2 (by norm_num)) hg this.le (by positivity)
  have hnn : 0 ≤ na * nb := mul_nonneg hna hnb
  nlinarith

/-! ### L5d: the transfer to the request (exact half) -/

/-- **L5d, ELEMENTWISE.** If `‖ã‖‖b̃‖ ≤ g·‖a‖‖b‖` and the plain pair has `‖a‖‖b‖ ≤ cond·|x|`, then
the weighted pair has `‖ã‖‖b̃‖ ≤ max(1, g·cond)·|x|` (same `x`, by L5a). -/
theorem kweight_transfer_elementwise {n nw x g cond : ℝ} (hg : 0 ≤ g) (hnw : nw ≤ g * n)
    (hc : n ≤ cond * |x|) : nw ≤ max 1 (g * cond) * |x| := by
  have h1 : nw ≤ g * cond * |x| := by
    have := mul_le_mul_of_nonneg_left hc hg; linarith
  exact h1.trans (mul_le_mul_of_nonneg_right (le_max_right _ _) (abs_nonneg _))

/-- **L5d, ABSOLUTE.** `‖a‖‖b‖ ≤ M̄` gives `‖ã‖‖b̃‖ ≤ g·M̄`. -/
theorem kweight_transfer_absolute {n nw g Mbar : ℝ} (hg : 0 ≤ g) (hnw : nw ≤ g * n)
    (hM : n ≤ Mbar) : nw ≤ g * Mbar :=
  hnw.trans (mul_le_mul_of_nonneg_left hM hg)

/-- **L5d, NORMWISE relative to the plain norms.** If the weighted call's NORMWISE bound is
`|D − x| ≤ β·‖ã‖‖b̃‖` with `β ≤ t′ = t/g`, then `|D − x| ≤ t·‖a‖‖b‖`. -/
theorem kweight_transfer_normwise {n nw g β t D x : ℝ} (hg : 0 < g) (hβ : 0 ≤ β)
    (hnw : nw ≤ g * n) (hβt : β ≤ t / g) (herr : |D - x| ≤ β * nw) (hn : 0 ≤ n) :
    |D - x| ≤ t * n := by
  have h1 : β * nw ≤ β * (g * n) := mul_le_mul_of_nonneg_left hnw hβ
  have h2 : β * g ≤ t := by rw [le_div_iff₀ hg] at hβt; exact hβt
  have h3 : β * (g * n) ≤ t * n := by
    rw [← mul_assoc]; exact mul_le_mul_of_nonneg_right h2 hn
  linarith

end Immerdink.KWeights
