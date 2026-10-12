import Proofs.Error
import Proofs.KWeights

/-!
# L6. The a-posteriori certificate (PROOFS.md §L6, NOTES-C1 §4.4)

Setting (as L3): rows `a, b`, quantized `a'_h = a_h·2^σ + δ_h`, `b'_h = b_h·2^τ + ε_h` with
`|δ_h|, |ε_h| ≤ δ`; the scaled rows satisfy L1b's `‖a·2^σ‖, ‖b·2^τ‖ ≤ d`; `√k ≤ κ`;
`z = Σ a'b'` (exact by L1–L4) and `x = Σ ab`.

* **L6a** `apost_bound`: `|z − x·2^{σ+τ}| ≤ 2δκd + kδ²` (any `B̄` above it; DESIGN's
  `B̄ = ⌈2δκd + kδ²⌉`). Neither L1d nor `s` is needed.
* **L6b** `apost_rho`, `apost_elementwise`: if `|z| > B̄` and `Y = |z| − B̄`, then
  `ρ = ‖a‖‖b‖/|x| ≤ X/Y` for any `X ≥ d²`, and, when the final rounding satisfies
  `|D − y| ≤ u|y|` (normal range), `|D − x| ≤ (u + (1+u)·B̄/Y)·|x|`.
* **L6c** `apost_pow2`: with `Y·2^r ≥ X`, `ρ ≤ 2^r` and the elementwise error is
  `≤ (u + (1+u)·B̄·2^r/X)·|x|`.

These are the lemmas of NOTES-C1's deferred stage 5 (trigger T3, "e.g. for the paper track").
Only the lemmas are proved here; the device statistics, `CertSound` and the certificates are
not built.
-/

namespace Immerdink.Apost

open Finset Real Immerdink.Scaling Immerdink.Error Immerdink.KWeights

variable {k : ℕ}

/-- **L6a.** `|z − x·2^{σ+τ}| ≤ 2δκd + kδ²`. -/
theorem apost_bound {a b : Fin k → ℝ} {a' b' : Fin k → ℤ} {σ τ : ℤ} {δ d : ℝ} {κ : ℕ}
    (hδ0 : 0 ≤ δ) (hδ : ∀ h, |(a' h : ℝ) - a h * 2 ^ σ| ≤ δ)
    (hε : ∀ h, |(b' h : ℝ) - b h * 2 ^ τ| ≤ δ)
    (hda : l2 (fun h => a h * 2 ^ σ) ≤ d) (hdb : l2 (fun h => b h * 2 ^ τ) ≤ d)
    (hκ : √(k : ℝ) ≤ κ) :
    |((∑ h, a' h * b' h : ℤ) : ℝ) - (∑ h, a h * b h) * 2 ^ (σ + τ)|
      ≤ 2 * δ * κ * d + k * δ ^ 2 := by
  set A : Fin k → ℝ := fun h => a h * 2 ^ σ
  set B : Fin k → ℝ := fun h => b h * 2 ^ τ
  have h := l1_form (a := A) (b := B) (a' := a') (b' := b') (σ := 0) (τ := 0) (ρ := δ)
    (fun h => by simpa [A] using hδ h) (fun h => by simpa [B] using hε h)
  simp only [neg_zero, sub_zero, zpow_zero, mul_one, one_mul] at h
  have e : ∑ h, A h * B h = (∑ h, a h * b h) * 2 ^ (σ + τ) := by
    rw [sum_mul]; refine sum_congr rfl fun h _ => ?_
    simp only [A, B]; rw [zpow_add₀ two_ne_zero]; ring
  rw [e] at h
  have hA : l1 A ≤ κ * d := by
    have := l1_le_sqrt_mul_l2 A
    have h2 := mul_le_mul hκ hda (l2_nonneg A) (Nat.cast_nonneg κ)
    linarith
  have hB : l1 B ≤ κ * d := by
    have := l1_le_sqrt_mul_l2 B
    have h2 := mul_le_mul hκ hdb (l2_nonneg B) (Nat.cast_nonneg κ)
    linarith
  have := mul_le_mul_of_nonneg_left (add_le_add hA hB) hδ0
  linarith

/-- The norms and the exact product at the scale of `z`: `|x|·2^{σ+τ} ≥ |z| − B̄` and
`‖a‖‖b‖·2^{σ+τ} ≤ d²`. -/
theorem apost_scaled {a b : Fin k → ℝ} {σ τ : ℤ} {d : ℝ}
    (hda : l2 (fun h => a h * 2 ^ σ) ≤ d) (hdb : l2 (fun h => b h * 2 ^ τ) ≤ d)
    {z x Bbar : ℝ} (hB : |z - x * 2 ^ (σ + τ)| ≤ Bbar) :
    |z| - Bbar ≤ |x| * 2 ^ (σ + τ) ∧ l2 a * l2 b * 2 ^ (σ + τ) ≤ d ^ 2 := by
  have hσ : (0 : ℝ) < 2 ^ σ := zpow_pos two_pos _
  have hτ : (0 : ℝ) < 2 ^ τ := zpow_pos two_pos _
  constructor
  · have h1 : |x * 2 ^ (σ + τ)| = |x| * 2 ^ (σ + τ) := by
      rw [abs_mul, abs_of_pos (zpow_pos two_pos (σ + τ) : (0 : ℝ) < 2 ^ (σ + τ))]
    have := abs_sub_abs_le_abs_sub z (x * 2 ^ (σ + τ))
    linarith
  · rw [l2_mul_const a hσ.le] at hda
    rw [l2_mul_const b hτ.le] at hdb
    have e : l2 a * l2 b * 2 ^ (σ + τ) = (l2 a * 2 ^ σ) * (l2 b * 2 ^ τ) := by
      rw [zpow_add₀ two_ne_zero]; ring
    rw [e, sq]
    exact mul_le_mul hda hdb (mul_nonneg (l2_nonneg b) hτ.le)
      ((mul_nonneg (l2_nonneg a) hσ.le).trans hda)

/-- **L6b, ρ.** If `B̄ < |z|` and `Y = |z| − B̄`, then `ρ = ‖a‖‖b‖/|x| ≤ X/Y` for `X ≥ d²`. -/
theorem apost_rho {a b : Fin k → ℝ} {σ τ : ℤ} {d : ℝ}
    (hda : l2 (fun h => a h * 2 ^ σ) ≤ d) (hdb : l2 (fun h => b h * 2 ^ τ) ≤ d)
    {z x Bbar X : ℝ} (hB : |z - x * 2 ^ (σ + τ)| ≤ Bbar) (hz : Bbar < |z|) (hX : d ^ 2 ≤ X) :
    0 < |x| ∧ l2 a * l2 b / |x| ≤ X / (|z| - Bbar) := by
  obtain ⟨h1, h2⟩ := apost_scaled hda hdb hB
  have hp : (0 : ℝ) < 2 ^ (σ + τ) := zpow_pos two_pos _
  have hY : 0 < |z| - Bbar := by linarith
  have hx : 0 < |x| := by
    by_contra hc
    have : |x| = 0 := le_antisymm (not_lt.1 hc) (abs_nonneg _)
    rw [this, zero_mul] at h1; linarith
  refine ⟨hx, ?_⟩
  rw [div_le_div_iff₀ hx hY]
  have hn : 0 ≤ l2 a * l2 b := mul_nonneg (l2_nonneg a) (l2_nonneg b)
  calc l2 a * l2 b * (|z| - Bbar) ≤ l2 a * l2 b * (|x| * 2 ^ (σ + τ)) :=
        mul_le_mul_of_nonneg_left h1 hn
    _ = (l2 a * l2 b * 2 ^ (σ + τ)) * |x| := by ring
    _ ≤ X * |x| := mul_le_mul_of_nonneg_right (h2.trans hX) (abs_nonneg _)

/-- **L6b, the error.** Under `apost_rho`'s hypotheses and the final rounding
`|D − y| ≤ u|y|` with `y = z·2^{−σ−τ}` (normal range):
`|D − x| ≤ (u + (1 + u)·B̄/Y)·|x|`, `Y = |z| − B̄`. -/
theorem apost_elementwise {σ τ : ℤ} {z x Bbar D u : ℝ} (hu : 0 ≤ u)
    (hB : |z - x * 2 ^ (σ + τ)| ≤ Bbar) (hz : Bbar < |z|)
    (hD : |D - z * 2 ^ (-σ - τ)| ≤ u * |z * 2 ^ (-σ - τ)|) :
    |D - x| ≤ (u + (1 + u) * (Bbar / (|z| - Bbar))) * |x| := by
  have hp : (0 : ℝ) < 2 ^ (σ + τ) := zpow_pos two_pos _
  have hY : 0 < |z| - Bbar := by linarith
  have hB0 : 0 ≤ Bbar := (abs_nonneg _).trans hB
  have hinv : (2 : ℝ) ^ (-σ - τ) = (2 ^ (σ + τ))⁻¹ := by
    rw [← zpow_neg]; congr 1; ring
  -- |y − x| = |z − x·2^{σ+τ}|·2^{−σ−τ} ≤ B̄·2^{−σ−τ}
  have hyx : |z * 2 ^ (-σ - τ) - x| ≤ Bbar * 2 ^ (-σ - τ) := by
    have e : z * 2 ^ (-σ - τ) - x = (z - x * 2 ^ (σ + τ)) * 2 ^ (-σ - τ) := by
      rw [hinv]; field_simp
    rw [e, abs_mul, abs_of_pos (zpow_pos two_pos (-σ - τ) : (0 : ℝ) < 2 ^ (-σ - τ))]
    exact mul_le_mul_of_nonneg_right hB (zpow_pos two_pos _).le
  -- 2^{−σ−τ} ≤ |x|/Y, from Y ≤ |x|·2^{σ+τ}
  have hsc : |z| - Bbar ≤ |x| * 2 ^ (σ + τ) := by
    have := abs_sub_abs_le_abs_sub z (x * 2 ^ (σ + τ))
    rw [abs_mul, abs_of_pos hp] at this
    linarith
  have h2 : (2 : ℝ) ^ (-σ - τ) ≤ |x| / (|z| - Bbar) := by
    rw [hinv, le_div_iff₀ hY, inv_mul_le_iff₀ hp]; linarith
  have h3 : |z * 2 ^ (-σ - τ) - x| ≤ Bbar / (|z| - Bbar) * |x| := by
    refine hyx.trans ?_
    calc Bbar * 2 ^ (-σ - τ) ≤ Bbar * (|x| / (|z| - Bbar)) :=
          mul_le_mul_of_nonneg_left h2 hB0
      _ = Bbar / (|z| - Bbar) * |x| := by ring
  have := round_step (x := x) hu hD
  have h4 := mul_le_mul_of_nonneg_left h3 (by linarith : (0 : ℝ) ≤ 1 + u)
  nlinarith

/-- **L6c, the power-of-two bin.** If `X > 0`, `Y = |z| − B̄ > 0` and `Y·2^r ≥ X`, then
`X/Y ≤ 2^r` (so `ρ ≤ 2^r` by `apost_rho`) and `B̄/Y ≤ B̄·2^r/X` (so the elementwise error of
`apost_elementwise` is `≤ (u + (1 + u)·B̄·2^r/X)·|x|`). -/
theorem apost_pow2 {Y X Bbar : ℝ} {r : ℕ} (hY : 0 < Y) (hX : 0 < X) (hB : 0 ≤ Bbar)
    (hr : X ≤ Y * 2 ^ r) : X / Y ≤ 2 ^ r ∧ Bbar / Y ≤ Bbar * 2 ^ r / X := by
  constructor
  · rw [div_le_iff₀ hY]; linarith
  · rw [div_le_div_iff₀ hY hX]
    have := mul_le_mul_of_nonneg_left hr hB
    linarith

end Immerdink.Apost
