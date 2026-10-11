import Proofs.Error
import Proofs.Summation

/-!
# L7. Segmented accumulation (PROOFS.md §L7, DESIGN §3.8 "Segmented accumulation")

The K-segmentation recipe: K is partitioned into `S = m + 1` bands, band `s` is one call with
α = 1, β = 0 (product `D s`, exact sum `x s`), and the later calls use β = 1, so
`C 0 = D 0` and `C (s+1) = fl(C s + D (s+1))`. Per band, with `y s = z_s·2^{−σ_s−τ_s}`:
* L3c (1): `|y s − x s| ≤ E s · n s`, `n s = ‖a^s‖‖b^s‖`, E s the band's `unitE`
  (`Error.l3c_unit`);
* `|x s| ≤ n s` (Cauchy–Schwarz);
* the final rounding `|D s − y s| ≤ u|y s|` (normal range) or `≤ u|y s| + η` (subnormal);
* the additions `|C (s+1) − (C s + D (s+1))| ≤ u|C s + D (s+1)|`.

Bands are indexed by `ℕ` (`range (m + 1)`); `x = Σ_s x s` is the contraction's sum over the
partition. The band statistics (`E s`, `n s`) are whatever L3c gives the band: L7 holds for
any partition, any per-band N and any per-band ν.

Results:
* `band_bound`: `|D s − x s| ≤ (u + (1+u)E_s)·n s` and `|D s| ≤ (1+u)(1+E_s)·n s` (+ η);
* **L7b** `kseg_compose`: `|C m − x| ≤ Σ_s n_s·W_s`,
  `W_s = u + (1+u)E_s + γ_m(1+u)(1+E_s)` (and `kseg_compose_sub` with the `(1+γ)·S·η` term);
  `kseg_rows` chains it from the band rows through L3c (1) (`Error.l3c_unit`);
* **L7c** `bands_norm_le` (`N_S ≤ ‖a‖‖b‖`) and `kseg_normwise`: `|C m − x| ≤ W̄·‖a‖‖b‖`;
  (7) `kseg_wbar_le` (`W ≤ U + (m+1)u(1+U)`) and `kseg_rho_le` (ρ_S from the output);
* **L7d** `gam_mul_le` (`γ_m(1+u) ≤ (m+1)u` for `m(m+2)u ≤ 1`) and `kseg_request_core`: the
  exact half of the recipe's request (`E(1+u) + u ≤ t′`, `t′ + 2Su ≤ t̄` ⇒ `W ≤ t̄`); the
  rounding of `t′` itself is checked by `cert_kseg_composition`.
-/

namespace Immerdink.Segments

open Finset Immerdink.Scaling Immerdink.Error Immerdink.Summation

/-- L7's per-band coefficient `W = u + (1+u)E + γ_m(1+u)(1+E)` (`S = m + 1` bands). -/
noncomputable def W (u E : ℝ) (m : ℕ) : ℝ := u + (1 + u) * E + gam m u * (1 + u) * (1 + E)

/-- One band through L3c and its final rounding: `|D − x| ≤ (u + (1+u)E)n + η` and
`|D| ≤ (1+u)(1+E)n + η` (η = 0 in the normal range). -/
theorem band_bound {x y D u E n η : ℝ} (hu : 0 ≤ u) (hx : |x| ≤ n) (hyx : |y - x| ≤ E * n)
    (hD : |D - y| ≤ u * |y| + η) :
    |D - x| ≤ (u + (1 + u) * E) * n + η ∧ |D| ≤ (1 + u) * (1 + E) * n + η := by
  have hy : |y| ≤ |x| + |y - x| := by
    have := abs_sub_abs_le_abs_sub y x; linarith
  have h0 : |D - x| ≤ |D - y| + |y - x| := abs_sub_le _ _ _
  have h3 := mul_le_mul_of_nonneg_left hy hu
  have h4 := mul_le_mul_of_nonneg_left hx hu
  have h5 := mul_le_mul_of_nonneg_left hyx hu
  have hDx : |D - x| ≤ (u + (1 + u) * E) * n + η := by nlinarith
  refine ⟨hDx, ?_⟩
  have : |D| ≤ |x| + |D - x| := by
    have := abs_sub_abs_le_abs_sub D x; linarith
  nlinarith

/-- **L7b with gradual underflow** (`kseg_compose` is the case η = 0):
`|C m − Σ_s x s| ≤ Σ_s n_s·W_s + (1 + γ_m)(m + 1)η`. -/
theorem kseg_compose_sub {u η : ℝ} (hu : 0 ≤ u) {m : ℕ} (hm : m * u < 1)
    {x y D E n C : ℕ → ℝ}
    (hx : ∀ s ≤ m, |x s| ≤ n s) (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * n s)
    (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s| + η)
    (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|) :
    |C m - ∑ s ∈ range (m + 1), x s|
      ≤ ∑ s ∈ range (m + 1), n s * W u (E s) m + (1 + gam m u) * (m + 1) * η := by
  have hg := gam_nonneg hu hm
  have hb : ∀ s ∈ range (m + 1), |D s - x s| ≤ (u + (1 + u) * E s) * n s + η ∧
      |D s| ≤ (1 + u) * (1 + E s) * n s + η := fun s hs => by
    have hs' : s ≤ m := Nat.lt_succ_iff.1 (mem_range.1 hs)
    exact band_bound hu (hx s hs') (hyx s hs') (hD s hs')
  have hsum := recsum hu (d := D) h0 hm hstep
  have hdiff : |∑ s ∈ range (m + 1), D s - ∑ s ∈ range (m + 1), x s|
      ≤ ∑ s ∈ range (m + 1), ((u + (1 + u) * E s) * n s + η) := by
    rw [← sum_sub_distrib]
    exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun s hs => (hb s hs).1)
  have habs : ∑ s ∈ range (m + 1), |D s| ≤ ∑ s ∈ range (m + 1), ((1 + u) * (1 + E s) * n s + η) :=
    sum_le_sum fun s hs => (hb s hs).2
  have hgam := mul_le_mul_of_nonneg_left habs hg
  have htri : |C m - ∑ s ∈ range (m + 1), x s|
      ≤ |C m - ∑ s ∈ range (m + 1), D s|
        + |∑ s ∈ range (m + 1), D s - ∑ s ∈ range (m + 1), x s| := abs_sub_le _ _ _
  have e : ∑ s ∈ range (m + 1), ((u + (1 + u) * E s) * n s + η)
      + gam m u * ∑ s ∈ range (m + 1), ((1 + u) * (1 + E s) * n s + η)
      = ∑ s ∈ range (m + 1), n s * W u (E s) m + (1 + gam m u) * (m + 1) * η := by
    have hP : ∑ s ∈ range (m + 1), n s * W u (E s) m
        = ∑ s ∈ range (m + 1), (u + (1 + u) * E s) * n s
          + gam m u * ∑ s ∈ range (m + 1), (1 + u) * (1 + E s) * n s := by
      rw [mul_sum, ← sum_add_distrib]
      exact sum_congr rfl fun s _ => by unfold W; ring
    simp only [sum_add_distrib, sum_const, card_range, nsmul_eq_mul]
    push_cast
    rw [hP]; ring
  linarith

/-- **L7b (`kseg_compose`).** `|C_S − x| ≤ Σ_s n_s·W_s`, `W_s = u + (1+u)E_s + γ_{S−1}(1+u)(1+E_s)`,
with `S = m + 1` bands, every band's L3c bound and normal-range final rounding, and the
additions' rounding hypothesis. -/
theorem kseg_compose {u : ℝ} (hu : 0 ≤ u) {m : ℕ} (hm : m * u < 1) {x y D E n C : ℕ → ℝ}
    (hx : ∀ s ≤ m, |x s| ≤ n s) (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * n s)
    (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s|)
    (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|) :
    |C m - ∑ s ∈ range (m + 1), x s| ≤ ∑ s ∈ range (m + 1), n s * W u (E s) m := by
  have := kseg_compose_sub (η := 0) hu hm hx hyx (fun s hs => by simpa using hD s hs) h0 hstep
  simpa using this

/-- Cauchy–Schwarz for one band: `|Σ a_h b_h| ≤ ‖a‖‖b‖`. -/
theorem abs_dot_le {k : ℕ} (a b : Fin k → ℝ) : |∑ h, a h * b h| ≤ l2 a * l2 b := by
  have h1 := abs_sum_le_sum_abs (fun h => a h * b h) univ
  have h2 := Real.sum_mul_le_sqrt_mul_sqrt univ (fun h => |a h|) (fun h => |b h|)
  simp only [sq_abs, ← abs_mul] at h2
  unfold l2
  exact h1.trans h2

/-- **L7 from the band rows.** Band `s` has rows `a s, b s` of length `k_s`, quantized forms
`a' s, b' s` (errors ≤ δ), shifts with L1d's bounds (`2^{−σ_s} ≤ 2‖a^s‖s_s/d_s`), ℓ1 ratios
`ν`, and its product's final rounding; the bands are summed with β = 1. Then
`|C_S − Σ_s Σ_h a^s_h b^s_h| ≤ Σ_s ‖a^s‖‖b^s‖·W(E_s)`, with `E_s = unitE` of band s (L3c (1)
per band, then L7b). -/
theorem kseg_rows {u δ : ℝ} (hu : 0 ≤ u) (hδ0 : 0 ≤ δ) {m : ℕ} (hm : m * u < 1) {ks : ℕ → ℕ}
    {a b : (s : ℕ) → Fin (ks s) → ℝ} {a' b' : (s : ℕ) → Fin (ks s) → ℤ} {σ τ : ℕ → ℤ}
    {νa νb sf d D C : ℕ → ℝ}
    (hδ : ∀ s ≤ m, ∀ h, |(a' s h : ℝ) - a s h * 2 ^ σ s| ≤ δ)
    (hε : ∀ s ≤ m, ∀ h, |(b' s h : ℝ) - b s h * 2 ^ τ s| ≤ δ)
    (hd : ∀ s ≤ m, 0 < d s)
    (hνa : ∀ s ≤ m, l1 (a s) ≤ νa s * l2 (a s)) (hνb : ∀ s ≤ m, l1 (b s) ≤ νb s * l2 (b s))
    (hσb : ∀ s ≤ m, (2 : ℝ) ^ (-σ s) ≤ 2 * l2 (a s) * sf s / d s)
    (hτb : ∀ s ≤ m, (2 : ℝ) ^ (-τ s) ≤ 2 * l2 (b s) * sf s / d s)
    (hD : ∀ s ≤ m, |D s - ((∑ h, a' s h * b' s h : ℤ) : ℝ) * 2 ^ (-σ s - τ s)|
      ≤ u * |((∑ h, a' s h * b' s h : ℤ) : ℝ) * 2 ^ (-σ s - τ s)|)
    (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|) :
    |C m - ∑ s ∈ range (m + 1), ∑ h, a s h * b s h|
      ≤ ∑ s ∈ range (m + 1),
          (l2 (a s) * l2 (b s)) * W u (unitE δ (νa s) (νb s) (sf s) (d s) (ks s)) m :=
  kseg_compose hu hm (x := fun s => ∑ h, a s h * b s h)
    (y := fun s => ((∑ h, a' s h * b' s h : ℤ) : ℝ) * 2 ^ (-σ s - τ s))
    (E := fun s => unitE δ (νa s) (νb s) (sf s) (d s) (ks s))
    (n := fun s => l2 (a s) * l2 (b s))
    (fun s _ => abs_dot_le (a s) (b s))
    (fun s hs => l3c_unit hδ0 (hδ s hs) (hε s hs) (hd s hs) (hνa s hs) (hνb s hs) (hσb s hs)
      (hτb s hs))
    hD h0 hstep

/-- **L7c (1).** `N_S = Σ_s ‖a^s‖‖b^s‖ ≤ √(Σ_s ‖a^s‖²)·√(Σ_s ‖b^s‖²)`, i.e. `≤ ‖a‖‖b‖` when
the bands partition K (Cauchy–Schwarz over the bands). -/
theorem bands_norm_le (m : ℕ) (na nb : ℕ → ℝ) :
    ∑ s ∈ range (m + 1), na s * nb s
      ≤ √(∑ s ∈ range (m + 1), na s ^ 2) * √(∑ s ∈ range (m + 1), nb s ^ 2) :=
  Real.sum_mul_le_sqrt_mul_sqrt _ _ _

/-- **L7c (2), NORMWISE.** With `n s = na s · nb s` (band norms, `≥ 0`) and every `W_s ≤ W̄`
(`W̄ ≥ 0`):
`|C_S − x| ≤ W̄·√(Σ‖a^s‖²)·√(Σ‖b^s‖²)`, i.e. `≤ W̄·‖a‖‖b‖`. -/
theorem kseg_normwise {u Wbar : ℝ} (hu : 0 ≤ u) (hWbar : 0 ≤ Wbar) {m : ℕ} (hm : m * u < 1)
    {x y D E na nb C : ℕ → ℝ} (hna : ∀ s, 0 ≤ na s) (hnb : ∀ s, 0 ≤ nb s)
    (hx : ∀ s ≤ m, |x s| ≤ na s * nb s) (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * (na s * nb s))
    (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s|) (hW : ∀ s ≤ m, W u (E s) m ≤ Wbar)
    (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|) :
    |C m - ∑ s ∈ range (m + 1), x s|
      ≤ Wbar * (√(∑ s ∈ range (m + 1), na s ^ 2) * √(∑ s ∈ range (m + 1), nb s ^ 2)) := by
  have h1 := kseg_compose hu hm hx hyx hD h0 hstep
  have h2 : ∑ s ∈ range (m + 1), na s * nb s * W u (E s) m
      ≤ ∑ s ∈ range (m + 1), na s * nb s * Wbar :=
    sum_le_sum fun s hs => mul_le_mul_of_nonneg_left
      (hW s (Nat.lt_succ_iff.1 (mem_range.1 hs))) (mul_nonneg (hna s) (hnb s))
  have h3 := bands_norm_le m na nb
  rw [← sum_mul] at h2
  calc |C m - ∑ s ∈ range (m + 1), x s|
      ≤ ∑ s ∈ range (m + 1), na s * nb s * W u (E s) m := by
        simpa [mul_comm, mul_left_comm, mul_assoc] using h1
    _ ≤ (∑ s ∈ range (m + 1), na s * nb s) * Wbar := h2
    _ ≤ (√(∑ s ∈ range (m + 1), na s ^ 2) * √(∑ s ∈ range (m + 1), nb s ^ 2)) * Wbar :=
        mul_le_mul_of_nonneg_right h3 hWbar
    _ = Wbar * (√(∑ s ∈ range (m + 1), na s ^ 2) * √(∑ s ∈ range (m + 1), nb s ^ 2)) := by ring

/-- **L7d (1).** `γ_m(1 + u) ≤ (m + 1)u` when `m(m + 2)u ≤ 1` (`S = m + 1 ≤ 2^26` at binary64). -/
theorem gam_mul_le {u : ℝ} (hu : 0 ≤ u) {m : ℕ} (hmm : m * (m + 2) * u ≤ 1) (hm : m * u < 1) :
    gam m u * (1 + u) ≤ (m + 1) * u := by
  have hpos : 0 < 1 - m * u := by linarith
  rw [gam, div_mul_eq_mul_div, div_le_iff₀ hpos]
  nlinarith [mul_nonneg hu (sub_nonneg.2 hmm)]

/-- **L7d (2), the exact half of the recipe's request.** If `E ≥ 0`, `E(1 + u) + u ≤ t′ ≤ 1`
and `t′ + 2(m + 1)u ≤ t̄`, then `W(E) ≤ t̄`. (`params.kseg_request`'s rounding gives the last
two hypotheses; `cert_kseg_composition` checks it.) -/
theorem kseg_request_core {u E t' tbar : ℝ} (hu : 0 ≤ u) (hE : 0 ≤ E) {m : ℕ}
    (hmm : m * (m + 2) * u ≤ 1) (hm : m * u < 1) (hU : E * (1 + u) + u ≤ t') (ht1 : t' ≤ 1)
    (htt : t' + 2 * (m + 1) * u ≤ tbar) : W u E m ≤ tbar := by
  have hg := gam_mul_le hu hmm hm
  have hE1 : E ≤ 1 := by nlinarith
  have hg0 := gam_nonneg hu hm
  unfold W
  have : gam m u * (1 + u) * (1 + E) ≤ (m + 1) * u * 2 := by
    have h2 : 1 + E ≤ 2 := by linarith
    calc gam m u * (1 + u) * (1 + E) ≤ (m + 1) * u * (1 + E) :=
          mul_le_mul_of_nonneg_right hg (by linarith)
      _ ≤ (m + 1) * u * 2 :=
          mul_le_mul_of_nonneg_left h2 (mul_nonneg (by positivity) hu)
  nlinarith

/-- **L7c (7), W̄ from the NORMWISE coefficient.** If `E ≥ 0` and `E(1 + u) + u ≤ U`, then
`W(E) ≤ U + (m + 1)u(1 + U)` (`params.kseg_wbar_up` evaluates the right side rounding
upward). -/
theorem kseg_wbar_le {u E U : ℝ} (hu : 0 ≤ u) (hE : 0 ≤ E) {m : ℕ}
    (hmm : m * (m + 2) * u ≤ 1) (hm : m * u < 1) (hU : E * (1 + u) + u ≤ U) :
    W u E m ≤ U + (m + 1) * u * (1 + U) := by
  have hg := gam_mul_le hu hmm hm
  have hEU : E ≤ U := by nlinarith [mul_nonneg hE hu]
  have h1 : gam m u * (1 + u) * (1 + E) ≤ (m + 1) * u * (1 + U) := by
    calc gam m u * (1 + u) * (1 + E) ≤ (m + 1) * u * (1 + E) :=
          mul_le_mul_of_nonneg_right hg (by linarith)
      _ ≤ (m + 1) * u * (1 + U) :=
          mul_le_mul_of_nonneg_left (by linarith) (mul_nonneg (by positivity) hu)
  unfold W
  nlinarith [h1, hU]

/-- **L7c (7), ρ_S from the output.** If `|C − x| ≤ W̄·N_S` (L7c (2)) and `W̄·N_S < |C|`,
then `N_S/|x| ≤ N_S/(|C| − W̄·N_S)`. -/
theorem kseg_rho_le {C x Wbar NS : ℝ} (hNS : 0 ≤ NS) (herr : |C - x| ≤ Wbar * NS)
    (hden : Wbar * NS < |C|) : NS / |x| ≤ NS / (|C| - Wbar * NS) := by
  have hx : |C| - Wbar * NS ≤ |x| := by
    have := abs_sub_abs_le_abs_sub C x; linarith
  exact div_le_div_of_nonneg_left hNS (by linarith) hx

/-- **L7c (7) without the normal-range hypothesis** (DESIGN §3.8, `params.kseg_eta`): with the
absolute term `η'` of `kseg_compose_sub` (`η' = (1 + γ)·S·η ≤ S·2⁻¹⁰⁷⁴`), if
`|C − x| ≤ W̄·N_S + η'` and `W̄·N_S + η' < |C|`, then `N_S/|x| ≤ N_S/(|C| − W̄·N_S − η')`. -/
theorem kseg_rho_le_sub {C x Wbar NS η' : ℝ} (hNS : 0 ≤ NS) (herr : |C - x| ≤ Wbar * NS + η')
    (hden : Wbar * NS + η' < |C|) : NS / |x| ≤ NS / (|C| - Wbar * NS - η') := by
  have hx : |C| - Wbar * NS - η' ≤ |x| := by
    have := abs_sub_abs_le_abs_sub C x; linarith
  exact div_le_div_of_nonneg_left hNS (by linarith) hx

/-- L7c (2), the common step: with every `W_s ≤ W̄`, `|C_S − x| ≤ W̄·N_S`, `N_S = Σ_s n_s`. -/
theorem kseg_wbar {u Wbar : ℝ} (hu : 0 ≤ u) {m : ℕ} (hm : m * u < 1) {x y D E n C : ℕ → ℝ}
    (hn : ∀ s, 0 ≤ n s) (hx : ∀ s ≤ m, |x s| ≤ n s) (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * n s)
    (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s|) (hW : ∀ s ≤ m, W u (E s) m ≤ Wbar)
    (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|) :
    |C m - ∑ s ∈ range (m + 1), x s| ≤ Wbar * ∑ s ∈ range (m + 1), n s := by
  refine (kseg_compose hu hm hx hyx hD h0 hstep).trans ?_
  rw [mul_sum]
  exact sum_le_sum fun s hs => by
    rw [mul_comm Wbar]
    exact mul_le_mul_of_nonneg_left (hW s (Nat.lt_succ_iff.1 (mem_range.1 hs))) (hn s)

/-- **L7c (2), ELEMENTWISE.** If `N_S ≤ ρ_S·|x|`: `|C_S − x| ≤ W̄·ρ_S·|x|`. -/
theorem kseg_elementwise {u Wbar ρ : ℝ} (hu : 0 ≤ u) (hWbar : 0 ≤ Wbar) {m : ℕ} (hm : m * u < 1)
    {x y D E n C : ℕ → ℝ} (hn : ∀ s, 0 ≤ n s) (hx : ∀ s ≤ m, |x s| ≤ n s)
    (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * n s) (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s|)
    (hW : ∀ s ≤ m, W u (E s) m ≤ Wbar) (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|)
    (hρ : ∑ s ∈ range (m + 1), n s ≤ ρ * |∑ s ∈ range (m + 1), x s|) :
    |C m - ∑ s ∈ range (m + 1), x s| ≤ Wbar * ρ * |∑ s ∈ range (m + 1), x s| := by
  have := kseg_wbar hu hm hn hx hyx hD hW h0 hstep
  have := mul_le_mul_of_nonneg_left hρ hWbar
  rw [← mul_assoc] at this
  linarith

/-- **L7c (2), ABSOLUTE.** If `N_S ≤ M̄`: `|C_S − x| ≤ W̄·M̄`. -/
theorem kseg_absolute {u Wbar Mbar : ℝ} (hu : 0 ≤ u) (hWbar : 0 ≤ Wbar) {m : ℕ}
    (hm : m * u < 1) {x y D E n C : ℕ → ℝ} (hn : ∀ s, 0 ≤ n s) (hx : ∀ s ≤ m, |x s| ≤ n s)
    (hyx : ∀ s ≤ m, |y s - x s| ≤ E s * n s) (hD : ∀ s ≤ m, |D s - y s| ≤ u * |y s|)
    (hW : ∀ s ≤ m, W u (E s) m ≤ Wbar) (h0 : C 0 = D 0)
    (hstep : ∀ i < m, |C (i + 1) - (C i + D (i + 1))| ≤ u * |C i + D (i + 1)|)
    (hM : ∑ s ∈ range (m + 1), n s ≤ Mbar) :
    |C m - ∑ s ∈ range (m + 1), x s| ≤ Wbar * Mbar :=
  (kseg_wbar hu hm hn hx hyx hD hW h0 hstep).trans (mul_le_mul_of_nonneg_left hM hWbar)

end Immerdink.Segments
