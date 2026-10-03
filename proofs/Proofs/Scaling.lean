import Mathlib

/-!
# L1. Shift rule ⇒ norm budget (PROOFS.md §L1, DESIGN.md §3.2–§3.3)

Setting (over `ℝ`, which contains the dyadic rationals the design works with):
* a row `a : Fin k → ℝ`, with a row exponent `γ` such that `2^γ ≤ max_h |a_h| < 2^{γ+1}`
  (`RowExp`); only the lower half (`∃ h, 2^γ ≤ |a_h|`) is ever used;
* `x_h = a_h·2^{−γ}`, `F ∈ ℕ`, `S = Σ_h ⌈x_h²·2^F⌉ ∈ ℤ` (`S`);
* `T, κ ∈ ℕ` with `√k ≤ κ` (true for `κ = ⌈√k⌉`, `sqrt_le_kappa`);
* the rule `c·S·4^u ≤ R·2^F` (`Rule`): RN uses `c = 4, R = (2T − κ)²`; TRUNC `c = 1, R = T²`;
* `σ = u − γ` and `a'_h = q(a_h·2^σ)` for a quantizer `q : ℝ → ℤ`.

Quantizers are abstract: RN is any `q` with `|q x − x| ≤ 1/2` (round-to-nearest with either
tie rule; `rne` below is ties-to-even), TRUNC any `q` with `|q x| ≤ |x|` (`truncZ`).

Results:
* L1a `exists_max_rule` (+ `exists_u_rn`, `exists_u_trunc`, `no_rule_of_R_zero`);
* L1b `budget_rn`, `budget_trunc` (and integer forms `budget_rn_int`, `budget_trunc_int`);
* L1c `crt_precondition`;
* L1d `sigma_bound_rn`, `sigma_bound_trunc`.
-/

namespace Immerdink.Scaling

open Finset Real

variable {k : ℕ}

/-! ### Norms -/

/-- `‖v‖₂`. -/
noncomputable def l2 (v : Fin k → ℝ) : ℝ := √(∑ h, v h ^ 2)

/-- `‖v‖₁`. -/
def l1 (v : Fin k → ℝ) : ℝ := ∑ h, |v h|

theorem l2_nonneg (v : Fin k → ℝ) : 0 ≤ l2 v := Real.sqrt_nonneg _

theorem sum_sq_nonneg (v : Fin k → ℝ) : 0 ≤ ∑ h, v h ^ 2 := sum_nonneg fun _ _ => sq_nonneg _

theorem l2_sq (v : Fin k → ℝ) : l2 v ^ 2 = ∑ h, v h ^ 2 := Real.sq_sqrt (sum_sq_nonneg v)

/-- Minkowski: `‖u + v‖₂ ≤ ‖u‖₂ + ‖v‖₂`. -/
theorem l2_add_le (u v : Fin k → ℝ) : l2 (fun h => u h + v h) ≤ l2 u + l2 v := by
  rw [l2, Real.sqrt_le_left (add_nonneg (l2_nonneg u) (l2_nonneg v))]
  have cs := Real.sum_mul_le_sqrt_mul_sqrt univ u v
  have e : ∑ h, (u h + v h) ^ 2 = ∑ h, u h ^ 2 + 2 * ∑ h, u h * v h + ∑ h, v h ^ 2 := by
    simp only [add_sq, sum_add_distrib, mul_sum]; congr 1; congr 1
    exact sum_congr rfl fun _ _ => by ring
  rw [e, add_sq, l2_sq, l2_sq]
  unfold l2
  linarith

/-- If every entry has `|v_h| ≤ r` then `‖v‖₂ ≤ √k·r`. -/
theorem l2_le_of_abs_le {v : Fin k → ℝ} {r : ℝ} (hr : 0 ≤ r) (hv : ∀ h, |v h| ≤ r) :
    l2 v ≤ √k * r := by
  rw [l2, Real.sqrt_le_left (mul_nonneg (Real.sqrt_nonneg _) hr), mul_pow,
    Real.sq_sqrt (Nat.cast_nonneg _)]
  calc ∑ h, v h ^ 2 ≤ ∑ _h : Fin k, r ^ 2 :=
        sum_le_sum fun h _ => by
          rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) (hv h) 2
    _ = k * r ^ 2 := by simp

/-- `‖v‖₁ ≤ √k·‖v‖₂` (Cauchy–Schwarz against the all-ones vector). -/
theorem l1_le_sqrt_mul_l2 (v : Fin k → ℝ) : l1 v ≤ √k * l2 v := by
  have cs := Real.sum_mul_le_sqrt_mul_sqrt univ (fun _ => (1 : ℝ)) (fun h => |v h|)
  simp only [one_mul, one_pow, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one, sq_abs] at cs
  exact cs

/-! ### The setting -/

/-- `S = Σ_h ⌈x_h²·2^F⌉` with `x_h = a_h·2^{−γ}`. -/
noncomputable def S (a : Fin k → ℝ) (γ : ℤ) (F : ℕ) : ℤ :=
  ∑ h, ⌈(a h * 2 ^ (-γ)) ^ 2 * 2 ^ F⌉

/-- The shift rule `c·S·4^u ≤ R·2^F`. -/
def Rule (c R : ℝ) (S : ℤ) (F : ℕ) (u : ℤ) : Prop := c * S * (4 : ℝ) ^ u ≤ R * 2 ^ F

/-- `γ` is the row exponent of `a`: `2^γ ≤ max_h |a_h| < 2^{γ+1}`. -/
def RowExp (a : Fin k → ℝ) (γ : ℤ) : Prop :=
  (∃ h, (2 : ℝ) ^ γ ≤ |a h|) ∧ ∀ h, |a h| < (2 : ℝ) ^ (γ + 1)

/-- `s = √(1 + k·2^{−F})`. -/
noncomputable def sF (k F : ℕ) : ℝ := √(1 + k * (2 : ℝ) ^ (-(F : ℤ)))

/-- `κ = ⌈√k⌉`. -/
noncomputable def kappa (k : ℕ) : ℕ := ⌈√(k : ℝ)⌉₊

theorem sqrt_le_kappa (k : ℕ) : √(k : ℝ) ≤ kappa k := Nat.le_ceil _

/-! ### Basic facts about `x`, `S` and the shift -/

section facts
variable (a : Fin k → ℝ) (γ : ℤ) (F : ℕ)

/-- `X = Σ_h x_h² = ‖x‖₂²`. -/
private noncomputable def X : ℝ := ∑ h, (a h * 2 ^ (-γ)) ^ 2

private theorem X_nonneg : 0 ≤ X a γ := sum_nonneg fun _ _ => sq_nonneg _

private theorem S_cast : (S a γ F : ℝ) = ∑ h, (⌈(a h * 2 ^ (-γ)) ^ 2 * 2 ^ F⌉ : ℝ) := by
  simp [S]

/-- `‖x‖²·2^F ≤ S`. -/
private theorem X_mul_le_S : X a γ * 2 ^ F ≤ S a γ F := by
  rw [S_cast, X, sum_mul]
  exact sum_le_sum fun h _ => Int.le_ceil _

/-- `S < ‖x‖²·2^F + k` (each ceiling adds less than 1; needs `k ≥ 1`). -/
private theorem S_lt (hk : (univ : Finset (Fin k)).Nonempty) :
    (S a γ F : ℝ) < X a γ * 2 ^ F + k := by
  rw [S_cast, X, sum_mul]
  have : ∑ h, ((a h * 2 ^ (-γ)) ^ 2 * 2 ^ F + 1 : ℝ)
      = ∑ h, (a h * 2 ^ (-γ)) ^ 2 * 2 ^ F + (k : ℝ) := by
    rw [sum_add_distrib]; simp
  rw [← this]
  exact sum_lt_sum_of_nonempty hk fun h _ => Int.ceil_lt_add_one _

variable {a γ}

/-- `‖x‖ ≥ 1`, i.e. `X ≥ 1`, from `2^γ ≤ |a_{h₀}|`. -/
private theorem one_le_X (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) : 1 ≤ X a γ := by
  obtain ⟨h0, h0γ⟩ := hγ
  have hx : 1 ≤ |a h0 * 2 ^ (-γ)| := by
    rw [abs_mul, abs_of_pos (zpow_pos two_pos (-γ) : (0 : ℝ) < 2 ^ (-γ)), zpow_neg, ← div_eq_mul_inv,
      le_div_iff₀ (zpow_pos two_pos _), one_mul]
    exact h0γ
  have : 1 ≤ (a h0 * 2 ^ (-γ)) ^ 2 := by
    rw [← sq_abs]; nlinarith
  exact this.trans (single_le_sum (f := fun h => (a h * 2 ^ (-γ)) ^ 2)
    (fun _ _ => sq_nonneg _) (mem_univ h0))

/-- `S ≥ 2^F > 0`. -/
theorem two_pow_le_S (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) : (2 : ℝ) ^ F ≤ S a γ F := by
  have := one_le_X hγ
  have := X_mul_le_S a γ F
  nlinarith [pow_pos (two_pos : (0 : ℝ) < 2) F]

theorem S_pos (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) : 0 < S a γ F := by
  have h1 := two_pow_le_S F hγ
  have : (0 : ℝ) < S a γ F := lt_of_lt_of_le (pow_pos two_pos F) h1
  exact_mod_cast this

private theorem four_zpow (u : ℤ) : (4 : ℝ) ^ u = ((2 : ℝ) ^ u) ^ 2 := by
  rw [sq, ← mul_zpow]; norm_num

/-- `a_h·2^σ = x_h·2^u` for `σ = u − γ`. -/
private theorem shift_eq (u σ : ℤ) (hσ : σ = u - γ) (h : Fin k) :
    a h * 2 ^ σ = a h * 2 ^ (-γ) * 2 ^ u := by
  rw [hσ, sub_eq_add_neg, zpow_add₀ two_ne_zero]; ring

/-- `‖a·2^σ‖² = 4^u·‖x‖²`. -/
private theorem sum_shift_sq (u σ : ℤ) (hσ : σ = u - γ) :
    ∑ h, (a h * 2 ^ σ) ^ 2 = X a γ * ((2 : ℝ) ^ u) ^ 2 := by
  rw [X, sum_mul]
  exact sum_congr rfl fun h _ => by rw [shift_eq u σ hσ h]; ring

/-- `‖a‖₂ = 2^γ·‖x‖₂`. -/
private theorem l2_a : l2 a = 2 ^ γ * √(X a γ) := by
  have e : ∑ h, a h ^ 2 = ((2 : ℝ) ^ γ) ^ 2 * X a γ := by
    rw [X, mul_sum]
    refine sum_congr rfl fun h _ => ?_
    have : a h = a h * 2 ^ (-γ) * 2 ^ γ := by
      rw [mul_assoc, ← zpow_add₀ two_ne_zero, neg_add_cancel, zpow_zero, mul_one]
    conv_lhs => rw [this]
    ring
  rw [l2, e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (zpow_pos two_pos _).le]

end facts

/-! ### L1a: the maximal `u` exists -/

/-- `Rule` is downward closed in `u`. -/
theorem rule_mono {c R : ℝ} {S : ℤ} {F : ℕ} (hc : 0 ≤ c) (hS : 0 ≤ S) {u v : ℤ} (hvu : v ≤ u)
    (hu : Rule c R S F u) : Rule c R S F v := by
  unfold Rule at *
  refine le_trans ?_ hu
  have : (4 : ℝ) ^ v ≤ 4 ^ u := zpow_le_zpow_right₀ (by norm_num) hvu
  have : (0 : ℝ) ≤ c * S := mul_nonneg hc (by exact_mod_cast hS)
  nlinarith

/-- **L1a.** For `c > 0`, `R > 0` and `S > 0`, `{u : c·S·4^u ≤ R·2^F}` is non-empty and bounded
above, so a largest such `u` exists. -/
theorem exists_max_rule {c R : ℝ} {S : ℤ} {F : ℕ} (hc : 0 < c) (hR : 0 < R) (hS : 0 < S) :
    ∃ u : ℤ, Rule c R S F u ∧ ∀ v, Rule c R S F v → v ≤ u := by
  have hS' : (0 : ℝ) < S := by exact_mod_cast hS
  have hcS : 0 < c * S := mul_pos hc hS'
  have h2F : (0 : ℝ) < 2 ^ F := pow_pos two_pos F
  apply Int.exists_greatest_of_bdd
  · -- bounded above: 4^u ≤ R·2^F/(c·S) < 4^n
    obtain ⟨n, hn⟩ := pow_unbounded_of_one_lt (R * 2 ^ F / (c * S)) (by norm_num : (1 : ℝ) < 4)
    refine ⟨n, fun u hu => ?_⟩
    unfold Rule at hu
    have : (4 : ℝ) ^ u < (4 : ℝ) ^ (n : ℤ) := by
      rw [zpow_natCast]
      refine lt_of_le_of_lt ?_ hn
      rw [le_div_iff₀ hcS]; linarith
    exact ((zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 4)).1 this).le
  · -- non-empty: u = −n with c·S < R·2^F·4^n
    obtain ⟨n, hn⟩ := pow_unbounded_of_one_lt (c * S / (R * 2 ^ F)) (by norm_num : (1 : ℝ) < 4)
    refine ⟨-(n : ℤ), ?_⟩
    unfold Rule
    rw [zpow_neg, zpow_natCast, ← div_eq_mul_inv, div_le_iff₀ (pow_pos (by norm_num) n)]
    rw [div_lt_iff₀ (mul_pos hR h2F)] at hn
    linarith

/-- The largest `u` fails the rule at `u + 1` (the only consequence of maximality used by L1d). -/
theorem not_rule_succ {c R : ℝ} {S : ℤ} {F : ℕ} {u : ℤ}
    (hmax : ∀ v, Rule c R S F v → v ≤ u) : ¬ Rule c R S F (u + 1) :=
  fun h => by have := hmax _ h; omega

/-- **L1a (RN).** With `2T − κ ≥ 1` the RN rule has a largest `u`. -/
theorem exists_u_rn (a : Fin k → ℝ) {γ : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) (F T κ : ℕ)
    (hTκ : κ + 1 ≤ 2 * T) :
    ∃ u, Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F u ∧
      ∀ v, Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F v → v ≤ u := by
  have : (1 : ℝ) ≤ 2 * T - κ := by
    have : ((κ + 1 : ℕ) : ℝ) ≤ ((2 * T : ℕ) : ℝ) := by exact_mod_cast hTκ
    push_cast at this; linarith
  exact exists_max_rule (by norm_num) (by positivity) (S_pos F hγ)

/-- **L1a (TRUNC).** With `T ≥ 1` the TRUNC rule has a largest `u`. -/
theorem exists_u_trunc (a : Fin k → ℝ) {γ : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) (F T : ℕ)
    (hT : 1 ≤ T) :
    ∃ u, Rule 1 ((T : ℝ) ^ 2) (S a γ F) F u ∧ ∀ v, Rule 1 ((T : ℝ) ^ 2) (S a γ F) F v → v ≤ u :=
  exists_max_rule one_pos (by have : (1 : ℝ) ≤ T := by exact_mod_cast hT
                              positivity) (S_pos F hγ)

/-- The soundness boundary: at `R = 0` (RN with `2T = κ`) no `u` satisfies the rule. -/
theorem no_rule_of_R_zero {c : ℝ} (hc : 0 < c) {S : ℤ} (hS : 0 < S) (F : ℕ) (u : ℤ) :
    ¬ Rule c 0 S F u := by
  unfold Rule
  have : (0 : ℝ) < S := by exact_mod_cast hS
  have : (0 : ℝ) < c * S * 4 ^ u := mul_pos (mul_pos hc this) (zpow_pos (by norm_num) _)
  simp only [zero_mul, not_le]; exact this

/-! ### L1b: the norm budget -/

/-- Core of L1b: the rule gives `c·‖a·2^σ‖² ≤ R`. -/
theorem rule_norm_sq {a : Fin k → ℝ} {γ : ℤ} {F : ℕ} {c R : ℝ} (hc : 0 ≤ c) {u σ : ℤ}
    (hσ : σ = u - γ) (hu : Rule c R (S a γ F) F u) :
    c * ∑ h, (a h * 2 ^ σ) ^ 2 ≤ R := by
  unfold Rule at hu
  rw [sum_shift_sq u σ hσ]
  rw [four_zpow] at hu
  have h1 := X_mul_le_S a γ F
  have h2F : (0 : ℝ) < 2 ^ F := pow_pos two_pos F
  have he : 0 ≤ c * ((2 : ℝ) ^ u) ^ 2 := mul_nonneg hc (sq_nonneg _)
  have : c * (X a γ * ((2 : ℝ) ^ u) ^ 2) * 2 ^ F ≤ R * 2 ^ F := by
    calc c * (X a γ * ((2 : ℝ) ^ u) ^ 2) * 2 ^ F
        = c * ((2 : ℝ) ^ u) ^ 2 * (X a γ * 2 ^ F) := by ring
      _ ≤ c * ((2 : ℝ) ^ u) ^ 2 * S a γ F := mul_le_mul_of_nonneg_left h1 he
      _ = c * S a γ F * ((2 : ℝ) ^ u) ^ 2 := by ring
      _ ≤ R * 2 ^ F := hu
  exact le_of_mul_le_mul_right this h2F

/-- **L1b (RN).** If `u` satisfies the RN rule (`4·S·4^u ≤ (2T − κ)²·2^F`), `κ ≤ 2T`,
`√k ≤ κ` and `q` rounds to nearest (`|q x − x| ≤ 1/2`), then `‖q(a·2^σ)‖₂ ≤ T`. -/
theorem budget_rn {a : Fin k → ℝ} {γ : ℤ} {F T κ : ℕ} {u σ : ℤ} (hσ : σ = u - γ)
    (hu : Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F u)
    (hTκ : (κ : ℝ) ≤ 2 * T) (hκ : √(k : ℝ) ≤ κ)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1 / 2) :
    l2 (fun h => (q (a h * 2 ^ σ) : ℝ)) ≤ T := by
  set y : Fin k → ℝ := fun h => a h * 2 ^ σ
  set δ : Fin k → ℝ := fun h => (q (y h) : ℝ) - y h
  -- ‖a·2^σ‖ ≤ T − κ/2
  have hy : l2 y ≤ T - κ / 2 := by
    have := rule_norm_sq (by norm_num) hσ hu
    rw [l2, Real.sqrt_le_left (by linarith)]
    have e : (2 * (T : ℝ) - κ) ^ 2 = 4 * (T - κ / 2) ^ 2 := by ring
    linarith
  -- ‖δ‖ ≤ √k/2 ≤ κ/2
  have hδ : l2 δ ≤ κ / 2 := by
    have := l2_le_of_abs_le (by norm_num) (fun h => hq (y h)) (v := δ)
    linarith
  have := l2_add_le y δ
  have e : (fun h => y h + δ h) = fun h => (q (a h * 2 ^ σ) : ℝ) := by
    funext h; simp [y, δ]
  rw [e] at this
  linarith

/-- **L1b (TRUNC).** If `u` satisfies the TRUNC rule (`S·4^u ≤ T²·2^F`) and `|q x| ≤ |x|`, then
`‖q(a·2^σ)‖₂ ≤ T`. -/
theorem budget_trunc {a : Fin k → ℝ} {γ : ℤ} {F T : ℕ} {u σ : ℤ} (hσ : σ = u - γ)
    (hu : Rule 1 ((T : ℝ) ^ 2) (S a γ F) F u)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ)| ≤ |x|) :
    l2 (fun h => (q (a h * 2 ^ σ) : ℝ)) ≤ T := by
  have := rule_norm_sq (by norm_num) hσ hu
  rw [l2, Real.sqrt_le_left (Nat.cast_nonneg _)]
  refine le_trans (sum_le_sum fun h _ => ?_) (by linarith)
  rw [← sq_abs, ← sq_abs (a h * 2 ^ σ)]
  exact pow_le_pow_left₀ (abs_nonneg _) (hq _) 2

/-- From `‖v‖₂ ≤ T` for an integer vector: `Σ v_h² ≤ T²` in `ℤ`. -/
theorem int_sum_sq_le {v : Fin k → ℤ} {T : ℕ} (h : l2 (fun i => (v i : ℝ)) ≤ T) :
    ∑ i, v i ^ 2 ≤ (T : ℤ) ^ 2 := by
  rw [l2, Real.sqrt_le_left (Nat.cast_nonneg _)] at h
  exact_mod_cast h

theorem budget_rn_int {a : Fin k → ℝ} {γ : ℤ} {F T κ : ℕ} {u σ : ℤ} (hσ : σ = u - γ)
    (hu : Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F u)
    (hTκ : (κ : ℝ) ≤ 2 * T) (hκ : √(k : ℝ) ≤ κ)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1 / 2) :
    ∑ h, q (a h * 2 ^ σ) ^ 2 ≤ (T : ℤ) ^ 2 :=
  int_sum_sq_le (v := fun h => q (a h * 2 ^ σ)) (budget_rn hσ hu hTκ hκ hq)

theorem budget_trunc_int {a : Fin k → ℝ} {γ : ℤ} {F T : ℕ} {u σ : ℤ} (hσ : σ = u - γ)
    (hu : Rule 1 ((T : ℝ) ^ 2) (S a γ F) F u)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ)| ≤ |x|) :
    ∑ h, q (a h * 2 ^ σ) ^ 2 ≤ (T : ℤ) ^ 2 :=
  int_sum_sq_le (v := fun h => q (a h * 2 ^ σ)) (budget_trunc hσ hu hq)

/-! ### L1c: the CRT precondition -/

/-- **L1c.** If `‖a'‖₂ ≤ T` and `‖b'‖₂ ≤ T` (integer vectors) and `T = ⌊√H⌋`, then
`Σ_h |a'_h||b'_h| ≤ T² ≤ H`; in particular `|Σ_h a'_h b'_h| ≤ H` (L2c's precondition). -/
theorem crt_precondition {a' b' : Fin k → ℤ} {T H : ℕ} (hT : T = Nat.sqrt H)
    (ha : ∑ h, a' h ^ 2 ≤ (T : ℤ) ^ 2) (hb : ∑ h, b' h ^ 2 ≤ (T : ℤ) ^ 2) :
    ∑ h, |a' h| * |b' h| ≤ (T : ℤ) ^ 2 ∧ (T : ℤ) ^ 2 ≤ H ∧ |∑ h, a' h * b' h| ≤ H := by
  have cs := sum_mul_sq_le_sq_mul_sq univ (fun h => |a' h|) (fun h => |b' h|)
  simp only [sq_abs] at cs
  have h0 : 0 ≤ ∑ h, |a' h| * |b' h| := sum_nonneg fun _ _ => by positivity
  have hT2 : (0 : ℤ) ≤ (T : ℤ) ^ 2 := by positivity
  have h1 : ∑ h, |a' h| * |b' h| ≤ (T : ℤ) ^ 2 := by
    have : (∑ h, |a' h| * |b' h|) ^ 2 ≤ ((T : ℤ) ^ 2) ^ 2 := by
      calc (∑ h, |a' h| * |b' h|) ^ 2 ≤ (∑ h, a' h ^ 2) * ∑ h, b' h ^ 2 := cs
        _ ≤ (T : ℤ) ^ 2 * (T : ℤ) ^ 2 :=
            mul_le_mul ha hb (sum_nonneg fun _ _ => sq_nonneg _) hT2
        _ = ((T : ℤ) ^ 2) ^ 2 := by ring
    exact (pow_le_pow_iff_left₀ h0 hT2 two_ne_zero).1 this
  have h2 : (T : ℤ) ^ 2 ≤ H := by
    have := Nat.sqrt_le' H; rw [← hT] at this; exact_mod_cast this
  refine ⟨h1, h2, ?_⟩
  calc |∑ h, a' h * b' h| ≤ ∑ h, |a' h * b' h| := abs_sum_le_sum_abs _ _
    _ = ∑ h, |a' h| * |b' h| := sum_congr rfl fun _ _ => abs_mul _ _
    _ ≤ H := h1.trans h2

/-! ### L1d: the σ bound -/

/-- Core of L1d: if `u + 1` fails the rule, `R = c·V²` with `V > 0`, and `2^γ ≤ |a_{h₀}|`, then
`2^{−σ} < 2‖a‖₂·s/V` with `s = √(1 + k·2^{−F})`. -/
theorem sigma_bound_core {a : Fin k → ℝ} {γ : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) {F : ℕ}
    {c R V : ℝ} (hc : 0 < c) (hV : 0 < V) (hR : R = c * V ^ 2) {u σ : ℤ} (hσ : σ = u - γ)
    (hmax : ¬ Rule c R (S a γ F) F (u + 1)) :
    (2 : ℝ) ^ (-σ) < 2 * l2 a * sF k F / V := by
  have hk : (univ : Finset (Fin k)).Nonempty := by
    obtain ⟨h0, _⟩ := hγ; exact ⟨h0, mem_univ _⟩
  have hX1 := one_le_X hγ
  set X := X a γ with hXdef
  set g : ℝ := 2 ^ γ with hg
  set e : ℝ := 2 ^ u with he
  have hg0 : 0 < g := zpow_pos two_pos _
  have he0 : 0 < e := zpow_pos two_pos _
  have h2F : (0 : ℝ) < 2 ^ F := pow_pos two_pos F
  -- s² = 1 + k·2^{−F}
  have hs2 : sF k F ^ 2 = 1 + k / 2 ^ F := by
    rw [sF, Real.sq_sqrt (by positivity), zpow_neg, zpow_natCast, div_eq_mul_inv]
  have hs0 : 0 ≤ sF k F := Real.sqrt_nonneg _
  -- 1. maximality: R·2^F < c·S·4^{u+1} = 4c·S·e²
  have h1 : R * 2 ^ F < 4 * c * S a γ F * e ^ 2 := by
    unfold Rule at hmax
    rw [not_le, four_zpow, zpow_add_one₀ two_ne_zero] at hmax
    linarith
  -- 2. S < ‖x‖²2^F + k ≤ ‖x‖²·2^F·s²
  have h2 : (S a γ F : ℝ) < X * 2 ^ F * sF k F ^ 2 := by
    have := S_lt a γ F hk
    have hk' : (k : ℝ) ≤ X * k := by nlinarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)]
    have : X * 2 ^ F * sF k F ^ 2 = X * 2 ^ F + X * k := by
      rw [hs2]; field_simp
    linarith
  -- 3. hence V² < 4·e²·‖x‖²·s²
  have h3 : V ^ 2 < (2 * e * √X * sF k F) ^ 2 := by
    have hcS : 4 * c * S a γ F * e ^ 2 < 4 * c * (X * 2 ^ F * sF k F ^ 2) * e ^ 2 := by
      have : 0 < 4 * c * e ^ 2 := by positivity
      nlinarith
    have : c * V ^ 2 * 2 ^ F < c * (4 * e ^ 2 * X * sF k F ^ 2) * 2 ^ F := by
      rw [← hR]; nlinarith
    have : V ^ 2 < 4 * e ^ 2 * X * sF k F ^ 2 := by
      have := lt_of_mul_lt_mul_right this h2F.le
      exact lt_of_mul_lt_mul_left this hc.le
    rw [mul_pow, mul_pow, mul_pow, Real.sq_sqrt (by linarith)]
    linarith
  have h4 : V < 2 * e * √X * sF k F := lt_of_pow_lt_pow_left₀ 2 (by positivity) h3
  -- 4. 2^{−σ} = g/e and ‖a‖ = g·√X
  have hσ' : (2 : ℝ) ^ (-σ) = g / e := by
    rw [hσ, neg_sub, zpow_sub₀ two_ne_zero]
  rw [lt_div_iff₀ hV, hσ', l2_a, ← hg]
  calc g / e * V < g / e * (2 * e * √X * sF k F) :=
        mul_lt_mul_of_pos_left h4 (div_pos hg0 he0)
    _ = 2 * (g * √X) * sF k F := by field_simp

/-- **L1d (RN).** For the largest RN `u` (so `u + 1` fails), with `κ < 2T`:
`2^{−σ} < 2‖a‖₂·s/(T − κ/2)`. -/
theorem sigma_bound_rn {a : Fin k → ℝ} {γ : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) {F T κ : ℕ}
    (hTκ : (κ : ℝ) < 2 * T) {u σ : ℤ} (hσ : σ = u - γ)
    (hmax : ¬ Rule 4 ((2 * T - κ : ℝ) ^ 2) (S a γ F) F (u + 1)) :
    (2 : ℝ) ^ (-σ) < 2 * l2 a * sF k F / (T - κ / 2) :=
  sigma_bound_core hγ (by norm_num) (by linarith) (by ring) hσ hmax

/-- **L1d (TRUNC).** For the largest TRUNC `u`, with `T ≥ 1`: `2^{−σ} < 2‖a‖₂·s/T`. -/
theorem sigma_bound_trunc {a : Fin k → ℝ} {γ : ℤ} (hγ : ∃ h, (2 : ℝ) ^ γ ≤ |a h|) {F T : ℕ}
    (hT : 1 ≤ T) {u σ : ℤ} (hσ : σ = u - γ)
    (hmax : ¬ Rule 1 ((T : ℝ) ^ 2) (S a γ F) F (u + 1)) :
    (2 : ℝ) ^ (-σ) < 2 * l2 a * sF k F / T :=
  sigma_bound_core hγ one_pos (by exact_mod_cast hT) (by ring) hσ hmax

/-! ### L1 in PROOFS.md's exact setting -/

/-- **L1 (RN), whole row.** With `γ` the row exponent, `κ = ⌈√k⌉`, `2T − κ ≥ 1`, and `u` the
largest integer with `4·S·4^u ≤ (2T − κ)²·2^F`, `σ = u − γ`, and RN quantization `q`:
`‖a'‖₂ ≤ T` (L1b) and `2^{−σ} < 2‖a‖₂·s/(T − κ/2)` (L1d). -/
theorem row_rn {a : Fin k → ℝ} {γ : ℤ} (hγ : RowExp a γ) {F T : ℕ}
    (hTκ : kappa k + 1 ≤ 2 * T) {u σ : ℤ} (hσ : σ = u - γ)
    (hu : Rule 4 ((2 * T - kappa k : ℝ) ^ 2) (S a γ F) F u)
    (hmax : ∀ v, Rule 4 ((2 * T - kappa k : ℝ) ^ 2) (S a γ F) F v → v ≤ u)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ) - x| ≤ 1 / 2) :
    l2 (fun h => (q (a h * 2 ^ σ) : ℝ)) ≤ T ∧
      (2 : ℝ) ^ (-σ) < 2 * l2 a * sF k F / (T - kappa k / 2) := by
  have : ((kappa k + 1 : ℕ) : ℝ) ≤ ((2 * T : ℕ) : ℝ) := by exact_mod_cast hTκ
  push_cast at this
  exact ⟨budget_rn hσ hu (by linarith) (sqrt_le_kappa k) hq,
    sigma_bound_rn hγ.1 (by linarith) hσ (not_rule_succ hmax)⟩

/-- **L1 (TRUNC), whole row.** With `T ≥ 1`, `u` the largest integer with `S·4^u ≤ T²·2^F`,
`σ = u − γ` and `|q x| ≤ |x|`: `‖a'‖₂ ≤ T` and `2^{−σ} < 2‖a‖₂·s/T`. -/
theorem row_trunc {a : Fin k → ℝ} {γ : ℤ} (hγ : RowExp a γ) {F T : ℕ} (hT : 1 ≤ T)
    {u σ : ℤ} (hσ : σ = u - γ) (hu : Rule 1 ((T : ℝ) ^ 2) (S a γ F) F u)
    (hmax : ∀ v, Rule 1 ((T : ℝ) ^ 2) (S a γ F) F v → v ≤ u)
    {q : ℝ → ℤ} (hq : ∀ x, |(q x : ℝ)| ≤ |x|) :
    l2 (fun h => (q (a h * 2 ^ σ) : ℝ)) ≤ T ∧ (2 : ℝ) ^ (-σ) < 2 * l2 a * sF k F / T :=
  ⟨budget_trunc hσ hu hq, sigma_bound_trunc hγ.1 hT hσ (not_rule_succ hmax)⟩

/-! ### Concrete quantizers satisfy the abstract hypotheses -/

/-- Round to nearest, ties to even. -/
noncomputable def rne (x : ℝ) : ℤ :=
  if Int.fract x = 1 / 2 then (if Even ⌊x⌋ then ⌊x⌋ else ⌊x⌋ + 1) else round x

theorem abs_rne_sub (x : ℝ) : |(rne x : ℝ) - x| ≤ 1 / 2 := by
  unfold rne
  have hx := Int.floor_add_fract x
  split_ifs with h1 h2
  · rw [h1] at hx; rw [abs_le]; constructor <;> linarith
  · rw [h1] at hx; push_cast; rw [abs_le]; constructor <;> linarith
  · rw [abs_sub_comm]; exact abs_sub_round x

/-- Truncation toward zero. -/
noncomputable def truncZ (x : ℝ) : ℤ := if 0 ≤ x then ⌊x⌋ else ⌈x⌉

theorem abs_truncZ_le (x : ℝ) : |(truncZ x : ℝ)| ≤ |x| := by
  unfold truncZ
  split_ifs with h
  · have h1 := Int.floor_le x
    have h2 : (0 : ℝ) ≤ ⌊x⌋ := by exact_mod_cast Int.floor_nonneg.2 h
    rw [abs_of_nonneg h2, abs_of_nonneg h]; exact h1
  · replace h := not_le.1 h
    have h1 := Int.le_ceil x
    have h2 : (⌈x⌉ : ℝ) ≤ 0 := by exact_mod_cast Int.ceil_le.2 (by simpa using h.le)
    rw [abs_of_nonpos h2, abs_of_neg h]; linarith

theorem abs_truncZ_sub_lt (x : ℝ) : |(truncZ x : ℝ) - x| < 1 := by
  unfold truncZ
  split_ifs
  · have := Int.floor_le x; have := Int.lt_floor_add_one x
    rw [abs_lt]; constructor <;> linarith
  · have := Int.le_ceil x; have := Int.ceil_lt_add_one x
    rw [abs_lt]; constructor <;> linarith

end Immerdink.Scaling
