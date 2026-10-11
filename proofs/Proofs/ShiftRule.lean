import Proofs.Scaling

/-!
# The shift rule as computed (DESIGN.md §3.2 steps 3–4)

`Scaling.lean` proves L1 for the largest `u` with `c·S·4^u ≤ R·2^F` and `κ = ⌈√k⌉`. DESIGN §3.2
computes both in integers, and `spec/OzakiScaling.tla` (`ShiftIsMaximal`, `UOf`) checks the
computation at toy widths. Here it is proved for all widths:

* **κ** (`kappa_isqrt`): `c = isqrt(k); κ = c + (c² < k)` equals `⌈√k⌉` (`Scaling.kappa`);
* **u** (`shift_estimate`, `u_eq_estimate`): with `A = c·S` and `B = R·2^F` positive integers,
  the estimate `ĉ = (bitlen(B) − bitlen(A) + 1) >> 1` (an arithmetic, i.e. floor, shift)
  satisfies the rule at `ĉ − 1` and fails it at `ĉ + 1`, so the largest `u` is
  `Rule(ĉ) ? ĉ : ĉ − 1`: one exact compare suffices;
* **S** (`S_le`, `S_lt_two_pow_128`): with `F = 60` and the row exponent's upper half
  (`max|x_h| < 2`), every term `⌈x_h²·2^F⌉ ≤ 2⁶²`, so `S < 2¹²⁸` (uint128) for `k < 2⁶⁶`.
-/

namespace Immerdink.Scaling

open Finset Real

/-! ### κ = ⌈√k⌉ in integers -/

/-- **DESIGN §3.2, κ.** `c = isqrt(k); κ = c + (c² < k)` is `⌈√k⌉`. -/
theorem kappa_isqrt (k : ℕ) :
    Nat.sqrt k + (if Nat.sqrt k * Nat.sqrt k < k then 1 else 0) = kappa k := by
  unfold kappa
  set c := Nat.sqrt k with hc
  have hle : c * c ≤ k := Nat.sqrt_le k
  have hlt : k < (c + 1) * (c + 1) := Nat.lt_succ_sqrt k
  split_ifs with h
  · -- c² < k < (c+1)²: c < √k ≤ c + 1
    symm
    rw [Nat.ceil_eq_iff (by omega)]
    simp only [Nat.add_sub_cancel, Nat.cast_add, Nat.cast_one]
    constructor
    · rw [Real.lt_sqrt (Nat.cast_nonneg _)]
      exact_mod_cast (by nlinarith : c ^ 2 < k)
    · rw [Real.sqrt_le_left (by positivity)]
      exact_mod_cast (by nlinarith : k ≤ (c + 1) ^ 2)
  · -- c² = k: √k = c
    have hk : k = c * c := by omega
    rw [add_zero, hk]
    push_cast
    rw [Real.sqrt_mul_self (Nat.cast_nonneg _), Nat.ceil_natCast]

/-! ### u by one estimate and one compare -/

private theorem two_pow_size_pred {n : ℕ} (hn : 0 < n) : (2 : ℝ) ^ ((n.size : ℤ) - 1) ≤ n := by
  have hs : 0 < n.size := Nat.size_pos.2 hn
  have h : 2 ^ (n.size - 1) ≤ n := Nat.lt_size.1 (by omega)
  have e : ((n.size : ℤ) - 1) = ((n.size - 1 : ℕ) : ℤ) := by omega
  rw [e, zpow_natCast]
  exact_mod_cast h

private theorem lt_two_pow_size (n : ℕ) : (n : ℝ) < (2 : ℝ) ^ (n.size : ℤ) := by
  rw [zpow_natCast]
  exact_mod_cast Nat.lt_size_self n

private theorem four_zpow' (u : ℤ) : (4 : ℝ) ^ u = (2 : ℝ) ^ (2 * u) := by
  rw [zpow_mul]; norm_num

/-- **DESIGN §3.2, the estimate.** For positive integers `A` (`= c·S`) and `B` (`= R·2^F`),
`ĉ = ⌊(bitlen B − bitlen A + 1)/2⌋` has `A·4^{ĉ−1} ≤ B < A·4^{ĉ+1}`. -/
theorem shift_estimate {nA nB : ℕ} (hA : 0 < nA) (hB : 0 < nB) :
    let c : ℤ := ((nB.size : ℤ) - nA.size + 1) / 2
    (nA : ℝ) * 4 ^ (c - 1) ≤ nB ∧ (nB : ℝ) < nA * 4 ^ (c + 1) := by
  intro c
  set a : ℤ := nA.size
  set b : ℤ := nB.size
  have hc1 : 2 * c ≤ b - a + 1 := by omega
  have hc2 : b - a ≤ 2 * c := by omega
  have hA1 := two_pow_size_pred hA
  have hA2 := lt_two_pow_size nA
  have hB1 := two_pow_size_pred hB
  have hB2 := lt_two_pow_size nB
  have h2 : (1 : ℝ) ≤ 2 := by norm_num
  constructor
  · -- A·4^{c−1} < 2^a·2^{2c−2} ≤ 2^{b−1} ≤ B
    rw [four_zpow']
    have hpos : (0 : ℝ) < (2 : ℝ) ^ (2 * (c - 1)) := zpow_pos two_pos _
    calc (nA : ℝ) * 2 ^ (2 * (c - 1)) ≤ 2 ^ a * 2 ^ (2 * (c - 1)) :=
          mul_le_mul_of_nonneg_right hA2.le hpos.le
      _ = 2 ^ (a + 2 * (c - 1)) := by rw [zpow_add₀ two_ne_zero]
      _ ≤ 2 ^ (b - 1) := zpow_le_zpow_right₀ h2 (by omega)
      _ ≤ nB := hB1
  · -- B < 2^b ≤ 2^{a−1}·2^{2c+2} ≤ A·4^{c+1}
    rw [four_zpow']
    have hpos : (0 : ℝ) < (2 : ℝ) ^ (2 * (c + 1)) := zpow_pos two_pos _
    calc (nB : ℝ) < 2 ^ b := hB2
      _ ≤ 2 ^ ((a - 1) + 2 * (c + 1)) := zpow_le_zpow_right₀ h2 (by omega)
      _ = 2 ^ (a - 1) * 2 ^ (2 * (c + 1)) := by rw [zpow_add₀ two_ne_zero]
      _ ≤ nA * 2 ^ (2 * (c + 1)) := mul_le_mul_of_nonneg_right hA1 hpos.le

/-- **DESIGN §3.2, one compare.** If `c·S = A` and `R·2^F = B` are positive integers and `u` is
the largest integer with `Rule c R S F u`, then `u = Rule(ĉ) ? ĉ : ĉ − 1` with
`ĉ = ⌊(bitlen B − bitlen A + 1)/2⌋`. -/
theorem u_eq_estimate {c R : ℝ} {S : ℤ} {F : ℕ} {nA nB : ℕ} (hA : 0 < nA) (hB : 0 < nB)
    (hnA : (nA : ℝ) = c * S) (hnB : (nB : ℝ) = R * 2 ^ F) {u : ℤ} (hu : Rule c R S F u)
    (hmax : ∀ v, Rule c R S F v → v ≤ u) :
    let ĉ : ℤ := ((nB.size : ℤ) - nA.size + 1) / 2
    u = if Rule c R S F ĉ then ĉ else ĉ - 1 := by
  intro ĉ
  obtain ⟨h1, h2⟩ := shift_estimate hA hB
  have hlo : ĉ - 1 ≤ u := hmax _ (by unfold Rule; rw [← hnA, ← hnB]; exact h1)
  have hApos : (0 : ℝ) < nA := by exact_mod_cast hA
  have hhi : u ≤ ĉ := by
    by_contra hcon
    have hle : ĉ + 1 ≤ u := by omega
    have : (nA : ℝ) * 4 ^ (ĉ + 1) ≤ nA * 4 ^ u :=
      mul_le_mul_of_nonneg_left (zpow_le_zpow_right₀ (by norm_num) hle) hApos.le
    unfold Rule at hu; rw [← hnA, ← hnB] at hu
    linarith
  split_ifs with h
  · exact le_antisymm hhi (hmax _ h)
  · have : u ≠ ĉ := fun he => h (he ▸ hu)
    omega

/-! ### The width of S -/

/-- With `F = 60` and `|x_h| < 2` (the upper half of the row exponent), every term of `S` is
`≤ 2⁶²`, so `S ≤ k·2⁶²`. -/
theorem S_le {k : ℕ} {a : Fin k → ℝ} {γ : ℤ} (hγ : ∀ h, |a h| < (2 : ℝ) ^ (γ + 1)) :
    S a γ 60 ≤ k * 2 ^ 62 := by
  unfold S
  have hterm : ∀ h, ⌈(a h * 2 ^ (-γ)) ^ 2 * 2 ^ 60⌉ ≤ (2 : ℤ) ^ 62 := by
    intro h
    rw [Int.ceil_le]
    have hx : |a h * 2 ^ (-γ)| < 2 := by
      rw [abs_mul, abs_of_pos (zpow_pos two_pos _)]
      have e : (2 : ℝ) ^ (γ + 1) * 2 ^ (-γ) = 2 := by
        rw [← zpow_add₀ two_ne_zero]; norm_num
      calc |a h| * 2 ^ (-γ) < 2 ^ (γ + 1) * 2 ^ (-γ) :=
            mul_lt_mul_of_pos_right (hγ h) (zpow_pos two_pos _)
        _ = 2 := e
    have hsq : (a h * 2 ^ (-γ)) ^ 2 ≤ 4 := by
      rw [← sq_abs]; nlinarith [abs_nonneg (a h * 2 ^ (-γ))]
    push_cast
    calc (a h * 2 ^ (-γ)) ^ 2 * 2 ^ 60 ≤ 4 * 2 ^ 60 :=
          mul_le_mul_of_nonneg_right hsq (by positivity)
      _ = 2 ^ 62 := by norm_num
  calc ∑ h, ⌈(a h * 2 ^ (-γ)) ^ 2 * 2 ^ 60⌉ ≤ ∑ _h : Fin k, (2 : ℤ) ^ 62 :=
        sum_le_sum fun h _ => hterm h
    _ = k * 2 ^ 62 := by simp

/-- **DESIGN §3.2, S fits uint128** for `k < 2⁶⁶` (each term is `≤ 2⁶² < 2⁶³`). -/
theorem S_lt_two_pow_128 {k : ℕ} (hk : k < 2 ^ 66) {a : Fin k → ℝ} {γ : ℤ}
    (hγ : ∀ h, |a h| < (2 : ℝ) ^ (γ + 1)) : S a γ 60 < 2 ^ 128 := by
  have := S_le hγ
  have hk' : (k : ℤ) < 2 ^ 66 := by exact_mod_cast hk
  calc S a γ 60 ≤ k * 2 ^ 62 := this
    _ < 2 ^ 66 * 2 ^ 62 := by
        have : (0 : ℤ) < 2 ^ 62 := by positivity
        nlinarith
    _ = 2 ^ 128 := by norm_num

end Immerdink.Scaling
