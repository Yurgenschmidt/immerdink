import Mathlib

/-!
# L7a. Recursive floating-point summation (PROOFS.md §L7, NOTES-C1 IR5)

Setting: reals `d 0, d 1, …` and a running sum `c` with `c 0 = d 0` and, for each of the `m`
additions, the per-step rounding hypothesis `|c (i+1) − (c i + d (i+1))| ≤ u·|c i + d (i+1)|`
(binary64 RN of an exact sum; an addition that underflows is exact, so this holds absent
overflow). There is no IEEE model: the rounding enters only through that hypothesis, in the
style of L3's final-rounding hypothesis.

Results:
* `pow_mul_le_one`, `pow_sub_one_le_gam`: `(1 + u)^m − 1 ≤ γ_m = m·u/(1 − m·u)` for `m·u < 1`;
* `recsum_pow`, `recsum`: `|c m − Σ_{i ≤ m} d i| ≤ γ_m·Σ_{i ≤ m} |d i|` (Higham §4.2).

With `S` terms there are `m = S − 1` additions, so a segmented sum (L7) gets `γ_{S−1}`. The
native FMA chain of DESIGN §3.7 (`acc = +0`, then `k` steps `acc = fl(a_h b_h + acc)`, each
product exact inside the fma) is the case `d 0 = 0` with `m = k`: `γ_k`, L3c (3).
-/

namespace Immerdink.Summation

open Finset

/-- `γ_m = m·u/(1 − m·u)`. -/
noncomputable def gam (m : ℕ) (u : ℝ) : ℝ := m * u / (1 - m * u)

/-- `(1 + u)^m (1 − m·u) ≤ 1` for every `m` (`u ≥ 0`). -/
theorem pow_mul_le_one {u : ℝ} (hu : 0 ≤ u) (m : ℕ) : (1 + u) ^ m * (1 - m * u) ≤ 1 := by
  induction m with
  | zero => simp
  | succ m ih =>
    have hp : 0 ≤ (1 + u) ^ m := pow_nonneg (by linarith) m
    have key : (1 + u) * (1 - (m + 1 : ℕ) * u) ≤ 1 - m * u := by
      push_cast; nlinarith [sq_nonneg u, mul_nonneg (Nat.cast_nonneg m : (0 : ℝ) ≤ m) (sq_nonneg u)]
    calc (1 + u) ^ (m + 1) * (1 - (m + 1 : ℕ) * u)
        = (1 + u) ^ m * ((1 + u) * (1 - (m + 1 : ℕ) * u)) := by ring
      _ ≤ (1 + u) ^ m * (1 - m * u) := mul_le_mul_of_nonneg_left key hp
      _ ≤ 1 := ih

/-- `(1 + u)^m − 1 ≤ γ_m` for `m·u < 1`. -/
theorem pow_sub_one_le_gam {u : ℝ} (hu : 0 ≤ u) {m : ℕ} (hm : m * u < 1) :
    (1 + u) ^ m - 1 ≤ gam m u := by
  have hpos : 0 < 1 - m * u := by linarith
  have := pow_mul_le_one hu m
  rw [gam, le_div_iff₀ hpos]
  nlinarith

/-- `γ_m ≥ 0` for `m·u < 1`. -/
theorem gam_nonneg {u : ℝ} (hu : 0 ≤ u) {m : ℕ} (hm : m * u < 1) : 0 ≤ gam m u :=
  div_nonneg (mul_nonneg (Nat.cast_nonneg _) hu) (by linarith)

/-- **L7a, power form.** `|c m − Σ_{i ≤ m} d i| ≤ ((1 + u)^m − 1)·Σ_{i ≤ m} |d i|`. -/
theorem recsum_pow {u : ℝ} (hu : 0 ≤ u) {d c : ℕ → ℝ} (h0 : c 0 = d 0) (m : ℕ)
    (hstep : ∀ i < m, |c (i + 1) - (c i + d (i + 1))| ≤ u * |c i + d (i + 1)|) :
    |c m - ∑ i ∈ range (m + 1), d i| ≤ ((1 + u) ^ m - 1) * ∑ i ∈ range (m + 1), |d i| := by
  induction m with
  | zero => simp [h0]
  | succ m ih =>
    have ih := ih fun i hi => hstep i (by omega)
    have hs := hstep m (by omega)
    set A := ∑ i ∈ range (m + 1), |d i| with hA
    set e := c m - ∑ i ∈ range (m + 1), d i
    have hA0 : 0 ≤ A := sum_nonneg fun _ _ => abs_nonneg _
    have hp : 1 ≤ (1 + u) ^ m := one_le_pow₀ (by linarith)
    -- |c m| ≤ A + |e| ≤ (1 + u)^m A
    have hc : |c m| ≤ (1 + u) ^ m * A := by
      have h1 : |∑ i ∈ range (m + 1), d i| ≤ A := abs_sum_le_sum_abs _ _
      have h2 : |c m| ≤ |e| + |∑ i ∈ range (m + 1), d i| := by
        have := abs_add_le e (∑ i ∈ range (m + 1), d i); simp [e] at this ⊢; linarith
      nlinarith
    have hθ : |c (m + 1) - (c m + d (m + 1))| ≤ u * (|c m| + |d (m + 1)|) :=
      hs.trans (mul_le_mul_of_nonneg_left (abs_add_le _ _) hu)
    have hs1 : ∑ i ∈ range (m + 1 + 1), d i = ∑ i ∈ range (m + 1), d i + d (m + 1) :=
      sum_range_succ _ _
    have hs2 : ∑ i ∈ range (m + 1 + 1), |d i| = A + |d (m + 1)| := by
      rw [hA, sum_range_succ]
    have hsplit : c (m + 1) - (∑ i ∈ range (m + 1), d i + d (m + 1))
        = (c (m + 1) - (c m + d (m + 1))) + e := by simp only [e]; ring
    rw [hs1, hs2, hsplit]
    have hd := abs_nonneg (d (m + 1))
    have hpow : (1 + u) ^ (m + 1) = (1 + u) ^ m * (1 + u) := pow_succ _ _
    have hu1 : u ≤ (1 + u) ^ (m + 1) - 1 := by rw [hpow]; nlinarith
    calc |(c (m + 1) - (c m + d (m + 1))) + e|
        ≤ |c (m + 1) - (c m + d (m + 1))| + |e| := abs_add_le _ _
      _ ≤ u * ((1 + u) ^ m * A + |d (m + 1)|) + ((1 + u) ^ m - 1) * A := by
          nlinarith [mul_le_mul_of_nonneg_left hc hu]
      _ = ((1 + u) ^ (m + 1) - 1) * A + u * |d (m + 1)| := by rw [hpow]; ring
      _ ≤ ((1 + u) ^ (m + 1) - 1) * (A + |d (m + 1)|) := by nlinarith

/-- **L7a (γ form).** `|c m − Σ_{i ≤ m} d i| ≤ γ_m·Σ_{i ≤ m} |d i|` for `m·u < 1`. -/
theorem recsum {u : ℝ} (hu : 0 ≤ u) {d c : ℕ → ℝ} (h0 : c 0 = d 0) {m : ℕ} (hm : m * u < 1)
    (hstep : ∀ i < m, |c (i + 1) - (c i + d (i + 1))| ≤ u * |c i + d (i + 1)|) :
    |c m - ∑ i ∈ range (m + 1), d i| ≤ gam m u * ∑ i ∈ range (m + 1), |d i| :=
  (recsum_pow hu h0 m hstep).trans
    (mul_le_mul_of_nonneg_right (pow_sub_one_le_gam hu hm) (sum_nonneg fun _ _ => abs_nonneg _))

/-- **L3c (3), the native FMA chain** (DESIGN §3.7 / §6f.2 DIRECT): `acc 0 = 0` and `k` steps
`acc (i+1) = fl(p i + acc i)` with each product exact inside the fma
(`|acc (i+1) − (acc i + p i)| ≤ u|acc i + p i|`): `|acc k − Σ_{i<k} p i| ≤ γ_k Σ_{i<k} |p i|`. -/
theorem fma_chain {u : ℝ} (hu : 0 ≤ u) {p acc : ℕ → ℝ} (h0 : acc 0 = 0) {k : ℕ}
    (hk : k * u < 1)
    (hstep : ∀ i < k, |acc (i + 1) - (acc i + p i)| ≤ u * |acc i + p i|) :
    |acc k - ∑ i ∈ range k, p i| ≤ gam k u * ∑ i ∈ range k, |p i| := by
  let d : ℕ → ℝ := fun i => if i = 0 then 0 else p (i - 1)
  have hd0 : acc 0 = d 0 := by simp [d, h0]
  have hs : ∀ i < k, |acc (i + 1) - (acc i + d (i + 1))| ≤ u * |acc i + d (i + 1)| := by
    intro i hi; simpa [d] using hstep i hi
  have := recsum hu hd0 hk hs
  have e1 : ∑ i ∈ range (k + 1), d i = ∑ i ∈ range k, p i := by
    rw [sum_range_succ']; simp [d]
  have e2 : ∑ i ∈ range (k + 1), |d i| = ∑ i ∈ range k, |p i| := by
    rw [sum_range_succ']; simp [d]
  rwa [e1, e2] at this

end Immerdink.Summation
