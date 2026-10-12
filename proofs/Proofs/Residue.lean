import Proofs.Exact

/-!
# Residues and exact reductions mod p (DESIGN.md §3.4, §3.4.1, §3.5, §3.6)

The pieces between the quantized integers `a'` and the INT8 GEMM that `Exact.lean` took as
hypotheses (`r ≡ a'`, `|r·s| ≤ Q`):

* `smod x p`, the centred residue `SMod(x, p) ∈ [−⌊p/2⌋, ⌈p/2⌉−1]` of DESIGN §3.4
  (`smod_range`), its congruence (`smod_modEq`), the int8 store for `p ≤ 256`
  (`smod_int8`), and the product bounds `|r·s| ≤ 128²` (`smod_mul_abs_le`) and
  `r·s ≥ −128·127` (`smod_mul_ge`);
* the `2^ε mod p` table factorization (`smod_table`), the odd-`p` negation the packs use
  (`smod_neg_odd`, and the byte form for `p = 256`, `smod_neg_byte`), and the v1.14
  byte-weight residue (`byte_weights_modEq`, `byte_weights_le`);
* the multiply-high reductions: the generic Granlund–Montgomery lemma
  (`mulhi_div_eq`), the A5 `mod_p_bytes32` lemma (`mod_p_bytes32`), the epilogue's
  `umulhi64` Barrett reduction with the bias (`barrett_epilogue`) and the CRT's
  `y = (q·W) mod p` (`crt_y_reduction`);
* the capstone `emulated_product_exact_smod`: `Exact.emulated_product_exact` with the residues
  the device actually computes, `r = smod a' p`, `s = smod b' p`, so the residue hypotheses
  are discharged.

`smod` is defined from the non-negative mod (`Int.emod`), as DESIGN §3.4 requires ("C `%` on a
negative μ is wrong").
-/

namespace Immerdink.Residue

open Finset

/-! ### SMod -/

/-- `SMod(x, p)`: the residue of `x` mod `p` in `[−⌊p/2⌋, ⌈p/2⌉−1]` (DESIGN §3.4). Computed
from the non-negative residue `r = x mod p ∈ [0, p)`: `r` if `r < ⌈p/2⌉`, else `r − p`. -/
def smod (x : ℤ) (p : ℕ) : ℤ :=
  if x % (p : ℤ) < ((p : ℤ) + 1) / 2 then x % (p : ℤ) else x % (p : ℤ) - p

/-- `SMod(x, p) ≡ x (mod p)`. -/
theorem smod_modEq (x : ℤ) (p : ℕ) : smod x p ≡ x [ZMOD p] := by
  unfold smod
  split_ifs
  · exact Int.mod_modEq x p
  · have h : x % (p : ℤ) - p ≡ x % (p : ℤ) - 0 [ZMOD p] :=
      Int.ModEq.sub_left _ ((Int.modEq_zero_iff_dvd).2 dvd_rfl)
    rw [sub_zero] at h
    exact h.trans (Int.mod_modEq x p)

/-- `SMod` depends only on the class of `x` mod `p`. -/
theorem smod_congr {x y : ℤ} {p : ℕ} (h : x ≡ y [ZMOD p]) : smod x p = smod y p := by
  unfold smod; rw [show x % (p : ℤ) = y % (p : ℤ) from h]

/-- **DESIGN §3.4, the range.** For `p ≥ 1`: `−⌊p/2⌋ ≤ SMod(x, p) ≤ ⌈p/2⌉ − 1`. -/
theorem smod_range (x : ℤ) {p : ℕ} (hp : 0 < p) :
    -((p : ℤ) / 2) ≤ smod x p ∧ smod x p ≤ ((p : ℤ) + 1) / 2 - 1 := by
  have hp' : (0 : ℤ) < p := by exact_mod_cast hp
  have h0 := Int.emod_nonneg x hp'.ne'
  have h1 := Int.emod_lt_of_pos x hp'
  unfold smod
  generalize x % (p : ℤ) = r at h0 h1 ⊢
  split_ifs with h <;> omega

/-- **DESIGN §3.4, the int8 store.** For `1 ≤ p ≤ 256`, `SMod(x, p) ∈ [−128, 127]`. -/
theorem smod_int8 (x : ℤ) {p : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) :
    -128 ≤ smod x p ∧ smod x p ≤ 127 := by
  have h := smod_range x hp
  have : (p : ℤ) ≤ 256 := by exact_mod_cast hp256
  omega

/-- Two int8 residues: `|r·s| ≤ 128² = Q` (L4's per-product bound). -/
theorem smod_mul_abs_le (x y : ℤ) {p p' : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) (hp' : 0 < p')
    (hp'256 : p' ≤ 256) : |smod x p * smod y p'| ≤ 128 ^ 2 := by
  have h1 := smod_int8 x hp hp256
  have h2 := smod_int8 y hp' hp'256
  rw [abs_mul]
  have a1 : |smod x p| ≤ 128 := abs_le.2 ⟨by omega, by omega⟩
  have a2 : |smod y p'| ≤ 128 := abs_le.2 ⟨by omega, by omega⟩
  calc |smod x p| * |smod y p'| ≤ 128 * 128 :=
        mul_le_mul a1 a2 (abs_nonneg _) (by norm_num)
    _ = 128 ^ 2 := by norm_num

/-- Two int8 values: `r·s ≥ −128·127` (the exact lower end of DESIGN §3.5's epilogue range;
`r·s ≤ 128²` is `smod_mul_abs_le`). -/
theorem int8_mul_ge {r s : ℤ} (hr : -128 ≤ r ∧ r ≤ 127) (hs : -128 ≤ s ∧ s ≤ 127) :
    -(128 * 127) ≤ r * s := by
  rcases le_or_gt 0 r with h | h
  · nlinarith
  · nlinarith

theorem smod_mul_ge (x y : ℤ) {p p' : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) (hp' : 0 < p')
    (hp'256 : p' ≤ 256) : -(128 * 127) ≤ smod x p * smod y p' :=
  int8_mul_ge (smod_int8 x hp hp256) (smod_int8 y hp' hp'256)

/-- **DESIGN §3.4, the table form.** With `a' = μ·2^ε`:
`SMod((μ mod p)·(2^ε mod p), p) = SMod(a', p)`, using the non-negative `mod`. -/
theorem smod_table (μ : ℤ) (ε : ℕ) (p : ℕ) :
    smod ((μ % (p : ℤ)) * ((2 : ℤ) ^ ε % (p : ℤ))) p = smod (μ * 2 ^ ε) p :=
  smod_congr (Int.ModEq.mul (Int.mod_modEq μ p) (Int.mod_modEq _ p))

/-- **DESIGN §3.4.1, the sign.** For odd `p`, `SMod(−x, p) = −SMod(x, p)`; the packs compute
the residue of `|a'|` and negate it for `a' < 0`. -/
theorem smod_neg_odd (x : ℤ) {p : ℕ} (hodd : Odd p) : smod (-x) p = -smod x p := by
  obtain ⟨m, rfl⟩ := hodd
  have hp' : (0 : ℤ) < ((2 * m + 1 : ℕ) : ℤ) := by positivity
  set P : ℤ := ((2 * m + 1 : ℕ) : ℤ) with hP
  have hPm : P = 2 * m + 1 := by rw [hP]; push_cast; ring
  have h0 := Int.emod_nonneg x hp'.ne'
  have h1 := Int.emod_lt_of_pos x hp'
  -- the residue of −x is 0 or P − r
  have hneg : (-x) % P = if x % P = 0 then 0 else P - x % P := by
    split_ifs with h
    · have : P ∣ x := Int.dvd_of_emod_eq_zero h
      exact Int.emod_eq_zero_of_dvd (dvd_neg.2 this)
    · have hc : (-x) ≡ P - x % P [ZMOD P] := by
        have e1 : P - x % P ≡ 0 - x % P [ZMOD P] :=
          Int.ModEq.sub_right _ ((Int.modEq_zero_iff_dvd).2 dvd_rfl)
        have e2 : (0 : ℤ) - x % P ≡ 0 - x [ZMOD P] := Int.ModEq.sub_left _ (Int.mod_modEq x P)
        simpa using (e1.trans e2).symm
      rw [show (-x) % P = (P - x % P) % P from hc]
      exact Int.emod_eq_of_lt (by omega) (by omega)
  unfold smod
  rw [← hP, hneg]
  generalize x % P = r at h0 h1 hneg ⊢
  split_ifs <;> omega

/-- For `p = 256` (even), the packs' negation agrees with `SMod` as an int8 byte:
`SMod(−x, 256) ≡ −SMod(x, 256) (mod 256)`. -/
theorem smod_neg_byte (x : ℤ) : smod (-x) 256 ≡ -smod x 256 [ZMOD 256] :=
  ((smod_modEq (-x) 256).trans (Int.ModEq.neg (smod_modEq x 256)).symm)

/-! ### The v1.14 byte-weight residue (DESIGN §3.4.1) -/

/-- **Byte weights.** For `X = Σ_{i<n} b_i·2^{8i}`, the dp4a sum with weights `2^{8i} mod p`
is congruent to `X`: `Σ_i b_i·(2^{8i} mod p) ≡ X (mod p)`. -/
theorem byte_weights_modEq (b : ℕ → ℕ) (n p : ℕ) :
    ∑ i ∈ range n, b i * (2 ^ (8 * i) % p) ≡ ∑ i ∈ range n, b i * 2 ^ (8 * i) [MOD p] := by
  induction n with
  | zero => simp [Nat.ModEq]
  | succ n ih =>
    rw [sum_range_succ, sum_range_succ]
    exact Nat.ModEq.add ih (Nat.ModEq.mul_left _ (Nat.mod_modEq _ _))

/-- The dp4a sum of `n` bytes with weights below `p ≤ 256` is `≤ n·255·255` (DESIGN §3.4.1:
`s ≤ 10·255·255 < 2²⁰` for the ten bytes of `X < 2⁷⁸`). -/
theorem byte_weights_le (b : ℕ → ℕ) (n : ℕ) {p : ℕ} (hp : 0 < p) (hp256 : p ≤ 256)
    (hb : ∀ i < n, b i ≤ 255) :
    ∑ i ∈ range n, b i * (2 ^ (8 * i) % p) ≤ n * (255 * 255) := by
  calc ∑ i ∈ range n, b i * (2 ^ (8 * i) % p) ≤ ∑ _i ∈ range n, 255 * 255 := by
        refine sum_le_sum fun i hi => Nat.mul_le_mul (hb i (mem_range.1 hi)) ?_
        have := Nat.mod_lt (2 ^ (8 * i)) hp
        omega
    _ = n * (255 * 255) := by simp

/-! ### Multiply-high reductions -/

/-- **Granlund–Montgomery, round-up form.** Let `m = ⌈D/p⌉` and `e = m·p − D` (so `0 ≤ e < p`).
If `s·e < D` then `⌊s·m/D⌋ = ⌊s/p⌋`. -/
theorem mulhi_div_eq {D p s : ℕ} (hp : 0 < p) (hse : s * ((D + p - 1) / p * p - D) < D) :
    s * ((D + p - 1) / p) / D = s / p := by
  set m := (D + p - 1) / p with hm
  -- D ≤ m·p ≤ D + p − 1
  have hdm := Nat.div_add_mod (D + p - 1) p
  have hml := Nat.mod_lt (D + p - 1) hp
  rw [← hm, mul_comm] at hdm
  have hmp1 : D ≤ m * p := by omega
  set e := m * p - D with he
  have hmpe : m * p = D + e := by omega
  have hD : 0 < D := by
    rcases Nat.eq_zero_or_pos D with h | h
    · rw [h] at hse; omega
    · exact h
  have hs := Nat.div_add_mod s p
  have hr : s % p < p := Nat.mod_lt s hp
  set q := s / p
  set r := s % p
  -- multiply everything by p: p·(s·m) = s·D + s·e
  have key : p * (s * m) = s * D + s * e := by rw [← mul_add, ← hmpe]; ring
  have hlo : q * D ≤ s * m := by
    refine Nat.le_of_mul_le_mul_left ?_ hp
    rw [key]
    have h1 : p * q ≤ s := by omega
    calc p * (q * D) = (p * q) * D := by ring
      _ ≤ s * D := Nat.mul_le_mul_right _ h1
      _ ≤ s * D + s * e := Nat.le_add_right _ _
  have hhi : s * m < (q + 1) * D := by
    refine lt_of_mul_lt_mul_left ?_ (Nat.zero_le p)
    rw [key]
    -- s·D + s·e < s·D + D ≤ (p·q + r + 1)·D ≤ p·(q + 1)·D
    have h1 : s * D + s * e < s * D + D := by omega
    have h2 : s * D + D ≤ p * ((q + 1) * D) := by
      have h3 : s + 1 ≤ p * (q + 1) := by rw [mul_add, mul_one]; omega
      calc s * D + D = (s + 1) * D := by ring
        _ ≤ p * (q + 1) * D := Nat.mul_le_mul_right _ h3
        _ = p * ((q + 1) * D) := by ring
    omega
  have h1 : q ≤ s * m / D := (Nat.le_div_iff_mul_le hD).2 hlo
  have h2 : s * m / D < q + 1 := (Nat.div_lt_iff_lt_mul hD).2 hhi
  omega

/-- **The A5 `mod_p_bytes32` lemma (DESIGN §3.4.1).** For `p ≤ 256` and `s < 2²⁴`,
`s − p·umulhi(s, ⌈2³²/p⌉) = s mod p`. -/
theorem mod_p_bytes32 {p s : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) (hs : s < 2 ^ 24) :
    s - p * (s * ((2 ^ 32 + p - 1) / p) / 2 ^ 32) = s % p := by
  have he : (2 ^ 32 + p - 1) / p * p - 2 ^ 32 < p := by
    have := Nat.div_mul_le_self (2 ^ 32 + p - 1) p
    omega
  have hse : s * ((2 ^ 32 + p - 1) / p * p - 2 ^ 32) < 2 ^ 32 := by
    calc s * ((2 ^ 32 + p - 1) / p * p - 2 ^ 32) ≤ s * 256 :=
          Nat.mul_le_mul_left _ (by omega)
      _ < 2 ^ 24 * 256 := by omega
      _ = 2 ^ 32 := by norm_num
  rw [mulhi_div_eq hp hse]
  have := Nat.div_add_mod s p
  omega

/-- **The epilogue reduction (DESIGN §3.5, `cert_barrett`).** For `2 ≤ p ≤ 256`, an int32
input `v ≥ −2³¹` and `n = v + bias` with `bias = p·⌈2³¹/p⌉` fitting a u32, the quotient
`q = umulhi64(⌈2⁶⁴/p⌉, n)` gives `n − p·q = v mod p` (the non-negative residue). -/
theorem barrett_epilogue {p : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) {v : ℤ}
    (hv : -(2 ^ 31) ≤ v) (hn : v + (p * ((2 ^ 31 + p - 1) / p) : ℕ) < 2 ^ 32) :
    let n : ℕ := (v + (p * ((2 ^ 31 + p - 1) / p) : ℕ)).toNat
    ((n - p * (n * ((2 ^ 64 + p - 1) / p) / 2 ^ 64) : ℕ) : ℤ) = v % p := by
  intro n
  set bias : ℕ := p * ((2 ^ 31 + p - 1) / p) with hbias
  have hb31 : 2 ^ 31 ≤ bias := by
    have := Nat.div_add_mod (2 ^ 31 + p - 1) p
    have := Nat.mod_lt (2 ^ 31 + p - 1) hp
    rw [hbias]; omega
  have hn0 : 0 ≤ v + (bias : ℤ) := by
    have : (2 ^ 31 : ℤ) ≤ bias := by exact_mod_cast hb31
    linarith
  have hnv : (n : ℤ) = v + bias := Int.toNat_of_nonneg hn0
  have hn32 : n < 2 ^ 32 := by
    have : (n : ℤ) < 2 ^ 32 := by rw [hnv]; exact hn
    exact_mod_cast this
  have he : (2 ^ 64 + p - 1) / p * p - 2 ^ 64 < p := by
    have := Nat.div_mul_le_self (2 ^ 64 + p - 1) p
    omega
  have hse : n * ((2 ^ 64 + p - 1) / p * p - 2 ^ 64) < 2 ^ 64 := by
    calc n * ((2 ^ 64 + p - 1) / p * p - 2 ^ 64) ≤ n * 256 :=
          Nat.mul_le_mul_left _ (by omega)
      _ < 2 ^ 32 * 256 := by omega
      _ ≤ 2 ^ 64 := by norm_num
  rw [mulhi_div_eq hp hse]
  have hmod : n - p * (n / p) = n % p := by
    have := Nat.div_add_mod n p; omega
  rw [hmod]
  push_cast
  rw [hnv]
  -- (v + bias) mod p = v mod p since p ∣ bias
  have : ((bias : ℕ) : ℤ) = p * (((2 ^ 31 + p - 1) / p : ℕ) : ℤ) := by rw [hbias]; push_cast; ring
  rw [this, Int.add_mul_emod_self_left]

/-- **The CRT digit (DESIGN §3.6).** `y = (q·W) mod p` by the same `umulhi64(⌈2⁶⁴/p⌉)`
reduction, for `q, W < p ≤ 256`. -/
theorem crt_y_reduction {p qd w : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) (hq : qd < p) (hw : w < p) :
    qd * w - p * (qd * w * ((2 ^ 64 + p - 1) / p) / 2 ^ 64) = qd * w % p := by
  have he : (2 ^ 64 + p - 1) / p * p - 2 ^ 64 < p := by
    have := Nat.div_mul_le_self (2 ^ 64 + p - 1) p
    omega
  have hqw : qd * w < 2 ^ 16 := by
    have : qd * w < 256 * 256 :=
      lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le (by omega) (by omega) (by omega)) le_rfl
    omega
  have hse : qd * w * ((2 ^ 64 + p - 1) / p * p - 2 ^ 64) < 2 ^ 64 := by
    calc qd * w * ((2 ^ 64 + p - 1) / p * p - 2 ^ 64) ≤ qd * w * 256 :=
          Nat.mul_le_mul_left _ (by omega)
      _ < 2 ^ 16 * 256 := by omega
      _ ≤ 2 ^ 64 := by norm_num
  rw [mulhi_div_eq hp hse]
  have := Nat.div_add_mod (qd * w) p
  omega

/-! ### Capstone with the device's residues -/

open Immerdink.CRT Immerdink.Chunk in
/-- **Exactness with SMod residues.** `Exact.emulated_product_exact` where the per-modulus
residues are the ones DESIGN §3.4 stores, `r_j = SMod(a', p_j)` and `s_j = SMod(b', p_j)`,
with `p_j ≤ 256` and chunks of at most `K` products with `128²·K ≤ A` (int32: `A = 2³¹ − 1`).
The residue hypotheses of the capstone are discharged by `smod_modEq` and
`smod_mul_abs_le`. -/
theorem emulated_product_exact_smod
    {N : ℕ} {p : ℕ → ℕ} (hp : ∀ l < N, 0 < p l) (hp256 : ∀ l < N, p l ≤ 256)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    {q : ℕ → ℕ} (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j])
    {H : ℕ} (hH : 2 * H + 1 ≤ P N p)
    {k T : ℕ} {a' b' : ℕ → ℤ} (hT : T = Nat.sqrt H)
    (ha : ∑ h ∈ range k, a' h ^ 2 ≤ (T : ℤ) ^ 2) (hb : ∑ h ∈ range k, b' h ^ 2 ≤ (T : ℤ) ^ 2)
    {K A : ℤ} (hQK : 128 ^ 2 * K ≤ A)
    {n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ} (hK : ∀ c < n, ((C c).card : ℤ) ≤ K)
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {ord : ℕ → ℕ → List ℕ} (hord : ∀ j < N, ∀ c < n, Enumerates (ord j c) (C c)) :
    let W : ℕ → ℕ := fun j =>
      (epi (p j) (fun c => satFold A (fun h => smod (a' h) (p j) * smod (b' h) (p j)) 0
        (ord j c)) (n - 1)).toNat
    z N p q W H = ∑ h ∈ range k, a' h * b' h :=
  Exact.emulated_product_exact hp hcop hq hH hT ha hb
    (r := fun j h => smod (a' h) (p j)) (s := fun j h => smod (b' h) (p j))
    (fun _ _ _ => smod_modEq _ _) (fun _ _ _ => smod_modEq _ _)
    (fun j hj _ => smod_mul_abs_le _ _ (hp j hj) (hp256 j hj) (hp j hj) (hp256 j hj))
    hQK hn hK hdisj hcover hord

end Immerdink.Residue
