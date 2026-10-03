import Mathlib

/-!
# L2. Division-free running CRT (PROOFS.md §L2, DESIGN.md §3.6)

Setting (indices are 0-based here: PROOFS.md's `p_1..p_N` are `p 0, …, p (N-1)`):

* `p : ℕ → ℕ`, the moduli `p 0, …, p (N-1)`, positive and pairwise coprime;
* `P = ∏_{l<N} p_l`, `M_l = ∏_{i<N, i≠l} p_i` (`= P / p_l`, see `M_eq_div`);
* `q l` is any number with `M_l · q_l ≡ 1 (mod p_l)` (one exists: `exists_q`);
* `W l` is the residue delivered for modulus `p l`.

The algorithm, on an unsigned word (modelled in `ℕ`):
```
x_0 = 0
y_l = (q_l·W_l) mod p_l
x'_l = x_l + M_l·y_l                 -- `xp l` (PROOFS.md's x'_{l+1})
x_{l+1} = x'_l − P·[x'_l ≥ P]        -- `x (l+1)`
z = (x_N ≤ H) ? x_N : x_N − P        -- `z`, in ℤ
```
L2a (`x_lt_P`, `xp_lt_two_P`, `xp_lt_word`), L2b (`x_N_modEq`) and L2c (`z_eq`, plus the
even-`P` form `z_eq_of_even` with `H = (P−2)/2` and the general-`P` form `z_eq_floor` with
`H = ⌊(P−1)/2⌋`).  L2c is proved for any `H` with `2H + 1 ≤ P` (`z_eq`).
-/

namespace Immerdink.CRT

open Finset

section
variable (N : ℕ) (p : ℕ → ℕ)

/-- `P = ∏_{l<N} p_l`. -/
def P : ℕ := ∏ l ∈ range N, p l

/-- `M_l = ∏_{i<N, i≠l} p_i`, which equals `P / p_l` (`M_eq_div`). -/
def M (l : ℕ) : ℕ := ∏ i ∈ (range N).erase l, p i

variable (q W : ℕ → ℕ)

/-- `y_l = (q_l·W_l) mod p_l`. -/
def y (l : ℕ) : ℕ := (q l * W l) % p l

/-- One step of the running sum: `x ↦ x + M_l·y_l`, then one conditional subtraction of `P`. -/
def step (l x : ℕ) : ℕ :=
  if P N p ≤ x + M N p l * y p q W l then x + M N p l * y p q W l - P N p
  else x + M N p l * y p q W l

/-- `x_l`: the running CRT word after processing moduli `0, …, l-1`. -/
def x : ℕ → ℕ
  | 0 => 0
  | l + 1 => step N p q W l (x l)

/-- `x'_l = x_l + M_l·y_l`, the word before the conditional subtraction at step `l`. -/
def xp (l : ℕ) : ℕ := x N p q W l + M N p l * y p q W l

/-- The centred output `z = (x_N ≤ H) ? x_N : x_N − P`, as an integer. -/
def z (H : ℕ) : ℤ :=
  if x N p q W N ≤ H then (x N p q W N : ℤ) else (x N p q W N : ℤ) - (P N p : ℤ)

end

variable {N : ℕ} {p : ℕ → ℕ} {q W : ℕ → ℕ}

theorem P_eq_mul_M {l : ℕ} (hl : l < N) : P N p = p l * M N p l :=
  (Finset.mul_prod_erase (range N) p (mem_range.2 hl)).symm

theorem M_eq_div {l : ℕ} (hl : l < N) (hp : 0 < p l) : M N p l = P N p / p l := by
  rw [P_eq_mul_M hl, Nat.mul_div_cancel_left _ hp]

theorem P_pos (hp : ∀ l < N, 0 < p l) : 0 < P N p :=
  Finset.prod_pos fun l hl => hp l (mem_range.1 hl)

theorem M_pos (hp : ∀ l < N, 0 < p l) (l : ℕ) : 0 < M N p l :=
  Finset.prod_pos fun i hi => hp i (mem_range.1 (mem_of_mem_erase hi))

theorem x_succ (l : ℕ) :
    x N p q W (l + 1) = if P N p ≤ xp N p q W l then xp N p q W l - P N p else xp N p q W l :=
  rfl

/-- `M_l·y_l < P`. -/
theorem M_mul_y_lt_P (hp : ∀ l < N, 0 < p l) {l : ℕ} (hl : l < N) :
    M N p l * y p q W l < P N p := by
  rw [P_eq_mul_M hl, mul_comm (p l)]
  exact Nat.mul_lt_mul_of_pos_left (Nat.mod_lt _ (hp l hl)) (M_pos hp l)

/-! ### L2a (bounds) -/

/-- **L2a**, first half: `0 ≤ x_l < P` for every `l ≤ N` (`0 ≤` is automatic in `ℕ`). -/
theorem x_lt_P (hp : ∀ l < N, 0 < p l) : ∀ l ≤ N, x N p q W l < P N p
  | 0, _ => P_pos hp
  | l + 1, hl => by
    have ih := x_lt_P hp l (by omega)
    have hm := M_mul_y_lt_P (q := q) (W := W) hp (l := l) (by omega)
    rw [x_succ]
    unfold xp
    split_ifs <;> omega

/-- **L2a**, second half: `x'_l < 2P` for every step `l < N`. -/
theorem xp_lt_two_P (hp : ∀ l < N, 0 < p l) {l : ℕ} (hl : l < N) :
    xp N p q W l < 2 * P N p := by
  have h1 := x_lt_P (q := q) (W := W) hp l hl.le
  have h2 := M_mul_y_lt_P (q := q) (W := W) hp hl
  unfold xp; omega

/-- **L2a**, word form: an unsigned word of `L_w` bits with `2P ≤ 2^{L_w}` never overflows. -/
theorem xp_lt_word (hp : ∀ l < N, 0 < p l) {Lw : ℕ} (hw : 2 * P N p ≤ 2 ^ Lw) {l : ℕ}
    (hl : l < N) : xp N p q W l < 2 ^ Lw ∧ x N p q W l < 2 ^ Lw :=
  ⟨(xp_lt_two_P hp hl).trans_le hw,
    ((x_lt_P (q := q) (W := W) hp l hl.le).trans_le (by omega)).trans_le hw⟩

/-! ### L2b (congruence) -/

/-- Each conditional subtraction removes a multiple of `P`:
`x_l ≡ Σ_{i<l} M_i·y_i (mod P)`. -/
theorem x_modEq_sum (l : ℕ) :
    x N p q W l ≡ ∑ i ∈ range l, M N p i * y p q W i [MOD P N p] := by
  induction l with
  | zero => simp [x]; rfl
  | succ l ih =>
    rw [x_succ, sum_range_succ]
    have h : xp N p q W l ≡ (∑ i ∈ range l, M N p i * y p q W i) + M N p l * y p q W l
        [MOD P N p] := Nat.ModEq.add_right _ ih
    split_ifs with hP
    · refine Nat.ModEq.trans ?_ h
      have : xp N p q W l = (xp N p q W l - P N p) + P N p := by omega
      calc xp N p q W l - P N p ≡ xp N p q W l - P N p + P N p [MOD P N p] :=
            (Nat.add_mod_right _ _).symm
        _ = xp N p q W l := this.symm
    · exact h

/-- **L2b.** `x_N ≡ W_j (mod p_j)` for every `j < N`, given `M_j·q_j ≡ 1 (mod p_j)`. -/
theorem x_N_modEq (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j]) {j : ℕ} (hj : j < N) :
    x N p q W N ≡ W j [MOD p j] := by
  have hdvdP : p j ∣ P N p := by rw [P_eq_mul_M hj]; exact dvd_mul_right _ _
  refine ((x_modEq_sum N).of_dvd hdvdP).trans ?_
  rw [← Finset.add_sum_erase _ _ (mem_range.2 hj)]
  -- every other term is ≡ 0 (mod p_j), because p_j ∣ M_i for i ≠ j
  have hrest : p j ∣ ∑ i ∈ (range N).erase j, M N p i * y p q W i := by
    refine Finset.dvd_sum fun i hi => Dvd.dvd.mul_right ?_ _
    exact Finset.dvd_prod_of_mem p
      (mem_erase.2 ⟨fun h => (ne_of_mem_erase hi) h.symm, mem_range.2 hj⟩)
  have h0 : ∑ i ∈ (range N).erase j, M N p i * y p q W i ≡ 0 [MOD p j] :=
    (Nat.modEq_zero_iff_dvd).2 hrest
  calc M N p j * y p q W j + ∑ i ∈ (range N).erase j, M N p i * y p q W i
      ≡ M N p j * y p q W j + 0 [MOD p j] := Nat.ModEq.add_left _ h0
    _ = M N p j * ((q j * W j) % p j) := by simp [y]
    _ ≡ M N p j * (q j * W j) [MOD p j] := Nat.ModEq.mul_left _ (Nat.mod_modEq _ _)
    _ = (M N p j * q j) * W j := by ring
    _ ≡ 1 * W j [MOD p j] := Nat.ModEq.mul_right _ (hq j hj)
    _ = W j := one_mul _

/-- Such `q_l` exist when the moduli are pairwise coprime (`q_l = M_l⁻¹ mod p_l`). -/
theorem exists_q (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j)) {l : ℕ} (hl : l < N) : ∃ ql, M N p l * ql ≡ 1 [MOD p l] := by
  have hc : Nat.Coprime (M N p l) (p l) := by
    refine Nat.Coprime.prod_left fun i hi => hcop i (mem_range.1 (mem_of_mem_erase hi)) l hl
      (ne_of_mem_erase hi)
  rcases Nat.lt_or_ge 1 (p l) with h1 | h1
  · obtain ⟨m, hm⟩ := Nat.exists_mul_mod_eq_one_of_coprime hc h1
    exact ⟨m, by rw [Nat.ModEq, hm.2, Nat.mod_eq_of_lt h1]⟩
  · have : p l = 1 := by have := hp l hl; omega
    exact ⟨0, by simp [Nat.ModEq, this]⟩

/-! ### L2c (exact recovery) -/

/-- Chinese remainder step: congruence modulo every `p_j` gives congruence modulo `P`. -/
theorem P_dvd_of_forall (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j)) {d : ℤ}
    (hd : ∀ j < N, (p j : ℤ) ∣ d) : (P N p : ℤ) ∣ d := by
  have : ((∏ l ∈ range N, (p l : ℤ))) ∣ d := by
    refine Finset.prod_dvd_of_coprime ?_ fun j hj => hd j (mem_range.1 hj)
    intro i hi j hj hij
    exact Nat.isCoprime_iff_coprime.2 (hcop i (mem_range.1 hi) j (mem_range.1 hj) hij)
  simpa [P] using this

/-- **L2c (exact recovery), general window.** For any `H` with `2H + 1 ≤ P`: if
`Z ≡ W_j (mod p_j)` for all `j` and `|Z| ≤ H`, then `z = Z`. -/
theorem z_eq (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j])
    {H : ℕ} (hH : 2 * H + 1 ≤ P N p)
    {Z : ℤ} (hZ : ∀ j < N, Z ≡ W j [ZMOD p j]) (hZH : |Z| ≤ H) :
    z N p q W H = Z := by
  -- 1. z ≡ x_N ≡ Z (mod P)
  have hxZ : (P N p : ℤ) ∣ (x N p q W N : ℤ) - Z := by
    refine P_dvd_of_forall hcop fun j hj => ?_
    have h1 : (x N p q W N : ℤ) ≡ W j [ZMOD p j] :=
      Int.natCast_modEq_iff.2 (x_N_modEq hq hj)
    exact ((hZ j hj).trans h1.symm).dvd
  have hzx : (P N p : ℤ) ∣ z N p q W H - (x N p q W N : ℤ) := by
    unfold z; split_ifs <;> simp
  have hdvd : (P N p : ℤ) ∣ z N p q W H - Z := by
    have := dvd_add hzx hxZ; simpa using this
  -- 2. z and Z both lie in the window [H+1−P, H] of P consecutive integers
  have hxlt : (x N p q W N : ℤ) < P N p := by exact_mod_cast x_lt_P (q := q) (W := W) hp N le_rfl
  have hHP : (2 * H + 1 : ℤ) ≤ P N p := by exact_mod_cast hH
  have hzlo : (H : ℤ) + 1 - P N p ≤ z N p q W H := by
    unfold z; split_ifs with h
    · have : (0 : ℤ) ≤ x N p q W N := by positivity
      linarith
    · have : (H : ℤ) < x N p q W N := by exact_mod_cast Nat.lt_of_not_le h
      linarith
  have hzhi : z N p q W H ≤ H := by
    unfold z; split_ifs with h
    · exact_mod_cast h
    · have : (0 : ℤ) ≤ H := by positivity
      linarith
  have hZ1 := (abs_le.1 hZH).1
  have hZ2 := (abs_le.1 hZH).2
  -- 3. two integers in such a window that are congruent mod P are equal
  have habs : |z N p q W H - Z| < P N p := by
    rw [abs_lt]; constructor <;> linarith
  exact sub_eq_zero.1 (Int.eq_zero_of_abs_lt_dvd hdvd habs)

/-- **L2c as stated in PROOFS.md:** `P` even and `H = (P − 2)/2`. -/
theorem z_eq_of_even (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j]) (heven : Even (P N p))
    {Z : ℤ} (hZ : ∀ j < N, Z ≡ W j [ZMOD p j]) (hZH : |Z| ≤ ((P N p - 2) / 2 : ℕ)) :
    z N p q W ((P N p - 2) / 2) = Z := by
  refine z_eq hp hcop hq ?_ hZ hZH
  obtain ⟨r, hr⟩ := heven
  have := P_pos hp
  omega

/-- **L2c for arbitrary `P`** with DESIGN.md's `H = ⌊(P − 1)/2⌋` (equal to `(P−2)/2` when
`P` is even). -/
theorem z_eq_floor (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j])
    {Z : ℤ} (hZ : ∀ j < N, Z ≡ W j [ZMOD p j]) (hZH : |Z| ≤ ((P N p - 1) / 2 : ℕ)) :
    z N p q W ((P N p - 1) / 2) = Z := by
  refine z_eq hp hcop hq ?_ hZ hZH
  have := P_pos hp
  omega

end Immerdink.CRT
