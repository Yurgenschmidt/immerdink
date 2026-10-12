import Proofs.CRT

/-!
# The running CRT on an unsigned word (DESIGN.md §2 `L_w`, §3.6)

`CRT.lean` proves L2 with the word modelled as one natural number. This file adds the facts
DESIGN §3.6 states about the realization, which `spec/params.py` (`cert_crt_peak`) and
`spec/CRTRealization.tla` check at finitely many sizes:

* **the exact peak.** Before step `l`, `x_l` is a multiple of `R_l = ∏_{j≥l} p_j` below `P`
  (`R_dvd_x`), so `x_l ≤ P − R_l` and `x'_l ≤ (P − R_l) + M_l(p_l − 1)` (`xp_le_peak`,
  `xp_le_crtPeak`). This is `params.crt_peak_exact`, and it is attained: for every `l` there
  are residues `W_j ∈ [0, p_j)` that reach it (`peak_attained`, `crtPeak_attained`);
* **`≥` and `>` agree.** `x_l + M_l·y_l = P` never occurs (`xp_ne_P`), so the conditional
  subtraction may compare either way (`le_xp_iff_lt`);
* **no wrap.** With `2P ≤ 2^{L_w}` every word value is exact modulo `2^{L_w}`
  (`xp_mod_word`);
* **centring, in both realizations:**
  - the two's-complement word form of DESIGN §3.6 and `spec/CRTRealization.tla`
    (`center_word`, `z_word`): `z = (x ≤ H) ? x : x − P` computed on the unsigned `L_w`-bit word
    (the subtraction wraps) and read back as a signed word is `z`; its sign bit is set exactly
    when the subtraction is taken, which is when its borrow-out is 1 (`x < P`), as the TLA+
    check `SignBit(x2) = borrow` requires. Borrows themselves are not modelled;
  - the sign-magnitude form the device computes (`crt_finish` in `src/kernels/numerics.cuh`:
    `neg = x > H`, `|z| = neg ? P − x : x`) is `z` (`center_signmag`, `z_signmag`).

The limb-by-limb carry and borrow chains stay in TLA+ (`CRTRealization` `RefinesWord`).
-/

namespace Immerdink.CRT

open Finset

section
variable (N : ℕ) (p : ℕ → ℕ)

/-- `R_l = ∏_{l ≤ j < N} p_j`: every CRT weight `M_j` with `j < l` is a multiple of it. -/
def R (l : ℕ) : ℕ := ∏ j ∈ Ico l N, p j

/-- `crt_peak_exact`: `max_{l<N} (P − R_l) + M_l·(p_l − 1)`. -/
def crtPeak : ℕ := (range N).sup fun l => P N p - R N p l + M N p l * (p l - 1)

end

variable {N : ℕ} {p : ℕ → ℕ} {q W : ℕ → ℕ}

theorem R_succ {l : ℕ} (hl : l < N) : R N p l = p l * R N p (l + 1) :=
  Finset.prod_eq_prod_Ico_succ_bot hl p

theorem R_dvd_P (l : ℕ) : R N p l ∣ P N p := by
  unfold R P
  refine Finset.prod_dvd_prod_of_subset _ _ _ fun j hj => ?_
  simp only [mem_Ico, mem_range] at hj ⊢; omega

theorem R_succ_dvd_M (l : ℕ) : R N p (l + 1) ∣ M N p l := by
  unfold R M
  refine Finset.prod_dvd_prod_of_subset _ _ _ fun j hj => ?_
  simp only [mem_Ico, mem_erase, mem_range] at hj ⊢; omega

theorem R_pos (hp : ∀ l < N, 0 < p l) (l : ℕ) : 0 < R N p l :=
  Finset.prod_pos fun j hj => hp j (by simp only [mem_Ico] at hj; omega)

/-- `R_l ∣ x_l`: the running word before step `l` is a multiple of `∏_{j≥l} p_j`. -/
theorem R_dvd_x : ∀ l ≤ N, R N p l ∣ x N p q W l
  | 0, _ => by simp [x]
  | l + 1, hl => by
    have ih := R_dvd_x l (by omega)
    have h1 : R N p (l + 1) ∣ x N p q W l := by
      rw [R_succ (by omega)] at ih; exact (dvd_mul_left _ _).trans ih
    have h2 : R N p (l + 1) ∣ xp N p q W l :=
      dvd_add h1 ((R_succ_dvd_M l).mul_right _)
    rw [x_succ]
    split_ifs
    · exact Nat.dvd_sub h2 (R_dvd_P _)
    · exact h2

/-- `x_l ≤ P − R_l`. -/
theorem x_le_P_sub_R (hp : ∀ l < N, 0 < p l) {l : ℕ} (hl : l ≤ N) :
    x N p q W l ≤ P N p - R N p l := by
  obtain ⟨t, ht⟩ := R_dvd_x (q := q) (W := W) l hl
  obtain ⟨t', ht'⟩ := R_dvd_P (N := N) (p := p) l
  have hlt := x_lt_P (q := q) (W := W) hp l hl
  have hR := R_pos hp l
  rw [ht, ht'] at hlt
  have htt : t < t' := Nat.lt_of_mul_lt_mul_left hlt
  have : R N p l * (t + 1) ≤ R N p l * t' := Nat.mul_le_mul_left _ htt
  rw [ht, ht']
  rw [mul_add, mul_one] at this
  omega

/-- **The exact peak bound.** `x'_l ≤ (P − R_l) + M_l·(p_l − 1)`. -/
theorem xp_le_peak (hp : ∀ l < N, 0 < p l) {l : ℕ} (hl : l < N) :
    xp N p q W l ≤ P N p - R N p l + M N p l * (p l - 1) := by
  have h1 := x_le_P_sub_R (q := q) (W := W) hp hl.le
  have h2 : y p q W l ≤ p l - 1 := by
    have := Nat.mod_lt (q l * W l) (hp l hl); unfold y; omega
  have h3 := Nat.mul_le_mul_left (M N p l) h2
  unfold xp; omega

theorem xp_le_crtPeak (hp : ∀ l < N, 0 < p l) {l : ℕ} (hl : l < N) :
    xp N p q W l ≤ crtPeak N p :=
  (xp_le_peak hp hl).trans
    (Finset.le_sup (f := fun l => P N p - R N p l + M N p l * (p l - 1)) (mem_range.2 hl))

/-! ### `≥` and `>` agree -/

theorem coprime_M (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j)) {l : ℕ}
    (hl : l < N) : Nat.Coprime (M N p l) (p l) :=
  Nat.Coprime.prod_left fun i hi => hcop i (mem_range.1 (mem_of_mem_erase hi)) l hl
    (ne_of_mem_erase hi)

/-- `x_l + M_l·y_l = P` never occurs (DESIGN §3.6). -/
theorem xp_ne_P (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j)) {l : ℕ} (hl : l < N) :
    xp N p q W l ≠ P N p := by
  intro h
  have hpl_R : p l ∣ R N p l := by rw [R_succ hl]; exact dvd_mul_right _ _
  have hpl_x : p l ∣ x N p q W l := hpl_R.trans (R_dvd_x l hl.le)
  have hpl_P : p l ∣ P N p := by rw [P_eq_mul_M hl]; exact dvd_mul_right _ _
  have hMy : M N p l * y p q W l = P N p - x N p q W l := by unfold xp at h; omega
  have hdvd : p l ∣ M N p l * y p q W l := by rw [hMy]; exact Nat.dvd_sub hpl_P hpl_x
  have hy : p l ∣ y p q W l := (coprime_M hcop hl).symm.dvd_of_dvd_mul_left hdvd
  have hy0 : y p q W l = 0 :=
    Nat.eq_zero_of_dvd_of_lt hy (Nat.mod_lt _ (hp l hl))
  have := x_lt_P (q := q) (W := W) hp l hl.le
  unfold xp at h; rw [hy0, mul_zero, add_zero] at h; omega

/-- The conditional subtraction can test `x' ≥ P` or `x' > P`: they agree at every step. -/
theorem le_xp_iff_lt (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j)) {l : ℕ} (hl : l < N) :
    P N p ≤ xp N p q W l ↔ P N p < xp N p q W l :=
  ⟨fun h => lt_of_le_of_ne h (xp_ne_P hp hcop hl).symm, le_of_lt⟩

/-! ### No wrap on the `L_w`-bit word -/

/-- With `2P ≤ 2^{L_w}`, `x'_l` and `x_l` are their own values modulo `2^{L_w}`: an unsigned
word that wraps would compute the same thing. -/
theorem xp_mod_word (hp : ∀ l < N, 0 < p l) {Lw : ℕ} (hw : 2 * P N p ≤ 2 ^ Lw) {l : ℕ}
    (hl : l < N) :
    xp N p q W l % 2 ^ Lw = xp N p q W l ∧ x N p q W l % 2 ^ Lw = x N p q W l :=
  ⟨Nat.mod_eq_of_lt (xp_lt_word hp hw hl).1, Nat.mod_eq_of_lt (xp_lt_word hp hw hl).2⟩

/-! ### Centring on the word -/

/-- The signed (two's complement) value of an `L`-bit word `w < 2^L`. -/
def toSigned (L w : ℕ) : ℤ := if w < 2 ^ (L - 1) then w else (w : ℤ) - 2 ^ L

/-- The centring step on an unsigned `L`-bit word: `x` if `x ≤ H`, else `x − P` wrapped
modulo `2^L`. -/
def centerWord (L Pw H x : ℕ) : ℕ := if x ≤ H then x else (x + 2 ^ L - Pw) % 2 ^ L

/-- **Centring on the word** (DESIGN §3.6, `CRTRealization`'s word form). With `1 ≤ L`,
`2P ≤ 2^L` and `x < P` (any `H`): the word is `< 2^L`, its sign bit (`≥ 2^{L−1}`) is set exactly
when the subtraction is taken (`H < x`; that subtraction's borrow-out is then 1, since `x < P`),
and read as two's complement it is the centred value. -/
theorem center_word {L Pw H x : ℕ} (hL : 1 ≤ L) (hw : 2 * Pw ≤ 2 ^ L) (hx : x < Pw) :
    centerWord L Pw H x < 2 ^ L ∧
      (2 ^ (L - 1) ≤ centerWord L Pw H x ↔ H < x) ∧
      toSigned L (centerWord L Pw H x) = if x ≤ H then (x : ℤ) else (x : ℤ) - Pw := by
  unfold centerWord toSigned
  have h2 : 2 ^ L = 2 * 2 ^ (L - 1) := by
    rw [← pow_succ']; congr 1; omega
  generalize 2 ^ (L - 1) = A at h2 ⊢
  by_cases hxH : x ≤ H
  · simp only [hxH, ↓reduceIte]
    have : x < A := by omega
    refine ⟨by omega, by omega, ?_⟩
    simp [this]
  · simp only [hxH, ↓reduceIte]
    have hlt : x + 2 ^ L - Pw < 2 ^ L := by omega
    rw [Nat.mod_eq_of_lt hlt]
    have hge : A ≤ x + 2 ^ L - Pw := by omega
    refine ⟨hlt, by omega, ?_⟩
    have hnot : ¬ x + 2 ^ L - Pw < A := by omega
    simp only [hnot, ↓reduceIte]
    have : ((x + 2 ^ L - Pw : ℕ) : ℤ) = (x : ℤ) + 2 ^ L - Pw := by
      have : Pw ≤ x + 2 ^ L := by omega
      push_cast [this]; ring
    rw [this]; ring

/-- **DESIGN §3.6's centring, on the word.** The running CRT's final word, centred on an
unsigned `L_w`-bit word and read back as two's complement, is `z`. -/
theorem z_word (hp : ∀ l < N, 0 < p l) {Lw : ℕ} (hL : 1 ≤ Lw) (hw : 2 * P N p ≤ 2 ^ Lw)
    (H : ℕ) :
    toSigned Lw (centerWord Lw (P N p) H (x N p q W N)) = z N p q W H := by
  rw [(center_word (H := H) hL hw (x_lt_P hp N le_rfl)).2.2]
  unfold z; split_ifs <;> rfl

/-- **Centring as the device computes it** (`crt_finish`): sign `neg = (x > H)` and magnitude
`P − x` or `x`; with `x < P` the signed value is the centred one. -/
theorem center_signmag {Pw H x : ℕ} (hx : x < Pw) :
    (if H < x then -(((Pw - x : ℕ) : ℤ)) else ((x : ℕ) : ℤ)) =
      if x ≤ H then (x : ℤ) else (x : ℤ) - Pw := by
  by_cases h : x ≤ H
  · have h' : ¬ H < x := by omega
    simp only [h, h', ↓reduceIte]
  · have h' : H < x := by omega
    simp only [h, h', ↓reduceIte]
    push_cast [hx.le]; ring

/-- The running CRT's final word, centred in the device's sign-magnitude form, is `z`. -/
theorem z_signmag (hp : ∀ l < N, 0 < p l) (H : ℕ) :
    (if H < x N p q W N then -(((P N p - x N p q W N : ℕ) : ℤ))
      else ((x N p q W N : ℕ) : ℤ)) = z N p q W H := by
  rw [center_signmag (x_lt_P hp N le_rfl)]
  unfold z; split_ifs <;> rfl

/-! ### The peak is attained -/

/-- The running sum over a prefix: for `i < l ≤ N`, `Σ_{j<l} M_j·y_j ≡ W_i (mod p_i)`. -/
theorem prefix_sum_modEq (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j]) {l i : ℕ} (hlN : l ≤ N)
    (hi : i < l) : ∑ j ∈ range l, M N p j * y p q W j ≡ W i [MOD p i] := by
  have hiN : i < N := by omega
  rw [← Finset.add_sum_erase _ _ (mem_range.2 hi)]
  have hrest : p i ∣ ∑ j ∈ (range l).erase i, M N p j * y p q W j := by
    refine Finset.dvd_sum fun j hj => Dvd.dvd.mul_right ?_ _
    have hjl := mem_range.1 (mem_of_mem_erase hj)
    exact Finset.dvd_prod_of_mem p
      (mem_erase.2 ⟨fun h => (ne_of_mem_erase hj) h.symm, mem_range.2 hiN⟩)
  have h0 : ∑ j ∈ (range l).erase i, M N p j * y p q W j ≡ 0 [MOD p i] :=
    (Nat.modEq_zero_iff_dvd).2 hrest
  calc M N p i * y p q W i + ∑ j ∈ (range l).erase i, M N p j * y p q W j
      ≡ M N p i * y p q W i + 0 [MOD p i] := Nat.ModEq.add_left _ h0
    _ = M N p i * ((q i * W i) % p i) := by simp [y]
    _ ≡ M N p i * (q i * W i) [MOD p i] := Nat.ModEq.mul_left _ (Nat.mod_modEq _ _)
    _ = (M N p i * q i) * W i := by ring
    _ ≡ 1 * W i [MOD p i] := Nat.ModEq.mul_right _ (hq i hiN)
    _ = W i := one_mul _

/-- For `l ≤ i < N`, every `M_j` with `j < l` is a multiple of `p_i`. -/
theorem prefix_sum_dvd {l i : ℕ} (hli : l ≤ i) (hiN : i < N) :
    p i ∣ ∑ j ∈ range l, M N p j * y p q W j :=
  Finset.dvd_sum fun j hj => Dvd.dvd.mul_right
    (Finset.dvd_prod_of_mem p (mem_erase.2 ⟨by have := mem_range.1 hj; omega, mem_range.2 hiN⟩)) _

/-- **The peak is reachable** (`params.crt_peak_exact` is exact, not only a bound): for every
step `l < N` there are residues `W_j ∈ [0, p_j)` with `x'_l = (P − R_l) + M_l·(p_l − 1)`. -/
theorem peak_attained (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j]) {l : ℕ} (hl : l < N) :
    ∃ W : ℕ → ℕ, (∀ j < N, W j < p j) ∧
      xp N p q W l = P N p - R N p l + M N p l * (p l - 1) := by
  set t := P N p - R N p l with ht
  let W : ℕ → ℕ := fun j =>
    if j < l then t % p j else if j = l then (p l - 1) * M N p l % p l else 0
  refine ⟨W, fun j hj => ?_, ?_⟩
  · simp only [W]; split_ifs with h1 h2
    · exact Nat.mod_lt _ (hp j hj)
    · subst h2; exact Nat.mod_lt _ (hp _ hj)
    · exact hp j hj
  -- 1. x_l = t, by CRT: both are < P and congruent modulo every p_i
  have hRP := R_dvd_P (N := N) (p := p) l
  have hRpos := R_pos hp l
  have hPpos := P_pos hp
  have hRle : R N p l ≤ P N p := Nat.le_of_dvd hPpos hRP
  have hx : x N p q W l = t := by
    have hcong : ∀ i < N, (x N p q W l : ℤ) ≡ t [ZMOD p i] := by
      intro i hi
      have hxs : x N p q W l ≡ ∑ j ∈ range l, M N p j * y p q W j [MOD p i] :=
        (x_modEq_sum l).of_dvd (by rw [P_eq_mul_M hi]; exact dvd_mul_right _ _)
      refine Int.natCast_modEq_iff.2 (hxs.trans ?_)
      rcases Nat.lt_or_ge i l with hil | hil
      · refine (prefix_sum_modEq hq hl.le hil).trans ?_
        simp only [W, hil, ↓reduceIte]; exact Nat.mod_modEq _ _
      · have h1 : ∑ j ∈ range l, M N p j * y p q W j ≡ 0 [MOD p i] :=
          (Nat.modEq_zero_iff_dvd).2 (prefix_sum_dvd hil hi)
        have hpiR : p i ∣ R N p l :=
          Finset.dvd_prod_of_mem p (mem_Ico.2 ⟨hil, hi⟩)
        have hpiP : p i ∣ P N p := hpiR.trans hRP
        have h2 : t ≡ 0 [MOD p i] := (Nat.modEq_zero_iff_dvd).2 (Nat.dvd_sub hpiP hpiR)
        exact h1.trans h2.symm
    have hdvd : (P N p : ℤ) ∣ (x N p q W l : ℤ) - t :=
      P_dvd_of_forall hcop fun i hi => (hcong i hi).symm.dvd
    have hxlt : (x N p q W l : ℤ) < P N p := by exact_mod_cast x_lt_P hp l hl.le
    have htlt : (t : ℤ) < P N p := by
      have : t < P N p := by omega
      exact_mod_cast this
    have habs : |(x N p q W l : ℤ) - t| < P N p := by
      rw [abs_lt]; constructor <;> omega
    have := Int.eq_zero_of_abs_lt_dvd hdvd habs
    omega
  -- 2. y_l = p_l − 1
  have hy : y p q W l = p l - 1 := by
    have hpl := hp l hl
    have e : y p q W l = (q l * ((p l - 1) * M N p l % p l)) % p l := by
      simp only [y, W, lt_irrefl, ↓reduceIte]
    have hc : q l * ((p l - 1) * M N p l % p l) ≡ p l - 1 [MOD p l] :=
      calc q l * ((p l - 1) * M N p l % p l) ≡ q l * ((p l - 1) * M N p l) [MOD p l] :=
            Nat.ModEq.mul_left _ (Nat.mod_modEq _ _)
        _ = (p l - 1) * (M N p l * q l) := by ring
        _ ≡ (p l - 1) * 1 [MOD p l] := Nat.ModEq.mul_left _ (hq l hl)
        _ = p l - 1 := mul_one _
    rw [e, hc]
    exact Nat.mod_eq_of_lt (by omega)
  unfold xp; rw [hx, hy]

/-- **`crt_peak_exact` is the exact maximum** of `x'_l` over every step and every residue
input (`xp_le_crtPeak` is the bound). -/
theorem crtPeak_attained (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j]) (hN : 0 < N) :
    ∃ l < N, ∃ W : ℕ → ℕ, (∀ j < N, W j < p j) ∧ xp N p q W l = crtPeak N p := by
  obtain ⟨l, hl, hsup⟩ := Finset.exists_mem_eq_sup (range N) (nonempty_range_iff.2 hN.ne')
    (fun l => P N p - R N p l + M N p l * (p l - 1))
  obtain ⟨W, hW, hxp⟩ := peak_attained hp hcop hq (mem_range.1 hl)
  exact ⟨l, mem_range.1 hl, W, hW, by rw [hxp, crtPeak, hsup]⟩

end Immerdink.CRT
