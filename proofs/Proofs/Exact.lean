import Proofs.CRT
import Proofs.Chunk
import Proofs.Scaling

/-!
# Capstone: L1c + L4 + L2c ⇒ the emulated integer product is exact

PROOFS.md: "Combining L1c and L2c with `Z = Σ a'b'` gives that the emulated product is exact."
This file chains the three machine-checked pieces:

* L1c: the quantized integer vectors satisfy `‖a'‖₂, ‖b'‖₂ ≤ T = ⌊√H⌋`, so `|Σ a'b'| ≤ H`;
* L4: for every modulus `p_j`, residues `r_j ≡ a'`, `s_j ≡ b'` (mod `p_j`) with
  `|r_j s_j| ≤ Q`, accumulated per chunk by a saturating fold in an arbitrary order and carried
  through the epilogue, give `W_j ≡ Σ a'b' (mod p_j)` with `W_j ∈ [0, p_j)`;
* L2c: the running CRT on `W_0, …, W_{N−1}` returns exactly `Σ a'b'`.
-/

namespace Immerdink.Exact

open Finset Immerdink.CRT Immerdink.Chunk

theorem emulated_product_exact
    -- moduli and CRT constants (L2)
    {N : ℕ} {p : ℕ → ℕ} (hp : ∀ l < N, 0 < p l)
    (hcop : ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (p i) (p j))
    {q : ℕ → ℕ} (hq : ∀ j < N, M N p j * q j ≡ 1 [MOD p j])
    {H : ℕ} (hH : 2 * H + 1 ≤ P N p)
    -- quantized operands with the L1 norm budget
    {k T : ℕ} {a' b' : ℕ → ℤ} (hT : T = Nat.sqrt H)
    (ha : ∑ h ∈ range k, a' h ^ 2 ≤ (T : ℤ) ^ 2) (hb : ∑ h ∈ range k, b' h ^ 2 ≤ (T : ℤ) ^ 2)
    -- per-modulus residues and chunked saturating accumulation (L4)
    {r s : ℕ → ℕ → ℤ} (hr : ∀ j < N, ∀ h, r j h ≡ a' h [ZMOD p j])
    (hs : ∀ j < N, ∀ h, s j h ≡ b' h [ZMOD p j])
    {Q K A : ℤ} (hQ : ∀ j < N, ∀ h, |r j h * s j h| ≤ Q) (hQK : Q * K ≤ A)
    {n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ} (hK : ∀ c < n, ((C c).card : ℤ) ≤ K)
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {ord : ℕ → ℕ → List ℕ} (hord : ∀ j < N, ∀ c < n, Enumerates (ord j c) (C c)) :
    let W : ℕ → ℕ := fun j =>
      (epi (p j) (fun c => satFold A (fun h => r j h * s j h) 0 (ord j c)) (n - 1)).toNat
    z N p q W H = ∑ h ∈ range k, a' h * b' h := by
  intro W
  refine z_eq hp hcop hq hH (fun j hj => ?_) ?_
  · -- L4: W_j ≡ Σ r s ≡ Σ a'b' (mod p_j)
    have hpj : (0 : ℤ) < p j := by exact_mod_cast hp j hj
    obtain ⟨hc, h0, -⟩ := carry_correct (hQ j hj) hQK hn hK hdisj hcover (hord j hj) hpj
    have hW : ((W j : ℕ) : ℤ) = epi (p j)
        (fun c => satFold A (fun h => r j h * s j h) 0 (ord j c)) (n - 1) :=
      Int.toNat_of_nonneg h0
    rw [hW]
    refine Int.ModEq.trans ?_ hc.symm
    rw [Int.modEq_iff_dvd]
    rw [← sum_sub_distrib]
    refine dvd_sum fun h _ => ?_
    have e : r j h * s j h - a' h * b' h = r j h * (s j h - b' h) + b' h * (r j h - a' h) := by
      ring
    rw [e]
    exact dvd_add (dvd_mul_of_dvd_right ((hs j hj h).symm.dvd) _)
      (dvd_mul_of_dvd_right ((hr j hj h).symm.dvd) _)
  · -- L1c: |Σ a'b'| ≤ H
    have := (Scaling.crt_precondition (a' := fun h : Fin k => a' h) (b' := fun h : Fin k => b' h)
      hT (by rw [Fin.sum_univ_eq_sum_range (fun h => a' h ^ 2)]; exact ha)
      (by rw [Fin.sum_univ_eq_sum_range (fun h => b' h ^ 2)]; exact hb)).2.2
    rwa [Fin.sum_univ_eq_sum_range (fun h => a' h * b' h)] at this

end Immerdink.Exact
