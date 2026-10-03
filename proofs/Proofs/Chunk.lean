import Mathlib

/-!
# L4. K-chunked accumulation with modular carry (PROOFS.md §L4, DESIGN.md §3.5)

Setting:
* integers `r h, s h` (`h < k`) with `|r_h·s_h| ≤ Q`;
* chunks `C c` (`c < n`) partitioning `[0, k)`, with `|C_c| ≤ K`;
* `Q·K ≤ A` (`A = A_max = 2³¹ − 1` in the design);
* a **saturating** accumulator: every addition is clamped to `[−A−1, A]` (`clamp`).

"Any accumulation order" is modelled two ways:
* `satFold` — a sequential fold of clamped additions over any enumeration of the chunk;
* `AccTree.eval` — an arbitrary binary reduction tree of clamped additions (covers any
  association/order an MMA unit or a split-K reduction may use).

Results: L4a (`partial_sum_bound`, `satFold_chunk_eq_sum`, `tree_chunk_eq_sum`), L4b
(`epi_modEq_total`, `epi_mem_range`, and the end-to-end `carry_correct`), L4c
(`epi_input_range`, `design_constants`).
-/

namespace Immerdink.Chunk

open Finset

/-! ### The saturating accumulator -/

/-- Clamp to `[−A−1, A]` (int32 `.satfinite` when `A = 2³¹ − 1`). -/
def clamp (A v : ℤ) : ℤ := max (-A - 1) (min A v)

theorem clamp_of_abs_le {A v : ℤ} (h : |v| ≤ A) : clamp A v = v := by
  have := abs_le.1 h
  unfold clamp
  rw [min_eq_right this.2, max_eq_right (by linarith [this.1])]

/-- Sequential saturating accumulation: `acc ← clamp(acc + f i)` for `i` in `ord`. -/
def satFold (A : ℤ) (f : ℕ → ℤ) (acc : ℤ) (ord : List ℕ) : ℤ :=
  ord.foldl (fun a i => clamp A (a + f i)) acc

/-- Any fold of clamped additions is exact when `|acc₀| + Σ|f i| ≤ A`: every partial sum
`acc₀ + Σ_{prefix} f` then has magnitude `≤ A`, so no clamp ever fires. -/
theorem satFold_eq {A : ℤ} {f : ℕ → ℤ} :
    ∀ (ord : List ℕ) (acc : ℤ), |acc| + (ord.map fun i => |f i|).sum ≤ A →
      satFold A f acc ord = acc + (ord.map f).sum
  | [], acc, _ => by simp [satFold]
  | i :: ord, acc, h => by
    simp only [List.map_cons, List.sum_cons] at h
    have hcl : clamp A (acc + f i) = acc + f i := by
      apply clamp_of_abs_le
      have := abs_add_le acc (f i)
      have : 0 ≤ (ord.map fun i => |f i|).sum :=
        List.sum_nonneg (by simp)
      linarith
    have ih := satFold_eq (A := A) (f := f) ord (acc + f i) (by
      have := abs_add_le acc (f i); linarith)
    simp only [satFold, List.foldl_cons, List.map_cons, List.sum_cons] at ih ⊢
    rw [hcl, ih]; ring

theorem abs_list_sum_le (f : ℕ → ℤ) :
    ∀ l : List ℕ, |(l.map f).sum| ≤ (l.map fun i => |f i|).sum
  | [] => by simp
  | i :: l => by
    simp only [List.map_cons, List.sum_cons]
    have := abs_list_sum_le f l
    have := abs_add_le (f i) (l.map f).sum
    linarith

/-- Arbitrary binary accumulation trees over leaf indices. -/
inductive AccTree
  | leaf : ℕ → AccTree
  | node : AccTree → AccTree → AccTree

namespace AccTree

/-- The leaf indices, left to right. -/
def leaves : AccTree → List ℕ
  | leaf i => [i]
  | node l r => l.leaves ++ r.leaves

/-- Evaluate with a clamped addition at every internal node. -/
def eval (A : ℤ) (f : ℕ → ℤ) : AccTree → ℤ
  | leaf i => f i
  | node l r => clamp A (l.eval A f + r.eval A f)

/-- A reduction tree of clamped additions is exact when `Σ_{leaves}|f| ≤ A`. -/
theorem eval_eq_sum {A : ℤ} {f : ℕ → ℤ} :
    ∀ t : AccTree, (t.leaves.map fun i => |f i|).sum ≤ A → t.eval A f = (t.leaves.map f).sum
  | leaf i, _ => by simp [eval, leaves]
  | node l r, h => by
    simp only [leaves, List.map_append, List.sum_append] at h ⊢
    have hl0 : 0 ≤ (l.leaves.map fun i => |f i|).sum := List.sum_nonneg (by simp)
    have hr0 : 0 ≤ (r.leaves.map fun i => |f i|).sum := List.sum_nonneg (by simp)
    have ihl := eval_eq_sum (A := A) (f := f) l (by linarith)
    have ihr := eval_eq_sum (A := A) (f := f) r (by linarith)
    simp only [eval]
    rw [ihl, ihr]
    apply clamp_of_abs_le
    have h1 := abs_list_sum_le f l.leaves
    have h2 := abs_list_sum_le f r.leaves
    have := abs_add_le (l.leaves.map f).sum (r.leaves.map f).sum
    linarith

end AccTree

/-! ### L4a: no saturation, any order -/

section chunk
variable {r s : ℕ → ℤ} {Q K A : ℤ}

/-- **L4a**, partial sums: for every `S ⊆ C`, `|Σ_{h∈S} r_h s_h| ≤ Q·K`. -/
theorem partial_sum_bound (hQ : ∀ h, |r h * s h| ≤ Q) {C : Finset ℕ} (hC : (C.card : ℤ) ≤ K)
    {S : Finset ℕ} (hS : S ⊆ C) : |∑ h ∈ S, r h * s h| ≤ Q * K := by
  have hQ0 : 0 ≤ Q := (abs_nonneg _).trans (hQ 0)
  calc |∑ h ∈ S, r h * s h| ≤ ∑ h ∈ S, |r h * s h| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _h ∈ S, Q := sum_le_sum fun h _ => hQ h
    _ = S.card * Q := by simp
    _ ≤ C.card * Q := by gcongr
    _ ≤ K * Q := by gcongr
    _ = Q * K := mul_comm _ _

/-- `Σ_{h∈C}|r_h s_h| ≤ Q·K`: the bound on `Σ|products|` that covers every order. -/
theorem sum_abs_bound (hQ : ∀ h, |r h * s h| ≤ Q) {C : Finset ℕ} (hC : (C.card : ℤ) ≤ K) :
    ∑ h ∈ C, |r h * s h| ≤ Q * K := by
  have hQ0 : 0 ≤ Q := (abs_nonneg _).trans (hQ 0)
  calc ∑ h ∈ C, |r h * s h| ≤ ∑ _h ∈ C, Q := sum_le_sum fun h _ => hQ h
    _ = C.card * Q := by simp
    _ ≤ K * Q := by gcongr
    _ = Q * K := mul_comm _ _

/-- A list enumerating the chunk `C` (each element exactly once, any order). -/
def Enumerates (ord : List ℕ) (C : Finset ℕ) : Prop := (ord : Multiset ℕ) = C.val

theorem list_sum_of_enumerates {ord : List ℕ} {C : Finset ℕ} (he : Enumerates ord C)
    (g : ℕ → ℤ) : (ord.map g).sum = ∑ h ∈ C, g h := by
  rw [Finset.sum_eq_multiset_sum, ← he, Multiset.map_coe, Multiset.sum_coe]

/-- **L4a**, sequential order: a saturating fold over any enumeration of the chunk returns the
exact chunk sum `acc_c = Σ_{h∈C} r_h s_h`. -/
theorem satFold_chunk_eq_sum (hQ : ∀ h, |r h * s h| ≤ Q) {C : Finset ℕ}
    (hC : (C.card : ℤ) ≤ K) (hQK : Q * K ≤ A) {ord : List ℕ} (he : Enumerates ord C) :
    satFold A (fun h => r h * s h) 0 ord = ∑ h ∈ C, r h * s h := by
  rw [satFold_eq ord 0, list_sum_of_enumerates he, zero_add]
  rw [abs_zero, zero_add, list_sum_of_enumerates he (fun i => |r i * s i|)]
  exact (sum_abs_bound hQ hC).trans hQK

/-- **L4a**, arbitrary reduction tree: any tree of clamped additions whose leaves enumerate the
chunk returns the exact chunk sum. -/
theorem tree_chunk_eq_sum (hQ : ∀ h, |r h * s h| ≤ Q) {C : Finset ℕ}
    (hC : (C.card : ℤ) ≤ K) (hQK : Q * K ≤ A) {t : AccTree} (he : Enumerates t.leaves C) :
    t.eval A (fun h => r h * s h) = ∑ h ∈ C, r h * s h := by
  rw [AccTree.eval_eq_sum t, list_sum_of_enumerates he]
  rw [list_sum_of_enumerates he (fun i => |r i * s i|)]
  exact (sum_abs_bound hQ hC).trans hQK

end chunk

/-! ### L4b: the modular-carry epilogue -/

/-- The epilogue: `W_0 = acc_0 mod p`, `W_{c+1} = (W_c + acc_{c+1}) mod p`, with the
non-negative mod (`Int.emod`, in `[0, p)` for `p > 0`). -/
def epi (p : ℤ) (acc : ℕ → ℤ) : ℕ → ℤ
  | 0 => acc 0 % p
  | c + 1 => (epi p acc c + acc (c + 1)) % p

theorem epi_modEq (p : ℤ) (acc : ℕ → ℤ) (c : ℕ) :
    epi p acc c ≡ ∑ c' ∈ range (c + 1), acc c' [ZMOD p] := by
  induction c with
  | zero => simp [epi, Int.ModEq]
  | succ c ih =>
    rw [sum_range_succ]
    exact (Int.mod_modEq _ _).trans (Int.ModEq.add_right _ ih)

/-- `W_c ∈ [0, p)`. -/
theorem epi_mem_range {p : ℤ} (hp : 0 < p) (acc : ℕ → ℤ) (c : ℕ) :
    0 ≤ epi p acc c ∧ epi p acc c < p := by
  cases c <;> exact ⟨Int.emod_nonneg _ hp.ne', Int.emod_lt_of_pos _ hp⟩

/-- **L4b.** If the chunks `C 0, …, C (n−1)` partition `[0, k)` and each `acc_c` is the exact
chunk sum, then `W_{n−1} ≡ Σ_{h<k} r_h s_h (mod p)` (and `W_{n−1} ∈ [0, p)` by
`epi_mem_range`). -/
theorem epi_modEq_total {r s : ℕ → ℤ} {k n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ}
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {acc : ℕ → ℤ} (hacc : ∀ c < n, acc c = ∑ h ∈ C c, r h * s h) (p : ℤ) :
    epi p acc (n - 1) ≡ ∑ h ∈ range k, r h * s h [ZMOD p] := by
  refine (epi_modEq p acc (n - 1)).trans ?_
  rw [Nat.sub_add_cancel hn, ← hcover, sum_biUnion hdisj]
  rw [sum_congr rfl fun c hc => hacc c (mem_range.1 hc)]

/-- **L4a + L4b end to end.** Every chunk is accumulated by a saturating fold in an arbitrary
order (`ord c` enumerates `C c`), then carried through the epilogue: the result is the exact
residue class of the full dot product, in `[0, p)`. -/
theorem carry_correct {r s : ℕ → ℤ} {Q K A : ℤ} (hQ : ∀ h, |r h * s h| ≤ Q) (hQK : Q * K ≤ A)
    {k n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ} (hK : ∀ c < n, ((C c).card : ℤ) ≤ K)
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {ord : ℕ → List ℕ} (hord : ∀ c < n, Enumerates (ord c) (C c)) {p : ℤ} (hp : 0 < p) :
    let W := epi p (fun c => satFold A (fun h => r h * s h) 0 (ord c)) (n - 1)
    W ≡ ∑ h ∈ range k, r h * s h [ZMOD p] ∧ 0 ≤ W ∧ W < p :=
  ⟨epi_modEq_total hn hdisj hcover
      (fun c hc => satFold_chunk_eq_sum hQ (hK c hc) hQK (hord c hc)) p,
    epi_mem_range hp _ _⟩

/-! ### L4c: the int32 epilogue range -/

/-- **L4c.** The epilogue input `W_{c−1} + acc_c` lies in `[−Q·K, (p−1) + Q·K]`. -/
theorem epi_input_range {p Q K : ℤ} (hp : 0 < p) {acc : ℕ → ℤ}
    (hacc : ∀ c, |acc c| ≤ Q * K) (c : ℕ) :
    -(Q * K) ≤ epi p acc c + acc (c + 1) ∧ epi p acc c + acc (c + 1) ≤ (p - 1) + Q * K := by
  have h1 := epi_mem_range hp acc c
  have h2 := abs_le.1 (hacc (c + 1))
  constructor <;> linarith

/-- The design constants: `Q = 128²`, `K_c = 131071` is the largest chunk with
`Q·K_c ≤ 2³¹ − 1`, the used chunk `K_c' = 131056` also fits, and for every `p ≤ 256` the
epilogue range `[−Q·K_c', (p−1) + Q·K_c']` lies inside int32 (max `2147221759`). -/
theorem design_constants :
    (128 ^ 2 : ℤ) * 131071 ≤ 2 ^ 31 - 1 ∧ (128 ^ 2 : ℤ) * 131072 > 2 ^ 31 - 1 ∧
    (128 ^ 2 : ℤ) * 131056 ≤ 2 ^ 31 - 1 ∧
    (∀ p : ℤ, p ≤ 256 → (p - 1) + 128 ^ 2 * 131056 ≤ 2 ^ 31 - 1) ∧
    (255 + 128 ^ 2 * 131056 : ℤ) = 2147221759 ∧
    -(2 ^ 31 : ℤ) ≤ -(128 ^ 2 * 131056) := by
  refine ⟨by norm_num, by norm_num, by norm_num, fun p hp => by linarith, by norm_num,
    by norm_num⟩

/-- **L4c instantiated.** With `|r_h s_h| ≤ 128²`, chunks of length `≤ 131056` and
`0 < p ≤ 256`, every epilogue input fits int32. -/
theorem epi_input_int32 {p : ℤ} (hp : 0 < p) (hp256 : p ≤ 256) {acc : ℕ → ℤ}
    (hacc : ∀ c, |acc c| ≤ 128 ^ 2 * 131056) (c : ℕ) :
    -(2 ^ 31 : ℤ) ≤ epi p acc c + acc (c + 1) ∧ epi p acc c + acc (c + 1) ≤ 2 ^ 31 - 1 := by
  have := epi_input_range hp hacc c
  constructor <;> linarith

end Immerdink.Chunk
