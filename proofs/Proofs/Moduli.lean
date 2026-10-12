import Proofs.Residue
import Proofs.CRTWord

/-!
# The design instance: the 20 moduli and the real-width constants (DESIGN.md §2, §3.5, §3.6)

The parametric lemmas need a few facts about the shipped constants, which `spec/params.py`
certifies (`cert_moduli`, `cert_pht`, `cert_limbs`, `cert_crt_peak`, `cert_kc`). Here they
are proved by kernel evaluation (`decide` only, never the compiled evaluator):

* `moduli`: `256, 255, …, 173` (DESIGN §2), all in `[1, 256]` (`pm_pos`, `pm_le_256`) and
  pairwise coprime (`pm_coprime`); `P` is even (`P_even`); `T = ⌊√H⌋ ≥ 180` (`T_ge_180`);
* the CRT word: `L32` is the least number of 32-bit limbs with `2P ≤ 2^{32·L32}`
  (`L32_exact`), and the exact reachable peak (`CRT.crtPeak`, attained by
  `CRT.crtPeak_attained`) needs every limb (`peak_needs_every_limb`);
* the chunk length: `K_c = 131071` is tight, `K_c' = 131008 = 2047·64` is the largest multiple
  of 64 below it (`chunk_constants`), and the epilogue input lies in DESIGN §3.5's range
  `[−128·127·K_c', (p−1) + 128²·K_c']` inside int32 (`epilogue_range_v113`; a sound enclosure for
  every `p ≤ 256`, tight at `p = 256`; odd `p` have the narrower `±⌊p/2⌋²·K_c'`);
* the CRT multipliers `qd` (`qd_inv`: `M_l·q_l ≡ 1 (mod p_l)` for every `N ≤ 20`), the epilogue's
  Barrett reduction on every reachable input (`barrett_epilogue_v113`), and the end-to-end
  capstone at the real moduli, `emulated_product_exact_design`, and with the shipped
  multipliers, `emulated_product_exact_shipped`.

`tools/check_proofs.sh` checks that `moduli` matches `spec/params_out.json`.
-/

namespace Immerdink.Moduli

open Finset Immerdink.CRT Immerdink.Chunk Immerdink.Residue

/-- DESIGN §2's moduli, greedy pairwise coprime from 256 downward. -/
def moduli : List ℕ :=
  [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181,
    179, 173]

/-- `p_l` (0-based); `1` past the table. -/
def pm (l : ℕ) : ℕ := moduli.getD l 1

theorem moduli_length : moduli.length = 20 := rfl

theorem pm_pos (l : ℕ) : 0 < pm l := by
  unfold pm
  rcases Nat.lt_or_ge l 20 with h | h
  · interval_cases l <;> decide
  · rw [List.getD_eq_default _ _ (by simp [moduli]; omega)]; decide

theorem pm_le_256 (l : ℕ) : pm l ≤ 256 := by
  unfold pm
  rcases Nat.lt_or_ge l 20 with h | h
  · interval_cases l <;> decide
  · rw [List.getD_eq_default _ _ (by simp [moduli]; omega)]; decide

/-- The 20 moduli are pairwise coprime (`cert_moduli`). -/
theorem pm_coprime : ∀ i < 20, ∀ j < 20, i ≠ j → Nat.Coprime (pm i) (pm j) := by decide

theorem pm_coprime_N {N : ℕ} (hN : N ≤ 20) :
    ∀ i < N, ∀ j < N, i ≠ j → Nat.Coprime (pm i) (pm j) :=
  fun i hi j hj hij => pm_coprime i (by omega) j (by omega) hij

/-- `P` is even for every `N ≥ 1` (`p_1 = 256`). -/
theorem P_even {N : ℕ} (hN : 1 ≤ N) : 2 ∣ P N pm :=
  (by decide : 2 ∣ pm 0).trans (Finset.dvd_prod_of_mem pm (mem_range.2 hN))

/-- The `M_l·q_l ≡ 1 (mod p_l)` multipliers exist at every `N ≤ 20`. -/
theorem exists_q_design {N : ℕ} (hN : N ≤ 20) {l : ℕ} (hl : l < N) :
    ∃ ql, M N pm l * ql ≡ 1 [MOD pm l] :=
  exists_q (fun l _ => pm_pos l) (pm_coprime_N hN) hl

/-- `T = ⌊√H⌋ ≥ 180` for every `N ≥ 2`, with `H = ⌊(P − 1)/2⌋` (DESIGN §3.2: at real sizes the
κ ≥ T policy's `k ≥ 2` clause is implied). -/
theorem T_ge_180 : ∀ N < 21, 2 ≤ N → 180 ≤ Nat.sqrt ((P N pm - 1) / 2) := by
  intro N hN h2
  refine Nat.le_sqrt.2 ?_
  revert h2
  revert N
  decide

/-! ### The CRT word -/

/-- DESIGN §2's limb count: 1 for `N ≤ 3`, 2 for `N ≤ 7`, 3 for `N ≤ 12`, 4 for `N ≤ 16`,
5 for `N ≤ 20`. -/
def L32 (N : ℕ) : ℕ :=
  if N ≤ 3 then 1 else if N ≤ 7 then 2 else if N ≤ 12 then 3 else if N ≤ 16 then 4 else 5

/-- **`L_w = 32·L32` is the least whole-limb width with `2P ≤ 2^{L_w}`** (`cert_limbs`). -/
theorem L32_exact : ∀ N < 21, 2 ≤ N →
    2 * P N pm ≤ 2 ^ (32 * L32 N) ∧ 2 ^ (32 * (L32 N - 1)) < 2 * P N pm := by decide

/-- **The reachable peak needs every limb** (`cert_crt_peak`): `crt_peak_exact` is below
`2P ≤ 2^{L_w}`, and at least `2^{L_w − 32}`, so one limb fewer would overflow on the inputs
`CRT.crtPeak_attained` constructs. -/
theorem peak_needs_every_limb : ∀ N < 21, 2 ≤ N →
    2 ^ (32 * (L32 N - 1)) ≤ crtPeak N pm ∧ crtPeak N pm < 2 * P N pm := by decide

/-! ### The chunk length and the epilogue range -/

/-- `K_c = 131071` is the largest chunk with `128²·K_c ≤ 2³¹ − 1`, and `K_c' = 131008 =
2047·64` is the largest multiple of 64 not above it (DESIGN §2, `cert_kc`). -/
theorem chunk_constants :
    (128 ^ 2 : ℤ) * 131071 ≤ 2 ^ 31 - 1 ∧ (2 ^ 31 - 1 : ℤ) < 128 ^ 2 * 131072 ∧
    (131008 : ℕ) = 2047 * 64 ∧ 131008 ≤ 131071 ∧ 131071 < 131008 + 64 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- A chunk of at most `K` products, each in `[lo, hi]` with `lo ≤ 0 ≤ hi`, sums into
`[lo·K, hi·K]`. -/
theorem chunk_sum_range {f : ℕ → ℤ} {lo hi K : ℤ} (hlo : lo ≤ 0) (hhi : 0 ≤ hi)
    {C : Finset ℕ} (hC : (C.card : ℤ) ≤ K) (hf : ∀ h ∈ C, lo ≤ f h ∧ f h ≤ hi) :
    lo * K ≤ ∑ h ∈ C, f h ∧ ∑ h ∈ C, f h ≤ hi * K := by
  have h1 : (C.card : ℤ) * lo ≤ ∑ h ∈ C, f h := by
    have := Finset.card_nsmul_le_sum C f lo fun h hh => (hf h hh).1
    simpa using this
  have h2 : ∑ h ∈ C, f h ≤ (C.card : ℤ) * hi := by
    have := Finset.sum_le_card_nsmul C f hi fun h hh => (hf h hh).2
    simpa using this
  constructor <;> nlinarith

/-- **DESIGN §3.5's epilogue range at `K_c' = 131008`** (L4c re-instantiated; `cert_kc`,
`spec/apalache/Epilogue.tla`). With SMod residues of moduli `≤ 256` and chunks of at most
131008 products, every epilogue input `W_{c−1} + acc_c` lies in (tight at `p = 256`)
`[−128·127·131008, (p − 1) + 128²·131008] = [−2129666048, (p − 1) + 2146435072]`, inside int32
(for `p = 256` the maximum is `2146435327`). -/
theorem epilogue_range_v113 {p : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) {pr : ℕ}
    (hpr : 0 < pr) (hpr256 : pr ≤ 256) {a' b' : ℕ → ℤ} {C : ℕ → Finset ℕ}
    (hC : ∀ c, ((C c).card : ℤ) ≤ 131008) {acc : ℕ → ℤ}
    (hacc : ∀ c, acc c = ∑ h ∈ C c, smod (a' h) pr * smod (b' h) pr) (c : ℕ) :
    (-2129666048 : ℤ) ≤ epi p acc c + acc (c + 1) ∧
      epi p acc c + acc (c + 1) ≤ (p - 1) + 2146435072 ∧
      (p - 1 : ℤ) + 2146435072 ≤ 2146435327 ∧ (2146435327 : ℤ) ≤ 2 ^ 31 - 1 ∧
      -(2 ^ 31 : ℤ) ≤ -2129666048 := by
  have hp' : (0 : ℤ) < p := by exact_mod_cast hp
  have hp256' : (p : ℤ) ≤ 256 := by exact_mod_cast hp256
  have hW := epi_mem_range hp' acc c
  have hsum := chunk_sum_range (lo := -(128 * 127)) (hi := 128 ^ 2) (by norm_num) (by norm_num)
    (hC (c + 1)) (f := fun h => smod (a' h) pr * smod (b' h) pr) fun h _ =>
      ⟨smod_mul_ge _ _ hpr hpr256 hpr hpr256,
        (abs_le.1 (smod_mul_abs_le _ _ hpr hpr256 hpr hpr256)).2⟩
  rw [hacc (c + 1)]
  refine ⟨by linarith [hsum.1], by linarith [hsum.2], by linarith, by norm_num, by norm_num⟩

/-- **The epilogue's Barrett reduction is exact on every reachable input** (DESIGN §3.5,
`cert_barrett`): `epilogue_range_v113` gives `barrett_epilogue`'s hypotheses
(`v ≥ −2³¹` and `v + bias < 2³²` with `bias = p·⌈2³¹/p⌉ ≤ 2³¹ + p − 1`), so the u32
`n = v + bias` and `q = umulhi64(⌈2⁶⁴/p⌉, n)` give `n − p·q = (W_{c−1} + acc_c) mod p`. -/
theorem barrett_epilogue_v113 {p : ℕ} (hp : 0 < p) (hp256 : p ≤ 256) {pr : ℕ}
    (hpr : 0 < pr) (hpr256 : pr ≤ 256) {a' b' : ℕ → ℤ} {C : ℕ → Finset ℕ}
    (hC : ∀ c, ((C c).card : ℤ) ≤ 131008) {acc : ℕ → ℤ}
    (hacc : ∀ c, acc c = ∑ h ∈ C c, smod (a' h) pr * smod (b' h) pr) (c : ℕ)
    {v : ℤ} (hv : v = epi p acc c + acc (c + 1)) {n : ℕ}
    (hn : n = (v + (p * ((2 ^ 31 + p - 1) / p) : ℕ)).toNat) :
    ((n - p * (n * ((2 ^ 64 + p - 1) / p) / 2 ^ 64) : ℕ) : ℤ) = v % p := by
  obtain ⟨h1, h2, h3, -, -⟩ := epilogue_range_v113 hp hp256 hpr hpr256 hC hacc c
  rw [← hv] at h1 h2
  have hb : p * ((2 ^ 31 + p - 1) / p) ≤ 2 ^ 31 + p - 1 := by
    rw [mul_comm]; exact Nat.div_mul_le_self _ _
  have hb' : ((p * ((2 ^ 31 + p - 1) / p) : ℕ) : ℤ) ≤ 2 ^ 31 + p - 1 := by
    have h := (Nat.cast_le (α := ℤ)).2 hb
    rw [Nat.cast_sub (by omega)] at h
    push_cast at h ⊢
    linarith
  have hp' : (p : ℤ) ≤ 256 := by exact_mod_cast hp256
  rw [hn]
  exact barrett_epilogue hp hp256 (by linarith) (by linarith)

/-! ### The CRT multipliers -/

/-- `q_l = M_l⁻¹ mod p_l` at `N` moduli, by search over `[0, p_l)` (the unique inverse, which is
`spec/params.py`'s `pow(M_l mod p_l, −1, p_l)`). -/
def qd (N l : ℕ) : ℕ :=
  ((List.range (pm l)).find? (fun q => M N pm l % pm l * q % pm l = 1)).getD 0

/-- The multipliers exist and are the table's: `M_l·q_l ≡ 1 (mod p_l)` for every `N ≤ 20`. -/
theorem qd_inv : ∀ N < 21, ∀ l < N, M N pm l * qd N l ≡ 1 [MOD pm l] := by decide

/-! ### The capstone at the real moduli -/

/-- **End-to-end exactness for the shipped design** (DESIGN §3.2–§3.6). For `N ≤ 20`
moduli of DESIGN §2, `H = ⌊(P−1)/2⌋`, `T = ⌊√H⌋`, quantized rows with `‖a'‖, ‖b'‖ ≤ T` (L1),
SMod residues (§3.4), chunks of at most `K_c' = 131008` products accumulated in a saturating
int32 (`A = 2³¹ − 1`) in any order, the modular carry (§3.5) and the running CRT (§3.6): the
CRT output is exactly `Σ_h a'_h b'_h`. Its only remaining hypothesis on the constants is the
choice of the multipliers `q_l` (any `M_l·q_l ≡ 1 (mod p_l)`; they exist by
`exists_q_design`). -/
theorem emulated_product_exact_design {N : ℕ} (hN : N ≤ 20)
    {q : ℕ → ℕ} (hq : ∀ j < N, M N pm j * q j ≡ 1 [MOD pm j])
    {k T : ℕ} {a' b' : ℕ → ℤ} (hT : T = Nat.sqrt ((P N pm - 1) / 2))
    (ha : ∑ h ∈ range k, a' h ^ 2 ≤ (T : ℤ) ^ 2) (hb : ∑ h ∈ range k, b' h ^ 2 ≤ (T : ℤ) ^ 2)
    {n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ} (hK : ∀ c < n, ((C c).card : ℤ) ≤ 131008)
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {ord : ℕ → ℕ → List ℕ} (hord : ∀ j < N, ∀ c < n, Enumerates (ord j c) (C c)) :
    let W : ℕ → ℕ := fun j =>
      (epi (pm j) (fun c => satFold (2 ^ 31 - 1)
        (fun h => smod (a' h) (pm j) * smod (b' h) (pm j)) 0 (ord j c)) (n - 1)).toNat
    z N pm q W ((P N pm - 1) / 2) = ∑ h ∈ range k, a' h * b' h := by
  have hP := P_pos (N := N) (p := pm) fun l _ => pm_pos l
  exact emulated_product_exact_smod (fun l _ => pm_pos l) (fun l _ => pm_le_256 l)
    (pm_coprime_N hN) hq (by omega) hT ha hb (by norm_num) hn hK hdisj hcover hord

/-- **End-to-end exactness for the shipped design, with the shipped multipliers.**
`emulated_product_exact_design` at `q_l = qd N l` (`qd_inv`): no hypothesis on the constants
remains. -/
theorem emulated_product_exact_shipped {N : ℕ} (hN : N ≤ 20)
    {k T : ℕ} {a' b' : ℕ → ℤ} (hT : T = Nat.sqrt ((P N pm - 1) / 2))
    (ha : ∑ h ∈ range k, a' h ^ 2 ≤ (T : ℤ) ^ 2) (hb : ∑ h ∈ range k, b' h ^ 2 ≤ (T : ℤ) ^ 2)
    {n : ℕ} (hn : 0 < n) {C : ℕ → Finset ℕ} (hK : ∀ c < n, ((C c).card : ℤ) ≤ 131008)
    (hdisj : (range n : Set ℕ).PairwiseDisjoint C) (hcover : (range n).biUnion C = range k)
    {ord : ℕ → ℕ → List ℕ} (hord : ∀ j < N, ∀ c < n, Enumerates (ord j c) (C c)) :
    let W : ℕ → ℕ := fun j =>
      (epi (pm j) (fun c => satFold (2 ^ 31 - 1)
        (fun h => smod (a' h) (pm j) * smod (b' h) (pm j)) 0 (ord j c)) (n - 1)).toNat
    z N pm (qd N) W ((P N pm - 1) / 2) = ∑ h ∈ range k, a' h * b' h :=
  emulated_product_exact_design hN (fun j hj => qd_inv N (by omega) j hj) hT ha hb hn hK hdisj
    hcover hord

end Immerdink.Moduli
