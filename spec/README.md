# spec/ — the executable theory of immerdink

Everything in DESIGN.md that can be checked is checked here, in three complementary ways:

| Layer | Tool | Covers | Run |
|---|---|---|---|
| Design logic | TLA+ / TLC | algorithms and protocols, **exhaustively at toy sizes** | `spec/check.sh [Module [model]]` |
| Real-width arithmetic | `params.py` | every constant and bound **at the real parameters**, exactly | `python3 spec/params.py --check` (`--table`, `--json`) |
| Numerics end to end | `oracle.py` + `test_oracle.py` | bit-exact reference of DESIGN §3 at real widths, random and adversarial inputs | `python3 spec/test_oracle.py` |

The hand proofs of the parametric lemmas (for all k, N and inputs) are in
`docs/PROOFS.md`, as a Lean blueprint.

Current status:
- 121 TLC models, all as expected (about 5 min with the default parallelism);
- 24/24 certificates;
- the oracle suite passes.

## Conventions
- `<Module>.tla` is the spec. `MC_<Module>.tla` holds model values; cfgs can't hold
  negatives or tuples.
- `MC_<Module>__pos_*.cfg` **must pass**. Positive models sit *on* each precondition's
  boundary. Coverage ASSUMEs check that the boundary is actually reached, so a model can't
  pass vacuously.
- `MC_<Module>__neg_*.cfg` **must fail**. They carry a `\* EXPECT: <TLC text>` line, and
  each one drops exactly one rule or moves exactly one boundary. Together they show every
  precondition is necessary.
- Some `neg_witness_*` models exist only to prove a corner case is *reachable*: a tie, a
  subnormal, overflow to Inf, an exact hit on the budget, and so on.
- `check.sh` runs models in parallel (`TLC_JOBS`, `TLC_WORKERS`, `TLC_TIMEOUT`). Model
  checking is pass/fail, so it is not gated on a quiet machine.
- Every artifact was attacked with mutation testing: realistic single-point bugs injected
  into copies. Every mutant from the review and fix rounds is killed by some model or test.

## Modules

| Module | DESIGN | What it establishes | Key invariants / properties | Models |
|---|---|---|---|---|
| `OzakiCRT` | §3.4–§3.6 | One output element: residues, the int8 operand store, a **saturating (havoc) INT32 accumulator** in any order, K-chunks with mod carry, and CRT uniqueness for `Σ|a'||b'| ≤ H` | `ResiduesFit`, `ChunkFits`, `EpilogueFits`, `WCongruent`, `CRTExact`, `CRTRecon` | 6 pos / 7 neg |
| `CRTRealization` | §3.6 | The division-free running CRT **refines** the exact CRT, through three layers: exact CRT → `L_w`-bit word → u32 limb carry chains. Covers the word-size boundary, borrow-based centering and q-weights. | `Refines`, `RefinesWord`, `XBelowP`, `PeakBound`, `NoWordOverflow`, `ZExact`, `Termination` | 8 pos / 13 neg |
| `OzakiScaling` (+`ToyFloat`) | §3.2, §3.3, §3.7, §3.8 | With a toy FP format including subnormals: the exact-integer shift rule is maximal; RN and TRUNC quantization; the norm budget ⇒ the CRT precondition; the ℓ1 and normwise error bounds; a **single** correctly-rounded final conversion (subnormal, overflow, sign of zero); κ = ⌈√k⌉ and RN soundness up to κ ≤ 2T−1 | `ShiftIsMaximal`, `SExact`, `NormBudget`, `CRTPrecondition`, `SigmaBound`, `ErrorBound`, `ContractL2`, `FinalRounding`, `QuantIsRNEven` | 15 pos / 26 neg |
| `IndexMap` | §5 | Label lowering and legality on tiny tensors (extents up to 3). `Scatter∘GEMM∘Gather` equals the einsum as a multiset of address triples; the encodings are bijections; the stride rule ⇒ injective C writes; status precedence; canonical orders; the corrected fusion rule | `EinsumPreserved`, `StrideRuleImpliesInjective`, `DecodeBijective`, `StatusCorrect`, `CanonicalOrderSorted` | 11 pos / 7 neg |
| `Pipeline` | §4, §4a | The kernel sequence K0–K5 as a state machine, with interleaved CTAs, residue groups g, K-chunks, the batch-launch split and DIRECT. Writes happen exactly once; no stale reads; buffers are reused only after retire; the **workspace layout is exact** and equals §4a; read rules for α = 0 / β = 0 / k = 0; aliasing; the two-stream hazard | `NoStaleRead`, `FinalAtMostOnce`, `FinalExactlyOnceAtEnd`, `PlanesFinalAtCRT`, `ChunksPartition`, `LayoutExact`, `RegionsFit`, `BetaZeroNoCRead`, `AlphaZeroNoABRead`, `Termination` | 10 pos / 18 neg |

## How the layers connect
- **L1 / OzakiScaling** shows `Σ|a'||b'| ≤ H`. That is exactly **OzakiCRT**'s precondition
  for `CRTExact`.
- **OzakiCRT**'s `CRTRecon` uses the weight formula. **CRTRealization** shows the word and
  limb algorithm refines that same exact CRT.
- **Pipeline** treats per-element numerics as atomic. It checks *when* and *where* data
  moves, using the same chunk and plane semantics as OzakiCRT.
- **params.py** evaluates every scaled-model precondition at the real parameters: moduli,
  K_c, L_w, the σ range, and the workspace formula. `cert_scaled_tla` checks that the toy
  parameters are produced by the same greedy rule as the real ones.
- **oracle.py** is the reference the GPU must match bit for bit from M1 on.

## Known follow-ups (DESIGN v1.1 items not yet in a model)
- IndexMap: the coalescing **operand-role swap** (§5 step 8); the workspace ∩ A/B/C
  ALIASING row; the MISALIGNED status.
- Pipeline: K-empty → DIRECT dispatch by descriptor (`pos_direct` covers `mode = DIRECT`).
  The per-batch modulus table is abstracted away.
- oracle.py: the K-empty → DIRECT dispatch. It currently emulates k = 1, which is exact
  but rounds differently from the DIRECT kernel.
