# spec/ — the executable theory of immerdink

Everything in DESIGN.md that can be checked is checked here, in four complementary ways:

| Layer | Tool | Covers | Run |
|---|---|---|---|
| Design logic | TLA+ / TLC | algorithms and protocols, **exhaustively at toy sizes** | `spec/check.sh [Module [model]]` |
| Real-width bounds, symbolically | Apalache (SMT) | the CRT word bound, the int32 chunk bound and the epilogue range **at the real widths**, as inductive invariants over unbounded integers | `spec/apalache/check.sh [name-prefix]` |
| Real-width arithmetic | `params.py` | every constant and bound **at the real parameters**, exactly | `python3 spec/params.py --check` (`--table`, `--json`) |
| Numerics end to end | `oracle.py` + `test_oracle.py` | bit-exact reference of DESIGN §3 at real widths, random and adversarial inputs | `python3 spec/test_oracle.py` |

The hand proofs of the parametric lemmas (for all k, N and inputs) are in
`docs/PROOFS.md`, as a Lean blueprint.

Current status:
- 151 TLC models, all as expected (about 8 min with the default parallelism);
- 20 Apalache checks, all as expected (about 30 s, one at a time);
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
- `check.sh` runs models in parallel (`TLC_JOBS`, `TLC_HEAP`, `TLC_WORKERS`, `TLC_TIMEOUT`).
  Model checking is pass/fail, so it is not gated on a quiet machine. Each JVM's heap and
  off-heap fingerprint set are capped (`TLC_HEAP`, default 2g), `TLC_JOBS` is clamped to what
  fits in memory, and the run sits inside the shared `tools/memcap.sh` budget. All 151 models
  peak at about 6.4 GB (TLC RSS; 8.3 min wall). Keep every model small (well under 1M distinct states, a few minutes):
  shrink constants rather than raising `TLC_HEAP`.
- Every artifact was attacked with mutation testing: realistic single-point bugs injected
  into copies. Every mutant from the review and fix rounds is killed by some model or test.

## Modules

| Module | DESIGN | What it establishes | Key invariants / properties | Models |
|---|---|---|---|---|
| `OzakiCRT` | §3.4–§3.6 | One output element: residues, the int8 operand store, a **saturating (havoc) INT32 accumulator** in any order, K-chunks with mod carry, and CRT uniqueness for `Σ|a'||b'| ≤ H` | `ResiduesFit`, `ChunkFits`, `EpilogueFits`, `WCongruent`, `CRTExact`, `CRTRecon` | 6 pos / 7 neg |
| `CRTRealization` | §3.6 | The division-free running CRT **refines** the exact CRT, through three layers: exact CRT → `L_w`-bit word → u32 limb carry chains. Covers the word-size boundary, borrow-based centering and q-weights. | `Refines`, `RefinesWord`, `XBelowP`, `PeakBound`, `NoWordOverflow`, `ZExact`, `Termination` | 8 pos / 13 neg |
| `OzakiScaling` (+`ToyFloat`) | §3.2, §3.3, §3.7, §3.8 | With a toy FP format including subnormals: the exact-integer shift rule is maximal; RN and TRUNC quantization; the norm budget ⇒ the CRT precondition; the ℓ1 and normwise error bounds; a **single** correctly-rounded final conversion (subnormal, overflow, sign of zero); κ = ⌈√k⌉ and RN soundness up to κ ≤ 2T−1 | `ShiftIsMaximal`, `SExact`, `NormBudget`, `CRTPrecondition`, `SigmaBound`, `ErrorBound`, `ContractL2`, `FinalRounding`, `QuantIsRNEven` | 15 pos / 26 neg |
| `IndexMap` | §5, §4a, §6 | Label lowering and legality on tiny tensors (extents up to 3). `Scatter∘GEMM∘Gather` equals the einsum as a multiset of address triples; the encodings are bijections; the stride rule ⇒ injective C writes; the corrected fusion rule. The **operand-role swap** (only C modes of extent > 1; before sizing, the GEMM limits and the canonical orders, with position from the caller's label order) makes j̄ C's fastest axis and preserves every claim. K-empty → DIRECT by descriptor; the workspace is sized from post-swap dims and is exact; the GemmCoord limit applies only in EMULATE with a non-empty output and k ≥ 2. `imd_contract`: null rule (C always; A/B only when α ≠ 0 and k ≠ 0; ws only when workspace_size > 0), empty output, ALIASING incl. `[ws, ws + workspace_size)` ∩ A/B/C, MISALIGNED, INSUFFICIENT_WORKSPACE, in DESIGN §6's order | `EinsumPreserved`, `StrideRuleImpliesInjective`, `DecodeBijective`, `StatusCorrect`, `CanonicalOrderSorted`, `CanonicalOrderExact`, `SwapWellDefined`, `SwapMakesJFastest`, `DispatchByDescriptor`, `WorkspaceFits`, `WorkspaceExact`, `GemmDimsFit`, `LimitScope`, `NoNullDeref`, `WsNoAlias`, `AliasingSound`, `AlignedAccess`, `WsSufficient` | 17 pos / 29 neg |
| `Pipeline` | §4, §4a | The kernel sequence K0–K5 as a state machine, with interleaved CTAs, residue groups g, K-chunks, the batch-launch split and DIRECT. Writes happen exactly once; no stale reads; buffers are reused only after retire; the **workspace layout is exact** and equals §4a; read rules for α = 0 / β = 0 / k = 0; K-empty → DIRECT decided by the descriptor (workspace = align(vec)); aliasing; the two-stream hazard | `DispatchByDescriptor`, `NoStaleRead`, `FinalAtMostOnce`, `FinalExactlyOnceAtEnd`, `PlanesFinalAtCRT`, `ChunksPartition`, `LayoutExact`, `RegionsFit`, `BetaZeroNoCRead`, `AlphaZeroNoABRead`, `Termination` | 11 pos / 19 neg |

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
- **Apalache** (`spec/apalache/`) re-proves the numeric bounds that TLC only checks at toy
  widths, at the real ones and symbolically. Each claim is an inductive invariant checked
  as three runs (`Init ⇒ IndInv`, `IndInv ∧ Next ⇒ IndInv'`, `IndInv ⇒ Safety`), plus
  expected violations that show each bound is tight:
  - `CRTWord16` / `CRTWord20` (generated by `gen_crtword.py` from `params_out.json`; the
    runner refuses stale ones): the running CRT keeps x < P and the peak x + M_l·y < 2P
    ≤ 2^{L_w} for the real moduli at N = 16 (L_w = 128) and N = 20 (L_w = 160). Bounded
    checking over all N steps shows the exact peak `crt_peak_exact` is reached and never
    exceeded, and that without the conditional subtraction x ≥ P;
  - `Int32Chunk`: any partial sum of ≤ 131071 products with |r·s| ≤ 128² stays in int32,
    and 131072 does not;
  - `Epilogue`: W + acc stays in int32 for K_c' = 131056, and the maximum is reached.

  Apalache runs one check at a time inside `tools/memcap.sh` with `JVM_ARGS=-Xmx4g`
  (knobs: `APA_JOBS`, `APA_TIMEOUT`). It is at `tools/apalache/bin/apalache-mc`
  (`tools/get_apalache.sh`) or `$APALACHE_MC`; usage notes: no RECURSIVE or CHOOSE,
  type annotations, constants precomputed.

## Known follow-ups (DESIGN v1.1 items not yet in a model)
- Pipeline: the per-batch modulus table is abstracted away. α is not an IndexMap input
  (only the §6 null rule depends on it there); the α = 0 read rules are in Pipeline.
- IndexMap: the gridDim.y split of n̂ (§4 launch limits) and the κ ≥ T accuracy policy's
  scope (EMULATE, non-empty, k ≥ 2) are not modelled.
