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
- 239 TLC models, all as expected (about 9 min with the default parallelism);
- 20 Apalache checks, all as expected (about 30 s, one at a time);
- 25/25 certificates;
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
  fits in memory, and the run sits inside the shared `tools/memcap.sh` budget. All 164 models (v1.1)
  peak at about 6.4 GB (TLC RSS; 9.1 min wall). Keep every model small (well under 1M distinct states, a few minutes):
  shrink constants rather than raising `TLC_HEAP`.
- Every artifact was attacked with mutation testing: realistic single-point bugs injected
  into copies. Every mutant from the review and fix rounds is killed by some model or test.

## Modules

| Module | DESIGN | What it establishes | Key invariants / properties | Models |
|---|---|---|---|---|
| `OzakiCRT` | §3.4–§3.6 | One output element: residues, the int8 operand store, a **saturating (havoc) INT32 accumulator** in any order, K-chunks with mod carry, and CRT uniqueness for `Σ|a'||b'| ≤ H` | `ResiduesFit`, `ChunkFits`, `EpilogueFits`, `WCongruent`, `CRTExact`, `CRTRecon` | 6 pos / 7 neg |
| `CRTRealization` | §3.6 | The division-free running CRT **refines** the exact CRT, through three layers: exact CRT → `L_w`-bit word → u32 limb carry chains. Covers the word-size boundary, borrow-based centering and q-weights. | `Refines`, `RefinesWord`, `XBelowP`, `PeakBound`, `NoWordOverflow`, `ZExact`, `Termination` | 8 pos / 13 neg |
| `OzakiScaling` (+`ToyFloat`) | §3.2, §3.3, §3.7, §3.8 | With a toy FP format including subnormals: the exact-integer shift rule is maximal; RN and TRUNC quantization; the norm budget ⇒ the CRT precondition; the ℓ1 and normwise error bounds; a **single** correctly-rounded final conversion (subnormal, overflow, sign of zero); κ = ⌈√k⌉ and RN soundness up to κ ≤ 2T−1 | `ShiftIsMaximal`, `SExact`, `NormBudget`, `CRTPrecondition`, `SigmaBound`, `ErrorBound`, `ContractL2`, `FinalRounding`, `QuantIsRNEven` | 15 pos / 26 neg |
| `IndexMap` | §5, §4a, §6 | Label lowering and legality on tiny tensors (extents up to 3). `Scatter∘GEMM∘Gather` equals the einsum as a multiset of address triples; the encodings are bijections; the stride rule ⇒ injective C writes; the corrected fusion rule. The **operand-role swap** (only C modes of extent > 1; before sizing, the GEMM limits and the canonical orders, with position from the caller's label order) makes j̄ C's fastest axis and preserves every claim. K-empty → DIRECT by descriptor; the workspace is sized from post-swap dims and is exact; the GemmCoord limit applies only in EMULATE with a non-empty output and k ≥ 2. The §3.2 **κ ≥ T accuracy policy** (NOT_SUPPORTED, symbolic T) rejects exactly when the emulated GEMM runs with RN (EMULATE, non-empty output, k ≥ 2); each scope condition is shown necessary. `imd_contract`: null rule (C always; A/B only when α ≠ 0 and k ≠ 0; ws only when workspace_size > 0), empty output, ALIASING incl. `[ws, ws + workspace_size)` ∩ A/B/C, MISALIGNED, INSUFFICIENT_WORKSPACE, in DESIGN §6's order | `EinsumPreserved`, `StrideRuleImpliesInjective`, `DecodeBijective`, `StatusCorrect`, `CanonicalOrderSorted`, `CanonicalOrderExact`, `SwapWellDefined`, `SwapMakesJFastest`, `DispatchByDescriptor`, `WorkspaceFits`, `WorkspaceExact`, `GemmDimsFit`, `LimitScope`, `KappaAccepted`, `KappaScope`, `NoNullDeref`, `WsNoAlias`, `AliasingSound`, `AlignedAccess`, `WsSufficient` | 18 pos / 34 neg |
| `Pipeline` | §4, §4a | The kernel sequence K0–K5 as a state machine, with interleaved CTAs, residue groups g, K-chunks, the batch-launch (gridDim.z) split, the **n̂ split across launches** (gridDim.y ≤ MaxY N tiles, symbolic; partial last tile and short last launch) and DIRECT. Writes happen exactly once (C elements and, per K-chunk, every W cell incl. the padding columns); the result is independent of the z/y splits; no stale reads; buffers are reused only after retire; the **workspace layout is exact** and equals §4a; read rules for α = 0 / β = 0 / k = 0; K-empty → DIRECT decided by the descriptor (workspace = align(vec)); aliasing; the two-stream hazard. v1.2: K0 is a kernel (one CTA per tensor's vectors; a short grid is caught); `K1Split` = K1a (γ into sA/sB as scratch) then K1b (σ over γ): σ final before any pack / K5 read, K1b reads γ only, one K1b CTA per row, the packs' event edge from K1b (both K1 variants have positive models; the split is not adopted, see NOTES-M4); `VecResident` (IMD_VEC_RESIDENT): no K0, no vec region, the vectors in a plan buffer whose size is exactly the vectors (`PlanBytesExact`). v1.4: `Fused` (KF = K0 + K1 + group 1's packs in one launch, offsets from the descriptor, σ per moduli part) and `MParts`. v1.6: `FusedLarge` (per operand, KL = K1 + group 1's pack of that operand in one launch after K0: reads the vectors only after K0 retires, one CTA per whole row, σ stored, group 1 only) | `DispatchByDescriptor`, `NoStaleRead`, `SigmaFinalAtUse`, `FinalAtMostOnce`, `FinalExactlyOnceAtEnd`, `PlanesFinalAtCRT`, `PlanesWrittenOnce`, `PlanesCoverAtEnd`, `LaunchGridOK`, `LaunchGridYOK`, `ChunksPartition`, `LayoutExact`, `RegionsFit`, `BetaZeroNoCRead`, `AlphaZeroNoABRead`, `Termination` | 30 pos / 39 neg |
| `HostAPI` | §6 (v1.2) | Host threads, plans, workspaces and streams around `imd_contract`: shared host locations are accessed as Begin/End pairs, so an overlapping conflicting access of another thread is a data race; GPU operations run per stream in order, streams interleaved. One plan shared by threads and streams is safe because `imd_contract` only reads it (GEMM parameters built at creation); a plan may be destroyed once every call using it returned, because K0 takes the descriptor by value. Negatives: a lazily filled plan cache, one mutable operator per plan, lazily initialized process-global state, a pinned K0 source or resident vectors with the generate-mode lifetime rule, a workspace shared by concurrent calls (also on one stream). Witnesses show the positives reach early destroy, concurrent reads and interleaved launches | `NoHostRace`, `ParamsOwn`, `NoUseAfterFree`, `NoWsHazard`, `CompleteAtEnd`, `Termination` | 5 pos / 10 neg |
| `PackedOperand` | §6b (**DRAFT**, D1) | Packed-operand reuse. *CompatSpec* (constant-level, §3 numerics at toy widths in exact integers): a contraction that reads a packed operand (σ, flags, all N planes in the §6b.4 byte layout, the plan **adopting the pack's K order**) gives the same σ, W planes and C as packing in-line, for both post-swap roles, every g, RN and TRUNC, one or two packed operands, non-finite inputs; the emulated result is K-order independent; the key check rejects every key mutation and accepts the matching pack. Negatives: each key part (K order not adopted, N, rounding, K set with a partner-only label, k̂ format, two packs' K orders) and each byte-contract rule (B-role strides, tail, zero K padding, flags) is necessary; a witness shows flagged (native-path) outputs depend on the K order. *StreamSpec*: imd_pack / contracts / a later re-pack, free or FP64 overwrite on streams with events: RAW and WAR ordering (the caller's job) is necessary; concurrent readers are safe | `SigmaEqual`, `WEqual`, `CEqual`, `NoOOBRead`, `EmulatedOrderFree`, `AcceptedWhenKeyMatches`, `NoStaleRead`, `PackReadsX1`, `CompleteAtEnd`, `Termination` | 11 pos / 17 neg |

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
  `oracle.lower(..., build_vec=False)` returns the same view with lazily built scatter
  vectors (`LazyVecs`), so status / lowering-only use costs O(rank) even for a legal
  2^31-row descriptor; the default (`build_vec=True`) still builds them eagerly.
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

## Known follow-ups (DESIGN items not yet in a model)
- IndexMap's workspace claims (`WorkspaceFits`, `WorkspaceExact`) model the default
  `IMD_VEC_GENERATE` layout; the v1.2 `IMD_VEC_RESIDENT` layout (no vec region) is checked by
  Pipeline (`LayoutExact`, `RegionsUsed`, `PlanBytesExact`) and `params.py cert_workspace`.
- Pipeline: the per-batch modulus table is abstracted away. α is not an IndexMap input
  (only the §6 null rule depends on it there); the α = 0 read rules are in Pipeline.
