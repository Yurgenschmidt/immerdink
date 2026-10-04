# immerdink roadmap (priorities set by Eric, 2026-10-03)

Where v1 stands, what comes next, and what each item addresses. **Proposed** ordering;
Eric reorders/approves before anything starts. Sizes: S ≈ a day of agent work, M ≈ a
milestone like M2/M3, L ≈ multi-milestone or research.

## Where we are (dev, DESIGN v1.3)

- **Done:** spec (TLA+ 191 models, Apalache 20, params certificates, bit-exact oracle),
  Lean blueprint/proofs, M0–M5, v1.2 (plan-time GEMM parameters, thread rules, K0 on
  device with `vec_mode`, capture-safe), v1.3 (CUDA-free `libimmerdink_host`: exact
  workspace/plan bytes and lowering summary from C and Python).
- **Performance vs GEMMul8 (same session, N = 14):** large 1.06–1.12×, low-k 1.5–1.9×,
  permuted tensors 1.28–1.44× (vs permute + GEMMul8), small (≤ 512) **0.35–0.8×**.
- **Accuracy:** no bound violation in 520 runs; N = 14 ≈ FP64 normwise, N = 16 guaranteed.
- **Tooling:** quiet-gated benches, memcap'd heavy jobs, nsys + ncu with counters.

## Scope rules (decided)

- **Kernel level only.** Caller-owned device pointers on the caller's stream. No H2D/D2H
  data movement, no internal tiling/out-of-core, no immerdink-owned streams or graphs
  (`imd_contract` is capture-safe instead). Callers (ehrenfest) own orchestration.
- **Modular, per-plan resource options with exact footprints**; one source of truth
  (`libimmerdink_host`) for all sizing.
- **Design-first:** DESIGN + TLA+ before code; bit-exact vs the oracle always.
- **General kernel, not CC-specific**; CC/ehrenfest shapes are test context only.

---

## Phase A — Kernel performance (near term)

| # | Item | Addresses | Evidence | Size | Your decision |
|---|---|---|---|---|---|
| A1 ✅ | **Small-problem overhead** (done 2026-10-03: fused KF launch for k̂ ≤ 512, 7 → 3 launches; small shapes at N = 8 0.34–0.52× → 0.84–0.96× GEMMul8 (0.84–0.86× in the orchestrator's same-session sweep b2c45cd, 0.92–0.96× in the agent's A/B), ≥ 1× at N = 14; docs/NOTES-A1.md) — fewer launches for small shapes (fuse K1 into the pack; one fused small-problem path; cheaper K4 setup) | ~50–70 µs/call floor; s128–s512 at 0.35–0.8× GEMMul8, worse at N = 8 | compare sweep; nsys launch gaps | M | none (kernel sequence change ⇒ Pipeline.tla first) |
| A2 ✅ | **K4 epilogue / main-loop overlap** (done 2026-10-03: persistent K4 + register-only mod-p epilogue; K4 at k=1024 60–64% → 80–86% of bound; docs/NOTES-A2.md) at moderate k (warp-specialized or pipelined mod-p epilogue) | K4 at 60–66% of bound at k ≈ 1024: low-k, DF, batch-innermost cases | per-phase profile; ncu now available | M–L | none |
| A3 ✅ | **Shape-dependent tile selection** (done 2026-10-03: four K4 configurations 128×256 / 128×128 / 64×128 / 64×64, chosen by a fitted rule in libimmerdink_host (`k4_select`, reported by `imd_lowering` v3); K1 long rows kept in L2 (r_mm); KF extended to k̂ ≤ 1024. N = 8 vs GEMMul8: s128–s512 0.82–0.87× → 1.00–1.11×, sq1024 0.94 → 1.04×, r_mm 0.81 → 0.96×; r_skinny / r_bigk unchanged (K4 at its bound; the gap is K1 + pack: D4); docs/NOTES-A3.md) — per-plan selection function, small table (or plan-time autotune) | one K4 config for every m:n:k; skinny/small/long-k | r_bigk 0.91–0.99×, small shapes | M | plan-time autotuning implemented as the optional plan option `k4_tune` (default off): 1–24 ms of plan creation and up to 256 MiB of temporary device memory, no measurable gain over the rule on the 4090 (NOTES-A3 §5) — **decided (Eric, 2026-10-04): keep it optional, default off**; revisit on other GPUs |
| A4 ✅ | Leftovers (done 2026-10-03: K1 long rows 55 → 69 %, K5 61–75 → 82–96 % of bound; docs/NOTES-A1.md): K1 on very long rows (55% of bound), K5 tuning | few % on long-k / large | per-phase profile | S | none |
| A5 | **Small-shape polish (low priority, not dropped)** (Eric, 2026-10-03: not a focus, but don't ignore forever): tiny cases (s128–sq1024 at N = 8) sit at roughly 0.87–1.2× GEMMul8 depending on clock state; the floor is three dependent launches (NOTES-A1). Candidates: KF beyond k̂ = 1024 (s_cc16), CRT inside a small-tile GEMM (2 launches), clock-locked measurements to settle the numbers | 20–110 µs calls; many-plans use | NOTES-A1, NOTES-A3, compare sweeps | M | when convenient |

## Phase B — Interfaces

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| B1 | **Python bindings for contraction** (nanobind; cupy/DLPack device arrays only; einsum front end reusing the host lib) | usability from Python/other projects; ehrenfest integration tests | M | priority vs A |
| B2 | **Performance model** on the host API: predicted per-phase time from §4/§5 bounds × calibrated efficiencies, using the lowering summary | callers choose N, g, vec_mode by cost; ehrenfest planning | M | after A stabilizes (else recalibrate) |
| B3 | **CI without a GPU**: host-only build (CUDA-free lib), oracle, params, TLC subset, Python tests on GitHub Actions; GPU tests stay local | regressions caught on every push to dev | S | OK to use GitHub Actions minutes? |
| B4 | Packaging (conda/pip for host lib + Python) | install outside this repo | S–M | when needed by ehrenfest |

## Phase C — Precision

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| C1 | **Adaptive precision** — options: (a) per-call N (re-pack; simplest), (b) auto-N from the M5 elementwise rule (N grows with log‖a‖‖b‖/‖x‖), (c) **Ozaki-I slice mode** (prefix slices: one packed operand serves every precision) | iterative methods that need full precision only near convergence; N = 8 is ~1.6× faster than N = 14 | L | which variant(s); TLA+ for slices |
| C2 | Accurate-mode scaling | more bits per modulus ⇒ smaller N at equal accuracy | M | after C1 |

## Phase D — New capabilities

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| D1 ✅ | **Packed-operand reuse** — pack an operand once, use it in many contractions (kernel-level: caller owns the packed buffer). Explicit handle (decided). Design approved 2026-10-03 (all §6b.10 recommendations); **implemented 2026-10-03** (DESIGN v1.7 §6b, host API v5: `imd_pack_create` / `imd_pack` / `imd_plan_create_packed` / `imd_contract_packed`, the CUDA-free sizing and compatibility queries; bit-exact; spec: IndexMap adopted orders, Pipeline packed slot, HostAPI pack handle, PackedOperand; docs/NOTES-D1.md). §6b.11 null-pointer clarification accepted (Eric, 2026-10-04) | same integrals/amplitudes in many contractions; pack is 10–15% of large calls, more at small g | M | interacts with C1(c) |
| D2 | Complex (2M / 3M via the moduli table) | complex-valued workloads | M | needed by ehrenfest? |
| D3 | **Packed symmetric contractions**, generic permutational symmetry (DF-CCSD PPL). Eric (2026-10-03): ideally **one kernel fits all** symmetry patterns; if that gets too hard, split it into separate sub-problems. Use TLA+ to settle whether the generic design is correct before choosing | memory and flops of symmetric tensors; no generic kernels exist even without Ozaki | L–XL (research) | after D1; symmetry descriptor API |
| D4 | Fused variants (pack inside the GEMM main loop; MEMORY_MODEL §7) — **part 1 done** (2026-10-03, user decision: K1 + pack fusion for large shapes first): per-operand launch KL = K1 + group 1's pack for k̂ > 1024 (DESIGN v1.6, host API v4 `fused_large`); DRAM reads of the scale + pack phase halved at g = N (one operand read saved at any g; workspace unchanged); N = 8 vs GEMMul8: r_skinny 0.93 → 1.18×, r_bigk 0.98 → 1.09×, r_mm 0.97 → 1.02×; sq2048–sq8192 and t_nn / t_df / t_cc / t_permout 8–18 % faster at N = 8; small shapes (KF) unchanged; docs/NOTES-D4.md | workspace and pack traffic | L | part 2 (pack inside the GEMM) not started |

## Phase E — Hardware

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| E1 | **Hopper/Blackwell back ends** (CUTLASS 3.x / wgmma; compare with cuTENSOR native FP64 and its 8XINT8 emulation) | server GPUs; where emulation actually beats native FP64 | L | needs H100/B200 access |

## Phase F — Verification and write-up

| # | Item | Addresses | Size |
|---|---|---|---|
| F1 | Test gaps: compute-sanitizer racecheck; a real n̂ > 16.7M split run (sampled check) | coverage | S |
| F2 | Paper track (if wanted): clock-locked runs, H100 numbers (E1), Lean proofs completed against DESIGN v1.3, write-up | credibility | L |

## Out of scope (decided)

Internal tiling / out-of-core, host↔device transfers, immerdink-owned CUDA graphs or
streams, multi-GPU orchestration, a workspace-budget option (g + the host query API
instead), a native-FP64 TBLIS path (cuTENSOR wins there), the K1 two-launch split
(measured slower; kept in the spec only), the pinned-host K0 copy.

## Priorities (Eric, 2026-10-03)

Focus: **straight performance and capabilities.** Software-engineering items are parked
until the repo matures.

1. **Now:** A1 small-problem overhead, A2 K4 overlap, A4 leftovers; then A3 tile selection (all done).
2. **Next:** D1 packed-operand reuse, D4 fused variants.
3. **Then:** D3 packed symmetric contractions, once performance is clean (high interest,
   not to be rushed).
4. **Medium:** B2 performance model (designed to be called directly from ehrenfest, so
   ehrenfest needs no model of its own), B1 Python bindings.
5. **After Ozaki-II is polished:** C1 adaptive precision.
6. **Parked:** B3 CI, B4 packaging, F1 test gaps, F2 paper track, D2 complex, E1
   Hopper/Blackwell (needs hardware).
