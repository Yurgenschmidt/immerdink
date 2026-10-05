# immerdink roadmap (priorities set by Eric, 2026-10-03)

Where v1 stands, what comes next, and what each item addresses. **Proposed** ordering;
Eric reorders/approves before anything starts. Sizes: S ≈ a day of agent work, M ≈ a
milestone like M2/M3, L ≈ multi-milestone or research.

## Where we are (dev, DESIGN v1.3)

- **Done:** spec (TLA+ 191 models, Apalache 20, params certificates, bit-exact oracle),
  Lean blueprint/proofs, M0–M5, v1.2 (plan-time GEMM parameters, thread rules, K0 on
  device with `vec_mode`, capture-safe), v1.3 (CUDA-free `libimmerdink_host`: exact
  workspace/plan bytes and lowering summary from C and Python).
- **Performance (RTX 4090):** sq8192 ≈ 64 / 37 TFLOPS-eq at N = 8 / 14 (NOTES-D4 §5),
  permuted tensors within a few % of the matrix case, K4 at 90–98 % of its bound on large
  shapes; small (≤ 512³) calls 15–35 µs (NOTES-A5 §3.3).
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
| A1 ✅ | **Small-problem overhead** (done 2026-10-03: fused KF launch for k̂ ≤ 512, 7 → 3 launches; small shapes s128–s512 at N = 8 51–66 → 19–36 µs per call; docs/NOTES-A1.md) — fewer launches for small shapes (fuse K1 into the pack; one fused small-problem path; cheaper K4 setup) | ~50–70 µs/call floor on s128–s512 | sweep; nsys launch gaps | M | none (kernel sequence change ⇒ Pipeline.tla first) |
| A2 ✅ | **K4 epilogue / main-loop overlap** (done 2026-10-03: persistent K4 + register-only mod-p epilogue; K4 at k=1024 60–64% → 80–86% of bound; docs/NOTES-A2.md) at moderate k (warp-specialized or pipelined mod-p epilogue) | K4 at 60–66% of bound at k ≈ 1024: low-k, DF, batch-innermost cases | per-phase profile; ncu now available | M–L | none |
| A3 ✅ | **Shape-dependent tile selection** (done 2026-10-03: four K4 configurations 128×256 / 128×128 / 64×128 / 64×64, chosen by a fitted rule in libimmerdink_host (`k4_select`, reported by `imd_lowering` v3); K1 long rows kept in L2 (r_mm); KF extended to k̂ ≤ 1024. N = 8: s128–s512 10–26 % faster per call, r_mm 2.64 → 2.25 ms; r_skinny / r_bigk unchanged (K4 at its bound; what is left is K1 + pack: D4); docs/NOTES-A3.md) — per-plan selection function, small table (or plan-time autotune) | one K4 config for every m:n:k; skinny/small/long-k | per-phase profile of small, skinny and long-k shapes | M | plan-time autotuning implemented as the optional plan option `k4_tune` (default off): 1–24 ms of plan creation and up to 256 MiB of temporary device memory, no measurable gain over the rule on the 4090 (NOTES-A3 §5) — **decided (Eric, 2026-10-04): keep it optional, default off**; revisit on other GPUs |
| A4 ✅ | Leftovers (done 2026-10-03: K1 long rows 55 → 69 %, K5 61–75 → 82–96 % of bound; docs/NOTES-A1.md): K1 on very long rows (55% of bound), K5 tuning | few % on long-k / large | per-phase profile | S | none |
| A5 ✅ | **Small-shape polish** (done 2026-10-04, docs/NOTES-A5.md): measured first — nsys: 3 launches, ≈ 3 µs launch latency and 1.3 µs of gaps per call, the rest kernel time with K5 the outlier; the clean sweep's weak s256 / s512 numbers were per-process performance states (host core type, DVFS) sampled by 10 sequential calls, so tiny-call A/Bs need an interleaved `--methods duel` or many calls. Changes (bit-identical, same launches): K5 compiled per N with 1/2/4 outputs per thread (780 → 417 instructions per output; s512 K5 9.4 → 6.5 µs), pack residues with the 32-bit magic, §3.2 term in 64-bit words (KF s512 12.5 → 11.2 µs, sq1024 40.8 → 33.7 µs). N = 8, 300-call A/B: s128 19.4 → 15.2 µs, s256 18.7 → 16.4, s512 29.9 → 27.0, sq1024 88.1 → 79.9 (small family 4–22 % faster); full sweep no regression (large shapes 0–6 % faster). Not needed: CRT fused into a small-tile GEMM (would serialize K5 on few CTAs), K4 rule change | 20–110 µs calls; many-plans use | NOTES-A5 | M | done |

## Phase B — Interfaces

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| B1 ✅ | **Python bindings for contraction** (done 2026-10-05, DESIGN v1.9 §6c): `python/immerdink` `Plan`, `Pack`, `PlanCache`, einsum front end `immerdink.contract`, raw entry points `contract_ptr` / `pack_ptr`. Decided by Eric: ctypes over the C ABI (no nanobind, no build dependency), any `__cuda_array_interface__` / DLPack array or raw pointer (`DeviceArray`), no array library required; float64 only; no host↔device copy, no allocation, caller's stream and workspace; calls capture-safe (tested: graph capture and replay). Exact footprints from the C library (`workspace_bytes`, `device_bytes`, `host_bytes`, `lowering`). Per-call host time on a 16³ call: bare ctypes `imd_contract` 3.9 µs, `contract_ptr` +0.05 µs, `Plan.contract` with CAI arrays +0.9–1.0 µs, `immerdink.contract` (cached plan) +1.9–2.1 µs (`bench/python_overhead.py`). Test `python_device` (pytest, 22 tests, bit-exact vs the oracle incl. packed operands, threads, streams, errors; no CuPy / PyTorch needed) | usability from Python/other projects; ehrenfest integration tests | M | done |
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
| D3 | **Packed symmetric contractions**, generic permutational symmetry (DF-CCSD PPL). Eric (2026-10-03): ideally **one kernel fits all** symmetry patterns; if that gets too hard, split it into separate sub-problems. Use TLA+ to settle whether the generic design is correct before choosing. **Starts with a design/theory discussion with Eric to narrow the questions (2026-10-04); no agent work before that** | memory and flops of symmetric tensors; no generic kernels exist even without Ozaki | L–XL (research) | after D1; symmetry descriptor API |
| D4 | Fused variants (pack inside the GEMM main loop; MEMORY_MODEL §7) — **part 1 done** (2026-10-03, user decision: K1 + pack fusion for large shapes first): per-operand launch KL = K1 + group 1's pack for k̂ > 1024 (DESIGN v1.6, host API v4 `fused_large`); DRAM reads of the scale + pack phase halved at g = N (one operand read saved at any g; workspace unchanged); N = 8: r_skinny 3.02 → 2.39 ms, r_bigk 7.39 → 6.67 ms, r_mm 2.19 → 2.10 ms; sq2048–sq8192 and t_nn / t_df / t_cc / t_permout 8–18 % faster at N = 8; small shapes (KF) unchanged; docs/NOTES-D4.md | workspace and pack traffic | L | part 2 (pack inside the GEMM) **deferred** (2026-10-04): part 1 closed the targeted gaps, K4 is at 90–98 % of bound on large shapes, and D1 packed operands already cut workspace; revisit only for a memory-tight use case |

## Phase E — Hardware

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| E1 | **Hopper/Blackwell back ends** (CUTLASS 3.x / wgmma) | server GPUs; where emulation actually beats native FP64 | L | needs H100/B200 access |

## Phase F — Verification and write-up

| # | Item | Addresses | Size |
|---|---|---|---|
| F1 | Test gaps: compute-sanitizer racecheck; a real n̂ > 16.7M split run (sampled check) | coverage | S |
| F2 | Paper track (if wanted): clock-locked runs, H100 numbers (E1), Lean proofs completed against DESIGN v1.3, write-up. **Clock-locked comparison done (2026-10-04)** (local evaluation workspace) | credibility | L |

## Out of scope (decided)

Internal tiling / out-of-core, host↔device transfers, immerdink-owned CUDA graphs or
streams, multi-GPU orchestration, a workspace-budget option (g + the host query API
instead), a native-FP64 TBLIS path (native FP64 is out of scope; cuTENSOR covers it), the
K1 two-launch split (measured slower; kept in the spec only), the pinned-host K0 copy.

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
