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
| A6 ✅ | **K4 load alignment** (DESIGN v1.13 §4b, docs/NOTES-K4ALIGN.md). Measured: CUTLASS loads a K-chunk's residue tile first, so row r's K4 loads sit at phase (r·k̂ + chunk end) mod 64. k̂ ≡ 16 / 48 (mod 64) costs +32–44 % in K4 on 128×256 and +69–94 % on 64×64; class 32 costs +3 % / +15–19 %. **Stage 0 + 1 done (2026-10-05):** one `k_hat_of` in libimmerdink_host, and K_c' = 131008 (= 2047·64, was 131056), so full chunks no longer shift the phase. This is bit-identical with no layout or API change; k̂ ∈ (j·131008, j·131056] gains a chunk. Measured: 2048² × 196608, N = 8 went from 23.5 to 32.0 TFLOPS-eq (70.1 → 51.5 ms), and 196640 from 25.8 to 31.8; the window cases were unchanged. **Threshold frozen (sweep 2026-10-05): `K_PAD64_MIN = 96`** (mean call loss 0.30 % vs per-k best; +2.8–5.6 % per call vs never padding). **Re-checked with the real A = 64 (2026-10-06, NOTES-K4ALIGN §11): 96 stays** (best k-only alternative, a per-k oracle, +0.065 % mean gain / 0.228 % mean loss vs the per-case best; padding-fraction rules lose; a per-case oracle gains only 0.29 %). **Stage 2 merged** (2026-10-06, 142b9a0; DESIGN §4b; K4Align + Pipeline + PackedOperand models, `cert_k_align`): k̂ = ⌈k⌉₆₄ for k > 96, pack format 2, host API 9 (`imd_lowering_t.k_align`), test hook `imd_test_set_k_align`, `imd_bench --k-align`; bit-identical results, the §4b.8 tests (new `host_k_align`, `contract_k_align`, `packed_k_align`). Perf model θ refit with A = 16-forced cases at k > 96 (NOTES-B2 §8, NOTES-K4ALIGN §10) and `k4_select` re-checked (no change), both merged in 142b9a0; the real-implementation threshold sweep (NOTES-K4ALIGN §11) kept `K_PAD64_MIN = 96`. **Next:** nothing open; the skinny-shape loss at k̂ ≡ 16 / 32 (−1 to −4 % per call) would need the deferred caller-chosen A = 16 pack override (DESIGN §4b.3), only if a user needs it | every k̂ ≢ 0 mod 64 (half of all k̂ pay ≥ 30 % in K4); long k (k̂ > 131008) | NOTES-K4ALIGN | M | done (threshold 96 kept) |
| A7 ✅ | **Branch-free ε in the pack emitter** (DESIGN v1.14 §3.4.1, 2026-10-06, branch `bf-eps`): |a'| held as W = 2 (N ≤ 16) or 3 (N = 17–20) u32 words, residue by W `dp4a` against byte weights 2^(8i) mod p + one `umulhi`; no ε > 0 branch, no lane-divergent `pow2_mod` load; one emitter for every N, templated on W per plan; same bytes (no format change); `cert_pack_words`, GPU unit test `pack_emitter`, contract suite `eps`. Measured: pack phases −20 to −41 % at N ≥ 15 on pack-heavy shapes (1024³ call +15–20 %, k-heavy +14–21 % at N ≥ 16), dominant-entry data no longer slower, N ≤ 13 unchanged (≤ 1 %). **Model follow-up done** (2026-10-06, branch `followups-combined`, DESIGN v1.15 §6c.1, docs/NOTES-BFEPS.md, NOTES-B2 §10): the perf model's eps path retired (`op_eps` 0, `eps_fraction` ignored, `eps_s` 0, keys optional, format 2 kept) and the shipped calibration refit . **v1.16 leftovers** (branch `emitter-w1`, NOTES-BFEPS §4): defined W = 2 shift (PTX shl), exact s maxima, W-aware occupancy query; W = 1 for N ≤ 8 evaluated and not adopted (pack −2.7 % < 3 % bar, +11 % library) | N ≥ 15 pack phases; data-dependent pack cost | eval findings/bf-eps-implementation.md, findings/openitems-emitter.md | S | done |

## Phase B — Interfaces

| # | Item | Addresses | Size | Your decision |
|---|---|---|---|---|
| B1 ✅ | **Python bindings for contraction** (done 2026-10-05, DESIGN v1.9 §6d): `python/immerdink` `Plan`, `Pack`, `PlanCache`, einsum front end `immerdink.contract`, raw entry points `contract_ptr` / `pack_ptr`. Decided by Eric: ctypes over the C ABI (no nanobind, no build dependency), any `__cuda_array_interface__` / DLPack array or raw pointer (`DeviceArray`), no array library required; float64 only; no host↔device copy, no allocation, caller's stream and workspace; calls capture-safe (tested: graph capture and replay). Exact footprints from the C library (`workspace_bytes`, `device_bytes`, `host_bytes`, `lowering`). Per-call host time on a 16³ call: bare ctypes `imd_contract` 3.9 µs, `contract_ptr` +0.05 µs, `Plan.contract` with CAI arrays +0.9–1.0 µs, `immerdink.contract` (cached plan) +1.9–2.1 µs (`bench/python_overhead.py`). Host queries on a hot path: `immerdink.planner` (DESIGN v1.11 §6e; `Planner.query_arrays` takes the device API's arrays / Layouts, 1.6–1.8 µs per workspace query vs 7.5–8.8 µs for `immerdink.workspace_size`). Test `python_device` (pytest, 26 tests, bit-exact vs the oracle incl. packed operands, threads, streams, the CAI v3 stream policy (default "check", opt-out "ignore"), plan-cache `clear()` / `freeze()`, errors; no CuPy / PyTorch needed) | usability from Python/other projects; ehrenfest integration tests | M | done |
| B2 ✅ | **Performance model** (done 2026-10-05, DESIGN v1.8 §6c, docs/NOTES-B2.md): libimmerdink_host `imd_perf_predict` / `imd_perf_predict_pack` / `imd_perf_rank` (host API v6; Python `host.predict`, `host.rank`) — per-phase and total time and TFLOPS-eq from features computed exactly from the lowering, linear in calibrated per-kernel parameters (K4: roofline of the A3 round model and DRAM bytes beyond an effective L2; a data-dependent eps-path fraction with a typical-data default and an exact bracket); ranking of N × g × vec mode × packed operands inside a memory budget with exact footprints and plan-time statuses. Shipped RTX 4090 calibration (`key = value` file, embedded), `tools/calibrate.sh` for another GPU (~12 min of GPU time). Hold-out (114 calls, disjoint shapes): median / p90 error 3.2 / 8.4 % on ≥ 1 ms calls, 5.1 / 17.5 % at 100 µs – 1 ms, 11.7 / 19.3 % below 100 µs (all targets met); e.g. sq8192 N = 14 37.05 TFLOPS-eq measured vs 37.78 predicted. Open: KF at N ≥ 15 on mid-size shapes 20–30 % low. **Fidelity follow-up** (2026-10-05, DESIGN v1.10 §6c, host API 7, calibration format 2, branch `b2-fidelity`, NOTES-B2 §6): wave / occupancy terms from launch rules shared with the kernels (measured CTAs per SM; KL / K2 / K3 memory waves, K5 DRAM term; K4's slow chunks are k̂ mod 32 = 16, unaligned residue rows), BURST / SUSTAINED regimes (power-cap model, default SUSTAINED; sustained windows median error 1.5 %), COLD / WARM cache state (COLD +4–16 % on L2-sized calls; WARM ≈ REPEAT), 80 % prediction bands (2.5 / 97.5 hold-out quantiles, 84 % cross-validated coverage) and rank ties within a fitted time-ratio window. Post-BIOS recalibration: hold-out ≥ 1 ms 1.6 / 6.2 % (median / p90), 100 µs – 1 ms 6.3 / 15.2 %, < 100 µs 6.8 / 13.4 %, all targets met. Reviewed 2026-10-05: default rank order TIME, ties opt-in. Later (NOTES-B2 §6.8): the K4 alignment kernel fix (K_c' = 131040 and k̂ padded to a multiple of 32: half of all k̂ pay 28–38 % in K4; a layout change, so DESIGN and TLA+ first), a settled-clock / hot calibration point, a clock-scaled e_byte. **Clock-scaled byte energy** (2026-10-06, DESIGN v1.15 §6c.7, host API 11, branch `hotclock`, NOTES-B2 §9.1): optional `power.byte_f_pj` (absent = 0, bit-identical), folded view for v7–v10 callers, `perf_fit --power-form scaled`, 60 s pre-heat sustained windows measured. **Combined refit** (2026-10-06, branch `followups-combined`, NOTES-B2 §10): eps path retired (DESIGN v1.15 §6c.1), shipped calibration refit on the v1.14 kernels with `power.byte_f_pj` 301 pJ/B, e_tc 0.708 pJ/op; hold-out targets met in every size class (as shipped before, the < 1 ms classes were missed on the new kernels). Open: recollect all 14 sustained windows (all ran pre-v1.14 commit 0787164) at the next sustained run; the hold-out gain is almost entirely the eps retirement (the refit is worse than old θ on < 100 µs calls; KF N = 8 and K3 predicted fast). **Model fix** (2026-10-06, DESIGN v1.17 §6c, host API 12, branch `model-fix`, NOTES-B2 §11): additive optional features `kf.elem` (KF per-element chain) and `pack.dram` with the PACK L2-hit rule (`warm.pack_l2_bytes`), format 2 kept, f_ref by parabola vertex with a nested LOO figure (1.19 %); full refit with all 14 sustained windows recollected: hold-out < 100 µs 3.8 / 12.9 %, 100 µs – 1 ms 2.6 / 9.6 %, ≥ 1 ms 2.2 / 6.6 % (median / p90), K3 phase 13.3 / 22.4 → 4.9 / 14.7 %. Open: KF roofline, W = 3 KF on tiny shapes, K4 on tiny shapes, capped-clock bias and r_mm power. **Hot-path planner** (2026-10-05, DESIGN v1.11 §6e, branch `planner-fast`): `immerdink.planner` (`Query`, `Planner`, opt-in `Memo`, lazy result views), bit-identical to `host.py`, per query 0.8–1.8 µs instead of 8–20 µs (C floor 0.25–0.86 µs), `rank` in one C call, a memo hit 0.6–1.0 µs; pure ctypes, no C API change, no nanobind (`bench/planner_overhead.py`, test `host_planner`). **Accuracy-aware ranking** (2026-10-05, DESIGN v1.12 §6f, host API 8, branch `b2-accuracy`, NOTES-B2 §7): `imd_err_estimate` / `imd_err_min_moduli`, predict `err_bound` / `err_typical`, rank flags candidates missing an error target as infeasible, so `rank[0]`, if feasible, is the cheapest configuration meeting it. Metrics normwise (no statistics needed), elementwise (ρ), absolute (norm bound); optional ℓ1/ℓ2 ratios tighten the bound up to ~9× on decaying / wide data. Rigorous bound = PROOFS L3c (L3a composed with the final rounding; subnormal-safe ABSOLUTE term), reference `params.error_estimate`, 0 violations in 300 held-out GPU cases (k ≤ 9.4e4, φ = 0.5–4, 2^±300 scaling). Opt-in typical estimate (Hoeffding, RN): 216 / 216 random-mantissa cases covered at the default p = 2⁻⁴⁰ (max 0.19 of it in the emulation regime), ~⅔ modulus cheaper than the bound; not valid for rows of equal entries (up to 20× above). Python `host.Accuracy`, `planner.Query.min_moduli` (~1.7 µs). Reviewed 2026-10-05 (NOTES-B2 §7.4) | callers choose N, g, vec_mode by cost; ehrenfest planning | M | done |
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
