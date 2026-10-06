# NOTES-K4ALIGN — K4 load alignment (DESIGN v1.13 §4b, ROADMAP A6)

## Status

Stage 0 and stage 1 are done (2026-10-05):
- **Stage 0:** one `k_hat_of(k, A)` in libimmerdink_host, with A = 16.
- **Stage 1:** K_c' = 131008.

Stage 2 is designed in DESIGN §4b (final, 2026-10-05; not yet implemented): k̂ = ⌈k⌉_A,
A = 64 iff k > `K_PAD64_MIN` = 96, else 16; pack format 2; host API 9 (`imd_lowering_t
.k_align`); the test hook `imd_test_set_k_align`. The threshold was frozen from the sweep
in §8. The rulings of the independent review (2026-10-05) are binding and listed below.
The TLA+ models (K4Align, Pipeline and PackedOperand additions) and the certificates are
in DESIGN §4b.7; the test plan in §4b.8.

The measurements in §2–§5 were taken before any change (origin/dev ca7141c), and the A/B in
§6 after it. All used the profiling build of `imd_bench` and the repo's gated runner (quiet
gate, shared GPU lock, warm-up), only while the machine was otherwise idle. The raw records and scripts are outside the product repo
(immerdink-eval `results/k4align`).

## 1. Mechanism

K4 is CUTLASS 2.x `DefaultMma`: int8 A row-major and B column-major, alignment 16, threadblock
K tile 64.
- RA/RB rows have stride `lda = ldb = k̂`, and the region and pack bases are 256-aligned.
- A launch covers the K-chunk `[k0, k0 + kc)`.
- CUTLASS's predicated tile iterator loads the chunk's **residue tile** (`kc mod 64`
  elements) **first** (`predicated_tile_access_iterator.h:204-213, 514-523`).
- Every later 64-element main-loop load of row r therefore starts at phase

  `φ(r, c) = (r·k̂ + e_c) mod 64`, where `e_c = k0 + kc` is the chunk end.

The phase depends on the stride and the chunk end together, not on the stride alone and not
on the chunk start. This one rule fits every case measured, including the two-chunk test
that NOTES-B2 §6 called inconclusive.

## 2. K4 time per MAC by `k̂ mod 64` class (measured)

- Shapes: 2048² and 16384×512.
- N = 8 and 14.
- k̂ = 992–1072 and 4064–4128.
- Values are relative to class 0, averaged over both k̂ bands. The best K4 rate in class 0 is
  about 620 TOPS-eq at k̂ = 4096.

| config | class 32 | class 16 | class 48 |
|---|---|---|---|
| 128×256 (the rule's usual choice) | 1.03 | 1.32–1.34 | 1.37–1.44 |
| 128×128 | 1.10–1.11 | 1.55–1.58 | 1.61–1.73 |
| 64×128 | 1.16–1.17 | 1.69–1.73 | 1.81–1.92 |
| 64×64 | 1.15–1.19 | 1.69–1.76 | 1.83–1.94 |

- N and the shape barely change the ratios.
- A zero tail costs nothing: k = 1016 (k̂ = 1024) runs like k = 1024.
- Class 32 is also slow on the three smaller configurations (+10 to +19 %), so the target is
  64, not 32.
- Example (2048², N = 14, 128×256):

  | k̂ | K4 throughput | K4 time | call throughput |
  |---|---|---|---|
  | 4096 | 618 TOPS-eq | 0.778 ms | 26.3 TFLOPS-eq |
  | 4080 | 435 TOPS-eq | 1.102 ms | 21.0 TFLOPS-eq |

- Measured but **not explained** by the phase rule:
  - class 48 is 5–15 % slower than class 16, although both have the same set of phases;
  - the sector multipliers are 1.5× with half of the rows misaligned and 2.0× with all of
    them (a 64-byte segment touching 3 sectors instead of 2 predicts at most 1.5×).

## 3. Padding emulated by running the padded k (per useful MAC)

| case | K4 throughput, k / padded (TOPS-eq) | K4 gain | call throughput, k / padded (TFLOPS-eq) | call gain |
|---|---|---|---|---|
| 2048², N = 14, 128×256, 4080 → 4096 | 435 / 616 | +42 % | 21.0 / 26.2 | +24 % |
| 2048², N = 14, 128×256, 1008 → 1024 | 380 / 548 | +44 % | 15.6 / 19.6 | +25 % |
| 2048², N = 14, 128×256, 1040 → 1056 (to 32 only) | 430 / 525 | +22 % | 16.3 / 18.0 | +10 % |
| 2048², N = 14, 128×256, 4064 → 4096 (class 32) | 610 / 613 | +1 % | 26.0 / 26.1 | 0 % |
| 16384×512, N = 8, 128×256, 4080 → 4096 | 418 / 595 | +42 % | 25.6 / 30.0 | +17 % |
| 2048², N = 14, 64×64, 4080 → 4096 | 170 / 311 | +83 % | 10.2 / 16.4 | +61 % |
| 2048², N = 14, 64×64, 4064 → 4096 (class 32) | 270 / 310 | +15 % | 14.9 / 16.3 | +10 % |

## 4. ncu, single K4 launches (2048², N = 8; ncu locks the clock at about 2.45 GHz)

| case | µs | tensor pipe | L2 requests | L2 sectors | smem store wavefronts |
|---|---|---|---|---|---|
| 128×256, k̂ 1024 | 134.7 | 80.6 % | 6.29 M | 12.58 M | 4.46 M |
| 128×256, k̂ 1040 | 180.2 | 62.7 % | 12.19 M (1.91× per k) | 19.27 M (1.48× per k) | 8.70 M |
| 64×128, k̂ 1024 | 177.4 | 60.1 % | 12.58 M | 25.17 M | 10.88 M |
| 64×128, k̂ 1056 | 221.9 | 50.6 % | 16.52 M (1.27× per k) | 25.95 M (same per k) | 14.62 M |

- DRAM reads are flat (34 MB) and the L2 hit rate is 96–98 %. The loss is in L1/L2 request
  and sector counts and in shared-memory store wavefronts, which stall the load pipeline.
- In class 32 the sector count is unchanged but requests rise by about 27 %. That fits loads
  crossing a 128-byte line, but it stays a hypothesis until it is profiled directly.

## 5. Long k: the chunk length (before stage 1, K_c' = 131056 ≡ 48 mod 64)

All runs are 2048² × k, N = 8, 128×256.

| k̂ | chunks | chunk-0 rows misaligned | call TFLOPS-eq | call ms | K4 TOPS-eq | K4 ms | chunk-0 L2 sectors |
|---|---|---|---|---|---|---|---|
| 131040 | 1 | none | 33.4 | 32.9 | 529 | 16.6 | 1.00× |
| 131056 | 1 | half | 29.4 | 37.4 | 427 | 20.6 | 1.50× |
| 196608 | 2 | all | 23.4 | 70.5 | 289 | 45.7 | 2.00× |
| 196624 | 2 | half | 27.5 | 60.0 | 378 | 34.9 | 1.50× |
| 196640 | 2 | all | 25.9 | 63.7 | 350 | 37.6 | 2.00× |

- **196608, an aligned power-of-two stride, is the worst case.** Its chunk 0 (131056 long) ends
  at 48 mod 64, so every row is misaligned; chunk 1 ends at k̂ and is aligned.
- That is the residue-first phase, **not a DRAM-channel effect**. NOTES-B2 §6 is corrected.
- These launches run at 67–74 % of DRAM peak. Misaligned loads raise their DRAM traffic too
  (chunk 0: 12.0 GB aligned against 25.5 GB all-misaligned).

## 6. Stage 1: K_c' = 131008

**The rule.** `K_c' = ⌊K_c/64⌋·64 = 131008 = 2047·64` replaces 131056.
- Every full chunk then has `k0 ≡ kc ≡ 0 (mod 64)`, so its phase is `r·k̂ mod 64`.
- The last chunk ends at k̂.
- The phase class therefore depends on k̂ alone, and no chunk gets worse. A k̂ ≡ 0 (mod 64)
  is aligned in every chunk.
- 131040 would leave full chunks in class 32.

**Exactness.**
- `128²·131008 + 255 = 2146435327 ≤ 2³¹ − 1` and `−128·127·131008 = −2129666048`.
- Chunking does not change W or C, so results are bit-identical.

| check | result |
|---|---|
| `spec/params.py --check` | cert_kc and cert_row_phase pass; the boundary k list covers 131007–131073, 196608 / 196624, 262016 / 262017 / 262112 / 262113 / 262144 / 262160, 393024 / 393025 and 524032 / 524033 |
| Apalache `Epilogue` | `Kp = 131008` |
| pre-check on the old code with `imd_test_set_limits(k_chunk = 131008)` | 11 long-k cases (k = 131009 … 262212, N = 2–14, RN and TRUNC) bit-exact against the oracle at chunk 131008, and identical to the production-chunk run |

**What changes.** The chunk count of `k̂ ∈ (j·131008, j·131056]` goes up by one. Each such
call does one more add-mod launch per group and part and one more W read-modify-write
(`N·b·m·n̂` bytes; about 30–40 µs at 2048², N = 8), and may need one more add-mod shape key.
Workspace and pack layout, pack format, host API version and calibration format do not
change. The perf-model values (`imd_perf_predict` / `imd_perf_rank`) do change for every
multi-chunk call (k̂ > 131008): chunk sizes and round counts, the format-2 slow-chunk set, and
the format-1 rule (full chunks are no longer slow); see the error figures below. The
"+1 chunk" holds while the windows are disjoint (j ≤ 2729, k̂ ≲ 3.58·10⁸).

**Measured after the change.** `bench/run_ab.sh` ran the stage 0 build (K_c' = 131056) and the
stage 1 build (131008) interleaved per case, gated and warmed as above. The 2048² rows use
`--k4 0` (128×256); the 512² rows use the rule's 128×128. N = 8.

| case | k̂ mod 64 | chunks before → after | call TFLOPS-eq, before → after | call ms | K4 ms |
|---|---|---|---|---|---|
| 2048² × 131040 | 32 | 1 → 2 (window) | 33.3 → 33.6 | 33.0 → 32.7 | 16.61 → 16.35 |
| 2048² × 131050 (k̂ 131056) | 48 | 1 → 2 (window) | 29.3 → 29.6 | 37.5 → 37.2 | 20.60 → 20.78 |
| 2048² × 131056 | 48 | 1 → 2 (window) | 29.3 → 29.6 | 37.5 → 37.2 | 20.77 → 21.34 |
| 2048² × 196608 | 0 | 2 → 2 | **23.5 → 32.0** | 70.1 → 51.5 | 45.08 → 25.92 |
| 2048² × 196624 | 16 | 2 → 2 | 27.5 → 28.0 | 59.9 → 59.0 | 34.41 → 33.43 |
| 2048² × 196640 | 32 | 2 → 2 | **25.8 → 31.8** | 63.9 → 51.9 | 37.98 → 25.98 |
| 512² × 262144 | 0 | 3 → 3 | 11.1 → 11.8 | 12.3 → 11.7 | 3.64 → 2.97 |
| 512² × 262160 | 16 | 3 → 3 | 11.4 → 11.5 | 12.1 → 11.9 | 3.37 → 3.24 |

- The 196608 call is 36 % faster in TFLOPS-eq, and K4 drops by 43 %, close to the projected
  23–25 ms.
- Class 32 (196640) gains 23 %. Odd multiples of 16 are unchanged, as the rule predicts.
- In the windows the extra chunk costs nothing measurable at 2048²: the call is within
  1 %, and K4 moves by −2 to +3 %.
- Timed-loop spread was ≤ 3.4 % for every record.
- The perf model's error on these calls (still θ fitted with 131056) changes from −38 / −32 %
  to −20 % on 196608 / 196640. The θ refit is an open item (review ruling: no format bump,
  refit θ), to be done with stage 2.

**Perf model.** `mac_ktail` format 2 ("k̂ mod 32 ≠ 0, or c·K_c' mod 32 ≠ 0") reduces exactly
to "k̂ mod 32 ≠ 0", because c·K_c' ≡ 0. The rule's chunk counts in the windows change with
`k_chunks`. Review ruling: no calibration format bump; refit θ (fitted with 131056). The
refit is an open item, to be done with stage 2.

## 7. Review rulings (2026-10-05, binding)

- **Stage 2 (AUTO).** A ∈ {16, 64} is a pure function of k, defined in `spec/params.py`. It
  is not chosen by the perf model, because sizing stays GPU-free and independent of the
  calibration.
  - `K_PAD64_MIN` is frozen from a sweep of k̂ = 80…1024 (step 16) on four configurations
    and two or three shapes, including a DF-CC shape.
  - There is no `imd_options_t` field: that struct has no `struct_size`, so a new field
    would break v8 callers.
  - A test hook forces A, and `imd_lowering_t` reports the resolved `k_align`.
- **Packs.** No pack override to force A = 16. The pack FORMAT word carries A (1 = A16,
  2 = A64).
- **Calibration.** No format bump; refit θ.
- **W/n̂ stride.** Excluded and unmeasured; a later item.
- **Stage-2 tests:**
  - a TLA+ model of CUTLASS's residue-first address sequence (absolute addresses ≡ 0 mod 64;
    `SectorK` renamed to `TileK`);
  - a poisoned-workspace test over the AUTO range for every producer path (KF at k ∈ {600,
    1000}; KL at k ∈ {1040, 4080}; K2/K3; packed A and B with the B-role tail; two-chunk k in
    131009–131072);
  - accuracy and κ paths identical across A.
- **Deferred.** The realign-kernel alternative stays one deferred paragraph. The ncu check of
  class 32 on 64×64 is wanted, but not as a gate.

### 7a. Adversarial review of the stage-2 design (2026-10-05, applied)

The design (d60201f) was judged sound to build once nine must-fixes were applied; they are
folded into DESIGN §4b.3–§4b.11, §6a and §6b.5 (changelog "Adversarial review"):
- the format lives only in the handle key; the hazard is **caller-persisted pack bytes**
  (v8 format-1 bytes in a v9 format-2 handle: silent wrong stride), so callers persisting
  bytes persist `format` and `pack_bytes` and re-pack on a difference;
- A is resolved **once per API call** (`analyze`, `pack_lower`) and passed explicitly
  (`WorkspaceShape`, `pack_layout`, `ExecPlan.k_align`); libimmerdink never resolves;
- the hook is one out-of-line atomic in libimmerdink_host (hidden visibility would
  duplicate a header static per `.so`);
- PACK_MISMATCH: an unknown format word first, the 1-vs-2 difference **last** (k = 96 pack,
  k = 97 plan: EXTENTS), against the plan's own format; the oracle does this now, and
  PackedOperand models the order (`neg_reason_fmt_first` fails);
- `k_hat_of` defaults to the policy and `PACK_FORMAT` is derived, so the policy flip is one
  switch; the test call sites that rely on a default are listed in DESIGN §4b.8;
- initcheck on unpoisoned fresh allocations; the [k, k̂) zero check over KL and K2/K3;
- the stage-1 window tests pin A = 16; 196608 and 262017 are added;
- with A = 64 the last add-mod shape key can merge (k ∈ (n·K_c' − 64, n·K_c' − 16], n ≥ 3,
  e.g. 393008: plan host bytes differ) and the 256 MiB `k4_tune` cap can be crossed;
- `host.test_set_k_align` clears the Python plan cache.
The review's three questions are resolved in DESIGN §4b.11 (reject; hook in
libimmerdink_host; FORMAT after the other words). Full text: immerdink-eval
`findings/k4align-s2-review.md`.

## 8. The AUTO threshold `K_PAD64_MIN` (sweep, 2026-10-05)

Setup: origin/dev ca7141c, profiling `imd_bench`, the gated runner (quiet gate, shared GPU
lock, warm-up); `ehrenfest_busy.sh` FREE before every batch; about 13 GPU-minutes. k = 16…1024
step 16, N ∈ {8, 14}, the four K4 configurations forced plus the rule; shapes 2048²,
16384×512, a DF-like `Qia,Qjb->iajb` (i = j = 24, a = b = 96: 2304² × Q) and 512² (rerun
with 300 reps; the first pass had 5–45 % spreads). A = 64 is emulated by running
k′ = ⌈k⌉₆₄ and counting the logical 2mnk as useful work (KF then reads k′ real columns, so
the gains are slightly understated). Raw data and scripts: immerdink-eval
`results/k4align/thr/`.

**Result: `K_PAD64_MIN = 96`** (A = 64 iff k > 96, i.e. ⌈k⌉₁₆ ≥ 112).

| threshold | mean call-time loss vs the per-k best (8 shape × N cases) |
|---|---|
| 0 (always pad) | 0.62 % |
| 64 | 0.36 % |
| **96** | **0.30 %** |
| 128 | 0.35 % |
| 192 | 0.47 % |
| 512 (the draft) | 1.30 % |
| never pad | 3.80 % |

- k̂ ≤ 64 has no main loop, so padding only costs there: k = 16 −6 to −12 %, k = 32 −1 to
  −9 %, k = 48 about 0.
- By `k̂ mod 64` (rule configuration, 128×256 on the large shapes): class 48 (16 columns of
  padding) wins from k ≈ 100 on every shape (+3 % near 112, +12–14 % above 512); class 16
  (48 columns) loses below k ≈ 250 and gains +7–8 % above 512; class 32 is about 0 at every
  k under 128×256 (the smaller configurations gain 5–20 % there).
- 96 is the per-case optimum in 5 of 8 cases and within 0.03 pp in 7 of 8; 16384×512 at
  N = 14 prefers 288 (+0.25 pp; it loses 5–6 % at k = 144 and 160).
- Worst single-k loss under 96: 2.3–6.8 % on the large shapes (k = 144–208), 10–12 % on 512²
  (k = 144 / 160, where the rule's configuration changes).
- Against never padding, the average call over k = 16…1024 is +2.8 to +5.6 % faster in
  useful TFLOPS-eq. Example, 2048², N = 14: k = 1008 runs at 18.24 instead of 15.70
  TFLOPS-eq (0.464 instead of 0.539 ms, +16 %); k = 112: +3.4 %.
- A rule that also looks at k̂ mod 64 reaches 0.26 %: not worth a second parameter.
- Forced configurations: 64×64 and 64×128 do best with thresholds of 16–96; 96 costs them
  at most 0.65 %.
- **Overhead (corrected):** per residue row A = 64 adds δ = ⌈k⌉₆₄ − ⌈k⌉₁₆ bytes, at most
  1/3 of ⌈k⌉₁₆ (k = 129…144: 144 → 192), +14 % at k = 97…112 (112 → 128), at most 4.6 % for
  k > 1024. The MMA work and the launch structure do not change (`params.cert_k_align`).
  (The sweep note's "43 % at k = 97–112" was 48/112, a slip.)
- **ncu, 64×64, 2048², N = 8:** class 32 is the 128-byte line crossing. k̂ = 1056 against
  1024: L2 requests 1.27× per k, sectors 1.000×, tensor pipe 46 → 38.5 %, time per k 1.22×
  (one segment in four crosses a line: +25 % predicted). Class 16 (k̂ = 1040): requests
  1.91×, sectors 1.51× per k, time per k 1.74×. DRAM reads flat.
