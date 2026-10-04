# D1 notes: packed-operand reuse

ROADMAP D1. Design: DESIGN v1.7 §6b (approved by Eric 2026-10-03, every §6b.10
recommendation), implemented as written; §6b.11 lists the choices the design left to the
implementation and one clarification of §6b.5 to confirm. Host API version 5.

## 1. What was built

**Spec (before code).**
- `spec/params.py`: `pack_layout`, `pack_ws_bytes`, the packed workspace
  (`workspace_size` / `region_offsets(packed=...)`) and the certificate `cert_pack_layout`
  (768 layouts: region order and alignment, every K4 pointer into a pack 16-byte aligned for
  every plane / batch / row / K-chunk, the B-role reads ending exactly at the tail, overflow,
  hand-computed examples). 26/26 certificates.
- `spec/oracle.py`: `PackDesc`, `pack_lower` (statuses, AUTO / GIVEN orders, the X-only fused
  modes, the key), `pack()` (every byte of a pack), `pack_desc_for`, `lower(orders=...)`
  (adopted orders), `lower_packed` (PACK_MISMATCH reasons, adoption by mode position, slot
  keys) and `contract_tensor(packs=...)` (σ, flags and residues read from the packed bytes).
  `spec/test_oracle.py` (10): pack bytes == the in-line run's σ / flags / residues with zero
  padding, tail and gaps; packed == unpacked in the adopted order on random contractions
  (one or both packed, both roles, RN / TRUNC, every N range, g, Inf / NaN), == the plain
  contraction for `pack_desc_for` packs and for unflagged outputs of AUTO packs; reuse
  across partners / C layouts / roles; A7 through a pack (the native path follows the
  adopted K order: NaN with X's order, −Inf with the plan's); every mismatch reason and pack
  status.
- TLA+ (IndexMap adopted orders, Pipeline packed slot, HostAPI pack handle): see §4.

**libimmerdink_host** (`src/host/pack.{hpp,cpp}`, `analysis.cpp analyze_packed`,
`index_map.cpp adopt_order`): `imd_pack_layout`, `imd_pack_desc_for`,
`imd_workspace_size_packed`, `imd_workspace_layout_packed`, `imd_lowering_packed`
(`imd_lowering_t` v5: `packed`, `pack_mismatch`), `IMD_STATUS_PACK_MISMATCH` (8), the key
and its comparison with reason codes. The plan adopts the packs' K / L / row orders
(fusion redone); KF and KL scale and pack the in-line operand only; launch counts and plan
host bytes (+ the copied keys) of the packed plan. Python: `PackSpec`, `pack_layout`,
`pack_desc_for`, `packing=` on `workspace_size`, `workspace_layout`, `lowering`.

**libimmerdink** (`src/api.cpp`, `src/contract.cu`, `src/gemm_k4.cu`):
- `imd_pack_create` / `destroy` / `bytes`: an immutable host object (X's descriptor for the
  position map and imd_pack's aliasing check, the pack lowering and key, imd_pack's launch
  data); no device work.
- `imd_pack`: the §6b.5 checks, then two launches on the caller's stream: `kp_prep` (X's
  row / k̄ / batch scatter vectors into the pack workspace, `8·(r + k + b)` bytes; the two
  alignment gaps, the tail and the last gap of the buffer zeroed) and **the v1.6 KL kernel**
  on X as a one-operand view (rows m = r, g = N so every plane is "group 1", rows_pad = r):
  σ, the flags and all N planes, zero K padding, exactly the bytes K1 + K2/K3 write (v1.6).
  Reading along rows, short rows get taller KL tiles (up to 64 rows). No allocation, no
  synchronization, capture-safe.
- `imd_plan_create_packed`: libimmerdink_host's `analyze_packed` (keys, adoption), then as
  `imd_plan_create`; the plan copies the slot keys.
- `imd_contract_packed`: the §6b.5 checks (with the §6b.11 reading of step 1), then
  `enqueue_contract` with the packed buffers: a packed operand has no K1 / KL / K2 / K3
  launch; K4 reads planes g₀.. of the pack at batch stride `r·k̂` (both roles), K5 reads its
  σ and flags; KF tiles / KL bits only for the in-line operand. `imd_contract` on a packed
  plan returns PACK_MISMATCH.

## 2. Tests (`IMMERDINK_CUTLASS_DIR=... tools/test.sh`: 32/32)

| test | what |
|---|---|
| `host_c_api` | v5 struct sizes; `imd_pack_desc_for`, `imd_pack_layout` (worked example 8192 B), AUTO == GIVEN key, INVALID / NOT_SUPPORTED, packed workspaces (28416 / 25600 / 23296 B), `imd_lowering_packed` (launches, KF, MODULI / ROUNDING / CLASSES) |
| `host_python` | on 1000 random descriptors × 6–8 packings (desc_for, AUTO, A / B / both, wrong N, wrong rounding, negated strides): wrapper == direct C == oracle (`lower_packed`, `pack_lower`) + params (layout, workspace, regions); launch counts and fused paths from an independent formula; reasons SUCCESS / MODULI / ROUNDING / CLASSES / STRIDES / ORDER all reached |
| `packed_packed` | 111 contractions, 1047 runs: every packed byte vs `oracle.pack` (guard zones around the buffer and imd_pack's workspace), C bit-exact vs the oracle's packed path in both vec modes, fused by rule / unfused / no small variants / KL forced / KL 64-row tiles, every K4 configuration and k4_tune; zero CUDA allocations in imd_pack and imd_contract_packed; launches == the plan's; desc_for packs == the plain oracle; random matrices (A / B / both, swapped roles, N ∈ {2, 3, 8, 9, 14, 16, 20}, RN / TRUNC, g, α / β, Inf / NaN), batched / permuted / DF / diagonal + X-only / extent-1 / neg-pad strides tensors, DIRECT mode, α = 0, k = 0 (0-byte pack), empty output, k̂ > 1024 (KL rule), the K4 batch split (b = 3300, N = 20), every g = 1..7, and the native-path order through a pack (AUTO: NaN; desc_for: −Inf like the plain call) |
| `packed_reuse` | one packed buffer read by 24 plans (4 partners / C layouts, X as the caller's A and B: both post-swap roles, g ∈ {1, 3, 9}, both vec modes), bit-exact; the buffer unchanged |
| `packed_api` | pack creation statuses; every PACK_MISMATCH reason through `imd_plan_create_packed` and `imd_lowering_packed` (MODULI, ROUNDING, CLASSES ×3, STRIDES, EXTENTS, ORDER); `imd_contract_packed`: renamed-label / GIVEN handles with the same key accepted, NULL operand, packed pointer or handle on an in-line slot, packed slot without / with another handle, NULL packed read / not read (α = 0), ALIASING (C or ws vs packed; packed over A allowed), MISALIGNED, short packed / ws; `imd_contract` on a packed plan; `imd_pack`: NULL X / packed / ws, ALIASING ×3, MISALIGNED ×3, short ×2, an empty pack with NULL pointers |
| `graph_capture` | imd_pack + imd_contract_packed captured into one graph (global mode), replayed twice with the packed buffers poisoned in between: A / both / B packed (KF on B, nothing, KL on A), a batched tensor, both vec modes; == the plain imd_contract |
| `threads` | one packed buffer read concurrently by 8 threads on own streams: a shared packed plan with the shared handle and with per-thread handles (same key), per-thread plans created concurrently from the shared handle; per-thread plans and handles destroyed as soon as their calls returned; == the plain sequential result |
| `host_vs_plan` | 290 real packed plans (desc_for / AUTO, A / B / both) == `imd_lowering_packed` / `imd_workspace_layout_packed`: view, layout, launches, fused paths, packed slots, pack bytes, host payload incl. the copied keys |
| `c_api` | the packed API from C |

## 3. Measurements

Tooling: `imd_bench --methods packed` (bench/imd_bench.cu), `bench/sweep_d1.txt` through
`bench/run_sweep.sh` (quiet gate, GPU lock, the 3 s `warm_run.sh` warm-up, clock state
recorded); PCIe is not involved. Every record checks that the packed C equals the plain C
bit for bit (`imd_pack_desc_for` orders): all `bit_identical: true`.

**Reuse (`bench/results/d1-packed-323a21c.jsonl`).** Per repetition, in one process and
interleaved: `R × imd_contract` vs `imd_pack + R × imd_contract_packed` (same partner, same
C), plus each call alone. The ratio plain / packed of the medians (> 1: packing wins):

| case | N | packed (role) | imd_pack / one call | R = 1 | R = 4 | R = 16 | per call at R = 16 (pack subtracted) |
|---|---|---|---|---|---|---|---|
| sq4096 | 8 | A (A) | 0.322 / 2.730 ms | 0.94× | 1.12× | 1.14× | 3.098 → 2.700 ms (−13 %) |
| sq4096 | 14 | A (A) | 0.438 / 4.916 ms | 0.98× | 1.04× | 1.07× | 5.049 → 4.680 ms (−7 %) |
| sq4096 | 14 | B (B) | 0.439 / 5.290 ms | 1.09× | 1.05× | 1.09× | 5.096 → 4.643 ms (−9 %) |
| sq8192 | 8 | A (A) | 1.224 / 19.98 ms | 1.01× | 1.06× | 1.09× | 20.60 → 18.85 ms (−8 %) |
| sq8192 | 14 | A (A) | 1.731 / 34.50 ms | 1.02× | 1.06× | 1.05× | 34.50 → 32.79 ms (−5 %) |
| t_cc | 8 | A (B) | 0.338 / 3.103 ms | 1.20× | 1.13× | 1.13× | 3.165 → 2.773 ms (−12 %) |
| t_cc | 14 | B (A) | 0.443 / 5.452 ms | 1.04× | 1.05× | 1.09× | 5.157 → 4.715 ms (−9 %) |
| t_df | 8 | A (A) | 0.171 / 1.521 ms | 1.00× | 1.14× | 1.13× | 1.821 → 1.601 ms (−12 %) |
| t_df | 14 | B (B) | 0.225 / 2.483 ms | 0.85× | 1.06× | 1.08× | 2.892 → 2.652 ms (−8 %) |
| r_skinny | 8 | A (A) | 1.208 / 2.316 ms | 1.00× | 1.69× | **1.98×** | 2.596 → 1.236 ms (−52 %) |
| r_skinny | 8 | B (B) | 0.066 / 2.735 ms | 1.21× | 1.03× | 1.02× | 2.574 → 2.531 ms (−2 %) |
| r_skinny | 14 | A (A) | 1.685 / 4.073 ms | 1.08× | 1.51× | **1.79×** | 3.940 → 2.090 ms (−47 %) |
| r_skinny | 14 | B (B) | 0.070 / 3.730 ms | 1.01× | 1.00× | 1.02× | 4.103 → 4.033 ms (−2 %) |

(All 18 rows, with the workspaces: docs/BENCHMARKS.md "D1". Clocks 2490–2805 MHz; the long
sequences run power-capped, `clocks_event_reasons` 0x4, which is why the per-call time in a
sequence exceeds a single call's.)

**Where the time goes (`bench/results/d1-phases-73b62ee.jsonl`, per-phase event marks of one
call, N = 14, ms):**

| | K0 | KL A | KL B | K4 | K5 | total | imd_pack |
|---|---|---|---|---|---|---|---|
| sq4096 plain | 0.005 | 0.435 | 0.437 | 3.432 | 0.887 | 5.276 | |
| sq4096, A packed | 0.009 | — | 0.434 | 3.268 | 0.431 | 4.139 | 0.437 (one KL with N planes) |
| r_skinny plain | 0.010 | 1.688 | 0.061 | 1.890 | 0.218 | 3.859 | |
| r_skinny, A packed | 0.008 | — | 0.065 | 1.628 | 0.218 | 1.920 | 1.690 |

- `imd_pack` is the KL launch of the packed operand with all N planes written (`pack KL`
  0.437 ms vs `KL A` 0.435 ms on sq4096; 1.690 vs 1.688 ms on r_skinny's A): at g = N the
  in-line KL also writes all N planes, so the pack costs one KL; the prep launch is ~5 µs
  (not marked separately). So R = 1 breaks even, and every reuse saves the packed operand's
  KL share of a call: 5–13 % on the square, CC and DF shapes (more at N = 8, where K4 is a
  smaller share), about half of r_skinny when its large operand (16384 × 4096) is packed —
  r_skinny is the shape where KL of A dominates (D4: K4 at its bound, the gap was K1 + the
  pack), and it reaches 1.79–1.98× at R = 16.
- At g < N the in-line path reads the operand once per residue group (⌈N/g⌉ reads, D4 §4
  "Residue groups"), while the pack is read only by K4: the saving grows as g shrinks (not
  in this sweep, which runs g = N).
- K5 in the plain sq4096 call: 0.84–1.0 ms in this session (dev's own build measured the
  same in a same-session A/B, `bench/run_ab.sh`), against 0.43 ms in earlier sweeps and in
  the packed call of the same process; a single-call phase effect of the GPU's state after
  long sequences, not a D1 change (the plain path's kernels and launches are unchanged; a
  workspace-offset sweep showed no address dependence). The R = 16 sequences, which run
  both sides in the same state, are the comparison to use.
- Workspace per contraction: the packed operand's sX, fX, RX are gone (25–33 % on the
  square / CC / DF shapes at g = N, 86 % for r_skinny's A); the caller keeps the pack
  (`N·b·r·k̂`, ≈ N/8 of the FP64 operand) for as long as it reuses it.

## 4. Spec (`spec/check.sh`: 284 models, all as expected)

- **IndexMap** (+10 pos / +8 neg): adopted orders (every invariant, EinsumPreserved
  included, on the adopted lowering for any order within a class), the slot key and its
  soundness (`PackKeySound`: equal keys ⇒ the pack's §6b.4 image is the plan's operand
  buffer, role B at batch stride r·k̂ with rows read from the next block or the tail),
  completeness (`PackComplete`, `DescForReproduces`), `PackNoOOB`, PACK_MISMATCH before
  κ ≥ T. Negatives: fusion not redone after adoption, lowering not adopted, plan not
  adopting, key fused over all carriers, key without strides, the v1 B-role stride, no tail.
  Findings (DESIGN §6b.11): `_ORDER` is implied by the key (a policy and reason code); the
  partner-only check is implied by the key except at k = 0 (harmless: such a plan never
  reads the pack); an extent-0 C-only label is in I (`pk_conly0`: a no-op plan rejects the
  A-side pack, as implemented).
- **Pipeline** (+9 pos / +8 neg): the packed slot (no K1 / KL / K2 / K3 for it, KF on the
  in-line operand only, sX / fX / RX absent, K4 reads the pack with its strides and tail, K5
  its σ / flags; RAW on the pack is the caller's). Negatives: the full workspace, K1 / K3
  still launched, K4 reading RX, K5 reading sB, the v1 B stride, no tail, a pack in flight.
- **HostAPI** (+3 pos / +7 neg): the immutable handle shared by threads, destroyed after
  return, a plan outliving it, a new handle with the same key; negatives: a lazily filled
  handle cache, a plan keeping a pointer to the handle, a GPU kernel reading it, two
  imd_pack calls into one buffer.

## 5. Open items

- **Confirm the §6b.11 reading of `imd_contract_packed` step 1**: a packed slot's NULL
  `packed` is INVALID_VALUE (step 2) only when it would be read; step 1 checks the handles
  (and that an in-line slot has neither `packed` nor `pack`). The literal "iff" would make
  step 2's packed null rule unreachable.
- AUTO orders change the partner's gather and C's scatter (§6b.2 "Performance
  consequence"); this sweep used `imd_pack_desc_for` packs (the plain orders). A sweep of
  AUTO packs on permuted shapes (where X's stride order differs from the plain plan's)
  would measure that cost.
- The packed-operand benefit at g < N (larger: the in-line operand is re-read per group) is
  not measured yet.
- Pack-format follow-ups belong to later items: an Ozaki-I slice format (C1(c)) would be a
  new `IMD_PACK_FORMAT`.
