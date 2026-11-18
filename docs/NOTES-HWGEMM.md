# Native FP64 contractions on tensor cores: design note (draft)

Status: **design settled for spec work**, 2026-11-16. Q1 and Q3 decided by Eric; Q2, Q4, Q5 and
the §4.5 API shape decided by the independent design reviewer (2026-11-16, binding unless Eric
overrules; §6). No code yet: TLA+ (§5) comes first.

## 1. What and why

immerdink computes FP64 tensor contractions by *emulation*: it splits each FP64 operand into
small integer residues (one per *modulus*, a small prime-like number), multiplies the residues
on INT8 tensor cores, and rebuilds the FP64 result with the Chinese remainder theorem (the
Ozaki-II scheme; DESIGN §3). That is fast where INT8 throughput is far above FP64 throughput,
as on the RTX 4090 (FP64 runs at 1/64 of FP32 there).

On Hopper (H100, H200) the picture is different: the FP64 tensor cores (DMMA instructions) are
fast, about 67 TFLOPS peak. Our own measurements show native DGEMM beating every emulator,
immerdink included, by 1.7–1.9× on small and batched shapes (sq1024, t_batch; eval findings
ozaki-spike-20261109). Large shapes still favour emulation.

Today the library has a native path, `IMD_DIRECT`, but it is a reference kernel: one thread
per output element, a sequential `fma` loop over k (`k_direct` in `src/contract.cu`, DESIGN §4
"DIRECT path"). It exists for K-empty contractions (k = 1, outer and Hadamard products) and for
callers who ask for native arithmetic, and it is far from tensor-core speed.

**Goal.** A native FP64 contraction path on FP64 tensor cores, driven by immerdink's own
contraction → GEMM lowering (the *IndexMap*, DESIGN §5), so that any contraction immerdink
accepts can also run natively at close to cuBLAS DGEMM speed. In effect, the native-FP64 part
of what cuTENSOR does, inside immerdink's existing plan / workspace / stream model.

**Who decides when to use it.** The caller. ehrenfest owns the emulate-or-native choice (Eric,
2026-11-16), so immerdink only needs to (a) run native when asked and (b) report its cost and
error bound through the existing host-side tables (`imd_perf_predict`, `imd_err_estimate`), so
ehrenfest can compare it with emulation. No auto-dispatch inside immerdink is planned.

**Out of scope.** Host-device transfers and orchestration (ehrenfest owns them), elementwise
and reduction operations, FP8 / FP4 paths (future work), and symmetric packed storage (ROADMAP
D3; CUTLASS's FP64 SYRK on sm_90 is noted in §4.4 for later).

## 2. Terms used here

- **Lowering**: the plan-time step that turns a contraction's index labels into a batched GEMM
  view: batch λ, rows ī, columns j̄, contracted k̄, each a list of *fused modes* (tensor
  indices merged into one GEMM index), with *scatter vectors* giving each index's memory offset
  (DESIGN §4, K0).
- **Affine operand**: an operand whose GEMM view (rows × K) is a plain strided matrix with
  stride exactly +1 in one dimension and stride ≥ 0 in the other (each GEMM index one fused
  mode, offset = row·s_r + k·s_k). CUTLASS can read it in place.
- **Packing**: copying an operand into a contiguous, GEMM-friendly layout in workspace first
  (as TBLIS does). The emulated path already packs (into int8 residues); here the pack would
  be an FP64 copy.
- **DMMA**: the FP64 tensor-core MMA instruction: `mma.sync` m8n8k4 on sm_80 / sm_89,
  m16n8k{4,8,16} on sm_90 and later. Hopper's `wgmma` has no FP64 form.
- **Slab**: a contiguous range of the contracted index K processed as its own GEMM, accumulating
  into C (§4.2), to bound the workspace.

## 3. Accuracy and reproducibility contract

Native FP64 is **not** correctly rounded, unlike the emulated path (DESIGN §3.7).

- **Error, no slabs.** `|D − x| ≤ γ_k·(|a||b|)_ij` per element (γ_k = k·u/(1 − k·u),
  u = 2⁻⁵³), for any summation order, under §6f's no-underflow hypothesis or with its absolute
  term; then α and β exactly as DESIGN §3.7 (`store_out`: E = α = 1 ? D : RN(α·D);
  C = β = 0 ? E : fma(β, C, E)). §6f already reports this bound for DIRECT.
- **Error, with S slabs.** Each product still passes at most k rounded additions, so γ_k still
  bounds the sum (Higham 2002, §3.1); no new lemma is needed for that part. What changes is α
  and β: slab 1 computes fma(β, C₀, E₁) with the caller's β, slabs 2..S add with β = 1, so the
  β·C₀ term is rounded S times (a γ_S·|β·C₀| term) and α ≠ 1 is applied S times. This is
  exactly DESIGN §3.8's K-segmentation composition (PROOFS L7), including its subnormal-band
  hazard (a slab product may be subnormal while x is not: carry the absolute term). The §6f
  estimator gets the L7 form for slabbed HWGEMM.
- **Bit identity with DIRECT (hypothesis to test).** Khattak and Mikaitis (arXiv:2512.07004,
  §4.1.2, Table 3) model FP64 DMMA as a sequential IEEE fma chain with round-to-nearest-even,
  tested on A100 and listed for H100 / H200 / B200 / RTX PRO 6000. If that holds, an unslabbed
  HWGEMM tile with the accumulator started at +0, K ascending, no sliced-K (warp K = CTA K), no
  split-K and the `store_out` epilogue gives **exactly DIRECT's bits** (`native_dot`,
  `src/contract.cu`). The first kernel test is a bit compare against DIRECT. If it holds, the
  non-finite caveat below disappears for unslabbed calls.
  **Confirmed on sm_89 (RTX 4090, 2026-11-17; `bench/hwgemm_bitid.cu`):** every `mma.sync`
  m8n8k4 f64 result equals the ascending fma chain from +0 on uniform, wide-exponent,
  cancelling, subnormal-product and signed-zero inputs (no flush to zero). The test
  discriminates: the reversed-order and one-rounding-per-instruction models mismatch on
  1–74 % of outputs. CUTLASS's Sm80 8×8×4 `device::Gemm` (64×64×16, 3 stages, TN and NN, k up
  to 4097 with a partial last K tile) is bit-identical to DIRECT's chain on every output.
  **Confirmed on sm_90 (H100 and H200, PACE embers/inferno, 2026-11-17)** by the same program
  built with `-DBITID_SM90`: the m16n8k4 instruction and CUTLASS's Sm90 16×8×4 `device::Gemm`
  equal the chain on every input family, layout and shape. cuBLAS 13.08 DGEMM was also bitwise
  equal to CUTLASS on all gate shapes, but cuBLAS does not promise its reduction order, so this
  is an observation, not a contract.
- **Reproducibility.** Bit-identical run to run for the same plan, `k_slab`, tile configuration
  and binary. Slabbed results are deterministic for a fixed `k_slab` but differ from unslabbed
  ones. No split-K of any kind by default (CUTLASS's serial split-K is deterministic but breaks
  DIRECT identity).
- **Non-finite inputs.** IEEE propagation; tests compare NaN-ness, not payloads. If the DIRECT
  identity fails, the order-dependent corner (DESIGN §3.7 oracle A7) can differ from DIRECT; with
  slabs that corner moves to the slab boundaries. Documented either way.
- **Subnormals.** A per-card unit test for each instruction actually used (m8n8k4 on sm_80 /
  sm_89, m16n8k4 on sm_90 / sm_120): subnormal inputs, products and accumulators.

## 4. Design

### 4.1 Where it plugs in

Same plan, same workspace query, same stream-ordered, graph-capture-safe call, no allocation
and no host sync inside `imd_contract` (DESIGN §4, kernel-scope rules); large-shared-memory
attributes are set at plan creation. New `mode = IMD_HWGEMM` (Q1), dispatch `HWGEMM`. K-empty
contractions keep `k_direct` (k = 1 is one fma); k = 0 and α = 0 keep the β-only kernel, run
once, never per slab.

**Int8 packs.** A HWGEMM plan follows DIRECT's rule exactly: a matching int8 pack is accepted
and ignored (the mode is not in the pack key, DESIGN §6b), the kernel reads `data`. No new spec.

### 4.2 Operand access (decided: reviewer, Q2)

**Per operand, at plan time:** an affine operand (§2) is read **in place**; any other operand
(negative strides, several fused modes in its rows or K, unit stride in neither dimension) is
**packed** by a gather kernel into a K-contiguous FP64 buffer in the workspace. Only the
non-affine operand is packed. The plan reports both bits in `imd_lowering_t` (struct_size
versioned). Layout variants N / T per operand are needed; the existing A/B swap halves them.
The pack layout follows whichever d1684 kernel layout pair measures fastest, not TN by
assumption.

**What packing costs.** A pack reads and writes the operand once: t_pack / t_gemm ≈ 8·(F/BW)/h
with h = mn/(m + n), F the achieved FP64 rate and BW the DRAM bandwidth. That is about 160/h on
an H100 (60 TFLOPS, 3 TB/s) and 110/h on an H200 (4.4 TB/s): under 5 % only for h above about
3200 / 2200 (squares above about 6400 / 4400), and about 30 % / 21 % at sq1024 (h = 512) before
gather coalescing or L2 residency. So packing is a real cost exactly where native beats
emulation. (The first draft's figure was off by about 8×.)

Consequences:
- **Fully affine A, B and C** (the GEMM-shaped common case) is the primary path: no K0, no vec
  region, workspace 0, one launch.
- **Main-loop gather** (CUTLASS examples 36 / 52) is a planned follow-up, chosen by the perf
  model, not a last resort. Its real scope: K must be one unit-stride mode, rows arbitrary,
  through an **int32** row-index vector (a new limit to check).
- **affine2** (non-unit stride in both dimensions, example 18) is dropped from v1: FP64 loads
  are 64-bit anyway (`default_mma_core_sm80.h`), the loss is sector efficiency, and example 18
  is Ampere-only. Such operands are packed.
- A GPU-free check before the kernel work: run `imd_lowering` on ehrenfest's real descriptors
  and count how many operands are affine.

**Batch.** One fused batch mode with a per-operand stride (0 allowed, for broadcast) maps to
CUTLASS's batched GEMM. A multi-mode batch reads its base offsets from the λ scatter vectors
(`a_bat`, `b_bat`, `c_bat`) once per CTA on blockIdx.z, which is graph-safe. **Never a host loop
over λ** (b launches would kill t_batch). grid.z ≤ 65535: chunk b deterministically.

**Output.** An affine C with a unit stride takes a plain strided store (swap A / B if C is
transposed); otherwise the epilogue stores through `c_row` / `c_col`. Both use HWGEMM's own
output functor with `store_out`'s rounding, **not** CUTLASS's `LinearCombination`
(fma(α, acc, β·C) rounds differently).

**K-slabs (Q3: in v1).** `k_slab` = 0 means no slabs. Otherwise K is processed in slabs of
`k_slab` (quantized to the GEMM K tile, so footprint and bits are well defined); each packed
operand needs `8·b·rows·k̂_slab` bytes (k̂_slab = the largest slab padded to the K tile, rows = m
or n̂), reused per slab, plus the vec region when any operand or C needs scatter vectors. Slab
s + 1's pack runs only after slab s's GEMM has read the buffer (stream order; checked in TLA+).
In-place operands need no workspace, so a fully affine call never slabs for memory (but may for
k > 2³¹ − 1, below). Slab 1 uses the caller's β with the C-not-read rule (β = 0: C is not
read); slabs 2..S use β = 1. k = 0 and K-empty never slab.

**Limits** (IndexMap status row 5 for HWGEMM): m, n ≤ 2³¹ − 1 (HWGEMM does not pad n, so n, not
n̂) and **k ≤ 2³¹ − 1** per GEMM (GemmCoord is int32; HWGEMM does not chunk K, so a longer k
needs slabs; without them it is INVALID_VALUE, like row 5 in EMULATE); n-tiles ≤ 65535 on grid.y
(or swap); batch chunks ≤ 65535 on grid.z; int32 index vectors for main-loop gather.

**Settled by the IndexMap spec (2026-11-16).**
- Affinity of an operand is decided by fusing over *that operand's own* strides.
- The batch is one decision per call: a single fused λ mode over all its carriers gives
  per-operand batch strides (negative and 0 allowed); otherwise per-λ offsets come from the λ
  scatter vectors, so vec is needed even when A, B and C are all affine.
- When vec is needed it is sized as the whole §4a vec region (conservative, simple).
- Int8 packs are ignored, so they never change affinity or the workspace size.
- A workspace size of 0 keeps the existing rule: a null ws is fine, a misaligned non-null ws is
  still MISALIGNED.
- Leading dimensions CUTLASS gets (ld 0 for a broadcast, empty groups) are a kernel-test item,
  not a spec claim.

### 4.3 The GEMM (decided: reviewer, Q4)

immerdink already uses CUTLASS 2.x (DESIGN §7). v1 uses 2.x `DefaultGemm` FP64 tensor-op
kernels: `arch::Sm90` with instruction shape 16×8×4 on sm_90 (and sm_120, to verify by a
compile), `arch::Sm80` with 8×8×4 on sm_80 / sm_89, which is a **different kernel** (m16n8k*
f64 needs sm_90: `cutlass/arch/config.h`). This is what NVIDIA's own profiler library generates
for sm_90 DGEMM (`GenerateSM90_TensorOp_1684`), and CUTLASS reports it at 2× Ampere per clock.
2.x specializations for 16×8×8 and 16×8×16 also exist (`cutlass/arch/mma_sm90.h`) but no
unit test covers them with the 2.x iterators: a compile check is the cheap fallback.

The instruction shape is probably not the limiter (inferred): at 67 TFLOPS and ~1.98 GHz an SM
needs one m16n8k4 per 16 clocks per sub-partition, far below the issue limit. The likely gaps to
cuBLAS are on small shapes (wave quantization: sq1024 at 128×128 tiles is 64 CTAs on 132 SMs;
cuBLAS may use split-K or stream-K), so **a per-shape tile rule is a deliverable**.

**Gate (one embers session, about 1 h, H100 and H200 if possible):** `cutlass_profiler` built
for `cutlass_tensorop_d1684gemm*` against cuBLAS at m = n = k ∈ {1024, 2048, 4096, 8192} and
the t_batch shape, TFLOPS first, plus a ~50-line register-only mma.sync loop comparing m16n8k4,
k8 and k16. Rule: if k4 reaches ≥ 95 % of the measured DMMA peak and the best d1684 kernel
≥ 90 % of cuBLAS on large squares, keep 16×8×4; a small-shape gap is then a tile / split-K
question. The build is cross-compiled locally or on the compute node, never the login node.

**Gate result (2026-11-17, `bench/hwgemm_gate.cu`; H100 inferno job 15791244, H200 pending).**
The gate used a hand-written microbenchmark and CUTLASS `device::Gemm` / `GemmBatched` instead
of `cutlass_profiler` (faster to build; same kernels).
- Instruction (8 warps/block): m8n8k4 33.5 TFLOPS (half peak), m16n8k4 66.6, m16n8k8 67.0,
  m16n8k16 67.2. k4 is at ≈ 99 % of k16, so **keep 16×8×4**; the Sm80 kernel on Hopper would be
  capped at half peak, so the Sm90 kernel is required there.
- GEMM, best CUTLASS configuration / cuBLAS: sq1024 92–93 %, sq2048 95–96 %, sq4096 95–99 %,
  sq8192 **88–89 %** (49 vs 55 TFLOPS), t_batch 98–99 %, s_batch 96–102 %. Best tile: 128×128
  (warp 32×64, 4 stages) from sq2048 up and on t_batch; 64×128 at sq1024; 64×64 on s_batch.
  128×128 with 64×64 warp tiles runs at ≈ 8 TFLOPS (register spills, inferred) and is dropped.
- **The prediction was wrong in where the gap lies:** small shapes and batches are close to
  cuBLAS; the gap is on the largest square. Hypotheses, unmeasured: threadblock rasterization
  (L2 reuse at 8192², the gate used CUTLASS's default identity swizzle) and a larger tile
  (128×256 or 256×128). These are tile-rule work, not a reason to change the instruction or
  library. **Decision (Eric, 2026-11-17): stay with CUTLASS** (order control for the DIRECT
  bit identity and the K-slab contract, epilogue fusion for scattered C, exact workspace
  accounting, no cuBLAS dependency).

### 4.4 Per hardware class

- **sm_90 (H100, H200)**: the target; tile rule and calibration there, with its own table set
  (DESIGN §6h).
- **sm_89 (RTX 4090)**: the `arch::Sm80` kernel, correctness and tests only (FP64 at 1/64).
- **sm_120 (RTX PRO 6000)**: the sm_90 kernel if it compiles there, correctness only.
- Later: CUTLASS's FP64 SYRK / SYR2K on sm_90 for self-contractions (ROADMAP D3), only if that
  work asks for it.

### 4.5 Planning API: memory and speed reported to ehrenfest (requirement)

**Requirement (Eric, 2026-11-16).** ehrenfest plans around memory and speed, so a HWGEMM
configuration reports its footprint and predicted performance through the same GPU-free
planning API (`libimmerdink_host`) that EMULATE configurations use. A HWGEMM path that runs but
cannot be planned for does not meet issue #75.

| quantity | today (EMULATE) | HWGEMM |
|---|---|---|
| exact workspace bytes | `imd_workspace_size`, `imd_perf_t.workspace_bytes` | §4.2: packed operands only, `8·b·rows·k̂_slab` each, plus vec when needed; 0 when fully affine |
| plan device bytes | `imd_plan_bytes` | as today |
| total bytes, budget | `imd_perf_candidate_t.total_bytes`, `budget_bytes` | the same fields |
| time, interval, power | `imd_perf_predict` | HWGEMM phases (below) |
| FLOPs and rate | `flops_eq` = 2bmnk, `tflops_eq` | the same 2bmnk (real FLOPs), so rates compare directly |
| error bound | `err_bound` (§6f) | γ_k, with the L7 slab form (§3) |
| status | `status`, `dispatch` | `IMD_DISPATCH_NATIVE` |

API changes (reviewer's verdict):
1. **`imd_options_t` grows: an ABI break, not additive.** It has no struct_size (as at v1.2 and
   v1.5), so callers rebuild and ehrenfest's ctypes mirror changes. Add `int64_t k_slab` (k is
   int64; keeps 8-byte alignment). `k_slab ≠ 0` outside HWGEMM is INVALID_VALUE.
2. **Phases.** `imd_perf_phase_t` and `IMD_PERF_MAX_PHASES` = 16 are frozen, so a slabbed call
   gets one phase per kind with `launches` = S: K0 (if any), HWGEMM_PACK_A, HWGEMM_PACK_B,
   HWGEMM_GEMM, CALL (≤ 5 for any S). New phase and kernel ids are additive; the `ops` comment
   says FP64 flops for HWGEMM.
3. **Calibration slots.** 96 = 16 classes × 6; classes 12–14 are free and class 15 has slots
   0–4 (slot 95 is `warm.kl_drop`). HWGEMM gets about three classes (pack, one or two GEMM);
   the tile configuration is a feature, not a class per tile. More would need a new
   calibration struct.
4. **FP64 peak.** Append `peak_fp64_tc_flops_per_s` (optional key, struct_size versioned):
   today's `peak_fp64_flops_per_s` is ambiguous between CUDA-core (DIRECT, ~34 TFLOPS on H100
   SXM) and tensor-core (~67).
5. **Uncalibrated cards.** The roofline fallback max(bytes / BW, 2bmnk / FP64-TC peak) is a
   lower bound, so ranked next to calibrated EMULATE fits it favours HWGEMM. Append an
   `uncalibrated` flag to `imd_perf_t` and `imd_perf_candidate_t` (versioned) and document the
   bias.
6. **Ranking.** Opt-in through appended request fields (a modes mask, a list of slab lengths).
   HWGEMM candidates appear only when both the request's struct_size and `candidate_size` are
   new enough (an old caller would otherwise see N = 0 and no mode). Mode and `k_slab` join the
   deterministic tie order. With a budget, the candidate is the **largest `k_slab` that fits**
   (fewest slabs, fastest). N, g and pack modes do not apply to HWGEMM (PACK_NONE only); vec
   mode does unless the call is fully affine. The ranking only reports: ehrenfest chooses.

Workspace exactness for HWGEMM goes into `cert_workspace`, so the bytes ehrenfest is told are
the bytes the call uses.

### 4.6 Other element types: FP32 later (Eric, 2026-11-16)

Eric wants native FP32 contractions considered too, so v1 is FP64-only but must not paint
FP32 into a corner. That is why the mode is named for *how* it computes, not the type:
`IMD_HWGEMM` means "the hardware's own GEMM arithmetic for the call's element type" (the name
`IMD_NATIVE` was dropped: `IMD_PLAN_NATIVE` and `IMD_CODE_NATIVE` already mean something else, §6h).
The element type is a separate axis of the plan; the mode does not change with it.

**What FP32 means on the hardware.** Hopper and Ada have no IEEE FP32 tensor-core instruction,
so "native FP32" is one of several arithmetics with different accuracy (H100 SXM dense
datasheet peaks, to be checked per card):

| arithmetic | accuracy | H100 peak |
|---|---|---|
| SIMT FFMA (CUDA cores) | IEEE FP32, γ_k with u = 2⁻²⁴ | ≈ 67 TFLOPS |
| 3×TF32 tensor cores (CUTLASS) | close to FP32 (operands split into 2 TF32 parts, 3 products) | ≈ 495/3 TFLOPS |
| TF32 tensor cores | inputs rounded to 10 mantissa bits: **not** FP32-accurate | ≈ 495 TFLOPS |
| Ozaki-style emulation on INT8 | FP32-accurate with enough moduli | INT8-bound |

So FP32 needs an accuracy choice, not just a type: the default must be FP32-accurate (FFMA or
3×TF32, chosen per shape like the FP64 tile rule) and TF32 must be an explicit opt-in, never a
silent default (cuBLAS's `CUBLAS_COMPUTE_32F_FAST_TF32` is the precedent). Which of these is
worth building, and whether ehrenfest needs FP32 at all, is open until there is a use case.

**What carries over unchanged.** The lowering (§5 of DESIGN), the in-place-or-packed rule (the
pack region becomes 4·b·rows·k̂ bytes), k-slabs, the `store_out` epilogue, the `HwGemmPipeline`
spec (it has no element width) and every planning-API field (bytes, `flops_eq`, time; the
error bound takes the arithmetic's unit roundoff).

**Decided now, so FP32 is additive later** (the §4.5 ABI break is the moment to do it):
1. `imd_options_t` gains `struct_size` (first member) as well as `k_slab`, so later fields do
   not break the ABI again.
2. It also gains `int32_t dtype` (`IMD_F64` = 0, the default) and `int32_t compute`
   (`IMD_COMPUTE_DEFAULT` = 0, meaning "accurate for the type"). Any other value is
   NOT_SUPPORTED until it is implemented.
3. `imd_contract` keeps its `double*` signature for FP64; another type gets its own typed entry
   point (for example `imd_contract_f32` with `float` α, β), not a `void*` form of the FP64 call.
4. The IndexMap sizing takes the element width as a parameter (8 today), so a 4-byte type is a
   new model, not a new spec.
5. Calibration classes and kernel ids are per (type, arithmetic); the FP64 ones come first.

FP32 *emulated* by Ozaki in EMULATE mode is a separate question (it is the same dtype axis
with mode EMULATE) and is not proposed here.

## 5. Spec, proof and test impact

- **TLA+.**
  - `IndexMap`: mode HWGEMM; dispatch (HWGEMM / DIRECT / K-empty / SCALE / NOOP); per-operand
    in-place-or-packed as a pure function of the lowering; status row 5's HWGEMM limits (§4.2
    Limits; the RN κ acceptance does not apply); the reads and contributions of the packed and
    in-place paths equal the einsum's; `k_slab` validity.
  - A new module `HwGemmPipeline` (as `K4Align` and `WgmmaPipeline` are separate): K0 (if
    any), per slab pack(s) then GEMM, version tags on the pack buffers and C: no stale read of
    a pack buffer, slab s + 1's pack after slab s's GEMM (WAR), the caller's β applied exactly
    once and α on every slab, C's original value never read when β = 0, every K position
    folded into C exactly once; the workspace regions in `LayoutExact` form.
  - `HostAPI`: no change (DIRECT's pack rule, §4.1).
- **Proof.** None new for the γ_k part; the slab composition reuses PROOFS L7 (cite it for
  HWGEMM in DESIGN).
- **Tests.** Bit compare against DIRECT (§3 hypothesis); the γ / L7 bound against the oracle on
  random and wide-range data; non-finite (NaN-ness) and subnormal cases per instruction; run-to-
  run identity; graph capture; workspace exactness (`cert_workspace`); the full contract suite
  in HWGEMM mode, with and without slabs, affine and packed operands.
- **Performance model.** The §4.5 phases and classes, calibrated on sm_90.

## 6. Decisions

- **Q1 (Eric):** new mode `IMD_HWGEMM`; `IMD_DIRECT` stays the sequential canonical-order
  reference.
- **Q2 (reviewer: accept with changes):** per-operand in place or packed; fully affine is the
  primary path; main-loop gather a planned follow-up chosen by the perf model; affine2 dropped;
  batch via strides or per-CTA λ offsets, never a host loop; own `store_out` epilogue (§4.2).
- **Q3 (Eric):** k-slabs in v1.
- **Q4 (reviewer: accept 16×8×4 behind a gate):** §4.3's embers measurement decides; tile rule
  per shape is a deliverable.
- **Q6 (Eric, 2026-11-16):** rename the mode to `IMD_HWGEMM` and keep FP32 in view (§4.6):
  `struct_size`, `dtype` and `compute` go into the same `imd_options_t` break as `k_slab`.
- **Q5 (reviewer: defer):** reusable FP64 packs later; HWGEMM accepts and ignores int8 packs
  like DIRECT.
- **§4.5 (reviewer: sound in intent, with changes 1–6).**

## 7. Plan and gates

1. TLA+ for §5 (IndexMap changes, `HwGemmPipeline`), then DESIGN text.
2. In parallel, GPU-free: count affine operands on ehrenfest's real descriptors (`imd_lowering`).
3. The Q4 gate: one embers session on H100 / H200 (§4.3).
4. Kernel and tests, locally on the 4090 for correctness only.
5. Host API: the §4.5 changes, the perf-model phases and classes, calibration on sm_90.
6. One PACE run on H100 and H200, against cuBLAS DGEMM and cuTENSOR FP64 on the standard shape
   set, reported per card, TFLOPS first.

**Predicted outcome** (to check, then explain any gap), split by limiter:
- Large affine squares (main-loop bound): ≥ 90 % of cuBLAS DGEMM.
- sq1024 / t_batch (tile, wave and split-K bound): ahead of N = 14 emulation as cuBLAS DGEMM is
  today; the gap to cuBLAS depends on the tile rule.
- Non-affine tensor cases: slower than the affine case by about §4.2's pack ratio, and ahead of
  cuTENSOR FP64 where its permutation pass costs more than our gather.

## References

- Ozaki, Uchino, Imamura, *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix
  multiplication using an integer modular technique*, arXiv:2504.08009,
  https://arxiv.org/abs/2504.08009.
- Uchino et al., *High-Performance and Power-Efficient Emulation of Matrix Multiplication using
  INT8 Matrix Engines*, SC25 workshops, doi:10.1145/3731599.3767539,
  https://arxiv.org/abs/2508.03984.
- Schwarz et al. (NVIDIA), *Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores
  Through Extensions of the Ozaki Scheme*, SCA/HPCAsia 2026, doi:10.1145/3773656.3773670,
  https://arxiv.org/abs/2511.13778.
- Van Zee, van de Geijn, *BLIS: A Framework for Rapidly Instantiating BLAS Functionality*,
  ACM TOMS 41(3), 2015, doi:10.1145/2764454 (packing).
- Matthews, *High-Performance Tensor Contraction without Transposition*, SIAM J. Sci. Comput.
  40(1), 2018, doi:10.1137/16M108968X (DOI not yet confirmed against Crossref),
  https://arxiv.org/abs/1607.00291 (TBLIS).
- Khattak, Mikaitis, *Accurate Models of NVIDIA Tensor Cores*, arXiv:2512.07004,
  https://arxiv.org/abs/2512.07004 (FP64 DMMA as a sequential fma chain).
- Fasi, Higham, Mikaitis, Pranesh, *Numerical behavior of NVIDIA tensor cores*, PeerJ Comput.
  Sci. 7:e330, 2021, doi:10.7717/peerj-cs.330.
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002,
  doi:10.1137/1.9780898718027 (§3.1, summation error bounds).
- CUTLASS v4.8.0 (vendored, `third_party/cutlass`): examples 18, 36, 52; FP64 sm_90 unit tests;
  `arch/mma_sm90.h`, `arch/config.h`, `python/cutlass_library/generator.py`, CHANGELOG 2.11.0.
