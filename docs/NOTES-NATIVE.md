# Native FP64 contractions on tensor cores: design note (draft)

Status: **draft for review**, 2026-11-16. No code yet. Nothing here changes the library until the
open questions in §6 are answered and the TLA+ work in §5 is done.

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
- **Affine operand**: an operand whose GEMM view is a plain strided matrix (each GEMM index is
  one fused mode, so offset = row·s_r + k·s_k). CUTLASS can read it directly.
- **Packing**: copying an operand into a contiguous, GEMM-friendly layout in workspace first
  (as TBLIS does). The emulated path already packs (into int8 residues); here the pack would
  be an FP64 copy.
- **DMMA**: the FP64 tensor-core MMA instruction (sm_80 and later).

## 3. Accuracy and reproducibility contract

Native FP64 is **not** correctly rounded, unlike the emulated path (DESIGN §3.7). The new
path's contract:

- **Error.** `|D − x| ≤ γ_k·(|a||b|)_ij` per element (γ_k = k·u/(1 − k·u), u = 2⁻⁵³), then α
  and β as in DESIGN §3.7. This holds for any summation order, so tiling does not change it.
  §6f's error estimator already reports γ_k for DIRECT; the new path reuses it.
- **Reproducibility.** Bit-identical run to run for the same plan on the same device class:
  one fixed tile configuration per plan, no split-K with atomics, a fixed K order inside each
  tile. Results differ in the last bits from `IMD_DIRECT` and from cuBLAS.
- **Non-finite inputs.** IEEE propagation. The order-dependent corner (a finite partial sum
  overflowing, DESIGN §3.7 oracle A7) can give a different Inf / NaN than DIRECT's canonical
  order; this is documented, not hidden.
- **Subnormals.** To verify on each card: DMMA must keep subnormal inputs and outputs (a unit
  test, not an assumption).

## 4. Design

### 4.1 Where it plugs in

Same plan, same workspace query, same stream-ordered, graph-capture-safe call, no allocation
and no host sync inside `imd_contract` (DESIGN §4, kernel-scope rules). Proposed API change:
one new `mode` value next to `IMD_EMULATE` and `IMD_DIRECT` (§6 Q1), dispatch `NATIVE`.
K-empty contractions keep `k_direct` (k = 1 is one fma; a GEMM is pointless there).

### 4.2 Operand access: three options

| option | how | extra cost | workspace |
|---|---|---|---|
| A. affine | operand is affine (§2): CUTLASS reads it in place with its strides; `affine2` layouts (CUTLASS example 18, FP64 on sm_80) allow a non-unit stride in both dimensions | none, but non-unit-stride loads are uncoalesced and lose `cp.async` vector width | 0 |
| B. pack | a gather kernel (like K2/K3, 8 B instead of 1 B per element) writes A and B K-contiguous into workspace; then a plain TN DGEMM | one extra read and write of A and B: 16·b·(m + n)·k bytes | 8·b·(m + n̂)·k̂ |
| C. gather in the main loop | CUTLASS gather (examples 36, 52) reads through the scatter vectors directly | uncoalesced unless one fused mode is unit-stride | 0 |

Proposal: **B as the general path, A when both operands are affine with a unit stride**, chosen
at plan time from the descriptor and published in `imd_lowering` (the existing pattern for KF /
KL). B's cost relative to the GEMM is about `8(m + n)k / (mnk·F/BW)`; on an H100 (≈ 3 TB/s,
≈ 60 TFLOPS achieved) packing costs < 5 % when `mn/(m + n)` is above about 400, i.e. it matters
only for skinny shapes, where A often applies. This is a prediction to check, not a measurement.

The footprint is the problem for B: 8192² × k = 8192 needs 1 GiB of workspace. Following the
per-plan resource toggles (memory note "modular plans"), B gets a **K-slab option**: pack and
multiply K in slabs of k_s, accumulating into C with β = 1 after the first slab. That bounds
the workspace at 8·b·(m + n̂)·k_s and adds ⌈k/k_s⌉ − 1 extra roundings, which the γ bound
absorbs (the bound becomes γ_k plus one u per slab boundary; to be written as a lemma).

### 4.3 The GEMM

immerdink already uses CUTLASS 2.x (DESIGN §7 rejects 3.x collectives for sm_80/sm_89). The
vendored CUTLASS (v4.8.0) has 2.x FP64 tensor-op GEMMs for both `arch::Sm80` and `arch::Sm90`
(unit tests `gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu`, instruction shape 16×8×4), so the
first kernel is a 2.x `DefaultGemm` with FP64 operands and accumulators. Open point: whether
the sm_90 DMMA shapes 16×8×8 and 16×8×16 (present in `cute/arch/mma_sm90.hpp`, used only from
CuTe) are needed to reach Hopper's peak; if so, a CuTe main loop for sm_90 (§6 Q4).

Epilogue: α / β and the store through C's scatter vectors, as K5 does today; a plain strided
store when C is affine.

Batching: the batch index λ maps to CUTLASS's batched GEMM when the batch stride is affine,
otherwise to a grouped launch over λ.

### 4.4 Per hardware class

Following "many device-specific solutions, selected per device class":
- **sm_90 (H100, H200)**: the target; tile configurations tuned there, with its own table set
  (DESIGN §6h).
- **sm_89 (RTX 4090), sm_120 (RTX PRO 6000)**: FP64 at 1/64 rate. The same kernel builds and
  runs, for correctness and tests only; no tuning.
- Later: CUTLASS's FP64 SYRK / SYR2K on sm_90 for self-contractions (ROADMAP D3, symmetry
  exploration), only if that work asks for it.

## 5. Spec, proof and test impact

- **TLA+.** `IndexMap`: the new mode and its dispatch rules (NATIVE vs DIRECT vs K-empty, and
  which status checks apply; the RN κ acceptance and CUTLASS m, n̂ limits). `Pipeline`: the
  NATIVE launch sequence (K0, optional pack, GEMM, no K5) and the workspace regions in
  `LayoutExact`, including K-slab reuse ordering (slab s + 1 packs only after slab s's GEMM
  has read the buffer). `HostAPI`: no change expected (same plan / workspace / pack ownership
  rules; packed residue operands are not accepted by NATIVE plans, like DIRECT).
- **Proof.** One lemma: the K-slab bound (γ_k plus the slab roundings) in PROOFS, and in Lean
  if it joins the paper track.
- **Tests.** Oracle check of the γ bound on random and wide-range data; non-finite and
  subnormal cases; run-to-run bit identity; graph capture; workspace exactness (`cert_workspace`
  extended); every contraction in the existing contract suite run in NATIVE mode.
- **Performance model.** One new kernel class (pack and GEMM), so ehrenfest can compare NATIVE
  with emulation through `imd_perf_rank`. Calibrated on sm_90 only.

## 6. Open questions

- **Q1. New mode or upgrade DIRECT?** Recommendation: a new mode (`IMD_NATIVE`), keeping
  `IMD_DIRECT` as the sequential, canonical-order reference (its bits are what the oracle and
  the flagged-row path in K5 match). Alternative: make DIRECT tiled and lose that reference.
- **Q2. Operand access.** Recommendation: B (pack) + A (affine) from §4.2, C only if B's pack
  cost shows up on real shapes.
- **Q3. K-slab option** in the first version, or only once a memory-tight caller appears?
  Recommendation: in the first version; ehrenfest already states memory limits.
- **Q4. sm_90 instruction shape.** Is the 2.x 16×8×4 DMMA enough to reach ≥ 90 % of cuBLAS
  DGEMM on H100 / H200? To settle by reading (CUTLASS profiler data, NVIDIA's Hopper tuning
  guide) and one short embers session, not by guessing.
- **Q5. Packed FP64 operands** (an FP64 pack reused across calls, the native analogue of D1):
  later, if ehrenfest reuses an operand natively.

## 7. Plan and gates

1. Answer §6 (independent reviewer, then Eric for Q1 and Q3).
2. TLA+ for §5, then DESIGN text.
3. Kernel and tests, locally on the 4090 for correctness only (FP64 is 1/64 there, so no
   timing locally).
4. One PACE run on H100 and H200 (embers first; inferno within the policy budget), against
   cuBLAS DGEMM and cuTENSOR FP64 on the standard shape set, reported per card in TFLOPS
   first.

**Predicted outcome** (to check, then explain any gap): ≥ 90 % of cuBLAS DGEMM on affine
GEMM-shaped cases; on non-affine tensor cases, faster than cuTENSOR FP64 when its permutation
pass is the extra cost, and within the pack cost of §4.2 otherwise; native ahead of emulation
at N = 14 on sq1024 / t_batch (as cuBLAS DGEMM is today) and behind it on large squares.

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
- CUTLASS v4.8.0 (vendored, `third_party/cutlass`): examples 18, 36, 52; FP64 sm_90 unit tests.
