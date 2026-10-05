# immerdink — scope

**Goal:** tensor contractions in the GETT/TBLIS style, computed with Ozaki scheme II
emulation. You get FP64-accurate (and later FP32-accurate) contractions
`C[IJL] = α · Σ_K A[IKL] · B[KJL] + β · C[IJL]`, but the inner products run on INT8
matrix hardware. Operands are never explicitly transposed into matrices.

Status: specification phase complete (see `spec/README.md`); implementation starts at PLAN M0.
Docs: [DESIGN.md](DESIGN.md) (what and how), [PLAN.md](PLAN.md) (milestones, questions),
[MEMORY_MODEL.md](MEMORY_MODEL.md) (equations), [PROOFS.md](PROOFS.md) (lemmas / Lean blueprint).

---

## 1. Background in one page

### Ozaki scheme II (Ozaki, Uchino, Imamura 2025, arXiv:2504.08009)

To emulate `C ≈ A·B` in FP64:

1. **Scale and truncate.** Choose power-of-two row scalings `D_A` (one per row of A)
   and column scalings `D_B` (one per column of B). Set `A' = trunc(D_A·A)` and
   `B' = trunc(B·D_B)`. Both are exact integer matrices, small enough that every entry of
   `A'·B'` stays below `M/2`, where `M = ∏ m_l`.
   - *fast mode* bounds `|A'B'|` cheaply (norm-based).
   - *accurate mode* spends one extra low-precision GEMM to get a tighter bound, which
     means fewer moduli for the same accuracy.
2. **Residues.** Pick N pairwise-coprime moduli `m_l ≤ 256`. Then `A'_l = A' mod m_l`
   and `B'_l = B' mod m_l` fit in 8 bits.
3. **N INT8 GEMMs.** `C_l = A'_l · B'_l` with INT32 accumulation. This is exact as long as
   k is not too large; split K otherwise.
4. **CRT reconstruction.** `C' = Σ_l w_l·(C_l mod m_l) mod M`, using fixed CRT weights
   `w_l`. This needs roughly two-word floating accumulation to stay exact enough.
5. **Unscale.** `C ≈ D_A⁻¹ · C' · D_B⁻¹`.

The number of GEMMs grows **linearly** in N. Ozaki scheme I grows quadratically in the
slice count. The paper's FP64 configurations use about 14 moduli (the "fast-14" variant).
Accuracy is normwise per row and column: entries far below their row or column max lose
relative accuracy. This is the main numerical caveat, and the tests must probe it.

### GETT / TBLIS

- **TBLIS** (Matthews 2018) reuses the BLIS loop nest and micro-kernels. Its packing
  routines read tensors through *block-scatter* index vectors, so the tensor-to-matrix
  "transpose" happens for free inside packing, which BLIS does anyway.
- **GETT** (Springer & Bientinesi) generates code per contraction: it enumerates loop
  orders and blocking, then chooses with a performance model.

### Why the two fit together

Ozaki-II is *one GEMM-shaped core plus elementwise work before and after it*:

| Ozaki-II step               | BLIS/TBLIS hook where it can be fused                         |
|-----------------------------|---------------------------------------------------------------|
| scale + truncate + residues | **pack A / pack B**: read strided tensor → emit N int8 panels  |
| N INT8 GEMMs                | int8 micro-kernel (VNNI / tensor core mma)                    |
| CRT + unscale + α/β         | **C update / epilogue**: write-back scattered into C's layout |

The tensor permutation, the precision conversion, and the residue computation all
happen in passes you need anyway. That fused pipeline is the project's reason to exist.
Neither GEMMul8 (matrices only) nor TBLIS (native float only) does this.

---

## 2. Hardware reality on the dev box

| Device                          | Native FP64        | INT8 path                         | Ozaki-II FP64 outlook                         |
|---------------------------------|--------------------|-----------------------------------|-----------------------------------------------|
| RTX 4090 (sm_89)                | ~1.3 TF peak, 1/64 rate | ~660 TOPS dense tensor core (`mma.sync …s8.s8`) | **Big win.** The int8 rate is ~500× the FP64 rate, so N ≈ 14 int8 GEMMs leave a wide margin over native DGEMM |
| i9-13900K (Raptor Lake, P+E)    | AVX2 FMA, 16 flop/cyc/P-core | AVX-VNNI `vpdpbusd` (u8×s8, 256-bit), roughly 8× the FP64 MAC rate | **Slower than native DGEMM.** N≈14 GEMMs at ~8× rate → ≤0.6× before overhead |

On this CPU, Ozaki-II is a **correctness and reference** target, not a speed target. It
would pay off on CPUs with AMX (Sapphire Rapids and later) or SME (Apple M4 class), where
the INT8:FP64 ratio is about 64× or more.

CPU detail: `vpdpbusd` multiplies **unsigned × signed**. Store A residues as
`[0, m_l)` (u8) and B residues as symmetric s8. Then `|product| ≤ 255·128`, and INT32
holds `k ≤ 2^16` before overflow, so K blocking must respect that. No offset correction is
needed. There is no s8×s8 VNNI (AVX-VNNI-INT8) on this part.

---

## 3. Decisions so far

| # | Decision | Rationale |
|---|---|---|
| D1 | **Spec-first.** Every core design question gets a TLA+ spec and TLC models before code. | Theory fully defined before implementation; see §6. |
| D2 | **TBLIS-style, not GETT.** | Ozaki-II must rewrite A and B into N int8 residue copies anyway. Writing them in GEMM-canonical layout makes the tensor→matrix permutation free (TBLIS's insight, by force). GETT's per-contraction code generation buys nothing once operands are rewritten, and it adds an autotuner. The only tensor-aware kernels left are the gather (pack) and the scatter (CRT epilogue). |
| D3 | **GPU (RTX 4090, sm_89) is the implementation target. No optimized CPU kernels.** | GPU-first is actually *simpler*: CUTLASS supplies the int8 GEMM, while a CPU path means hand-writing a BLIS nest that won't beat DGEMM on this CPU anyway. The CPU's role is the **exact reference oracle** (Python big-int/`Fraction`, no FP), which also matches the TLA+ semantics. |
| D4 | **CUTLASS 2.x** (`GemmUniversal` kBatched, SM80 int8 on sm_89) with an `Sm80EVT` mod-p → u8 epilogue from the start; v4.8.0 pinned. | 3.x has no sm80/sm89 builders; the epilogue was compile-verified; stock int32 output (V0) only as a baseline. |
| D5 | **From scratch.** C++20 + CUDA (13.4), CMake. C ABI, then Python bindings (nanobind) later. | Everything is local. cuBLAS and cuTENSOR 2.7 are already in the `main` conda env for baselines. |
| D6 | **Dedicated conda env** (`environment.yml`) instead of `main`. | `main` is used by another project's GPU test suite; isolating the env keeps versions and benchmarks clean. |
| D7 | **General-purpose kernel.** CC was the motivation but drives no design choice; batch indices (L) are in v1; complex is reserved (2M method), not precluded. | User guidance: build a good Ozaki TBLIS-style kernel, don't overengineer toward a downstream user. |
| D8 | **Accuracy contract:** the caller picks N. *Auto* and *bounded auto* (data-chosen N ≤ N_max) come later. A DGEMM-equivalent guarantee is low priority. | N is explicit, so memory and time are exact functions of it. Bounded auto keeps memory plannable: `workspace_size(problem, N_max)`. |
| D9 | **Workspace:** caller-provided; `imd_workspace_size` is a pure host function equal to MEMORY_MODEL §3 / DESIGN §4a (certified and model-checked). Callers tile I/J/L. | Exact, plannable memory; no allocation in the hot path. |
| D10 | **CRT arithmetic:** exact unsigned multi-word running CRT (division-free, one conditional subtract), 32·L32-bit words: N ≤ 16 in 128 bits, N ≤ 20 in 160. | Replaces the first int128 running-mod plan (cap N ≤ 15); refines exact CRT (spec: CRTRealization). |
| D11 | **Exact-integer fast-mode scaling** (Cauchy–Schwarz shift chosen by exact integer comparison) — [deviation] | Overflow-free, order-independent, reproducible, provable; at least as tight as the float-log rule. |
| D12 | **Round-to-nearest-even quantization** by default (TRUNC available) — [deviation] | Unbiased; 2.4× (zero-mean) to 19–191× (same-sign data) more accurate at equal N; costs a negligible κ/2 slack. |
| D13 | **Single final rounding** of the exact integer at the final exponent; bitwise determinism; Python oracle is bit-exact | Every emulated step is exact integer math, so the GPU must equal the oracle bit-for-bit. |
| D14 | **Non-finite rows/columns** handled by a native FP64 fma path for the affected outputs only | IEEE-consistent propagation without host sync or a separate library. |
| D15 | **IndexMap:** arbitrary int64 strides, A/B diagonals and single-operand labels lowered into K, sorted-stride C rule, status precedence, corrected fusion rule | General TBLIS-like semantics, always-on legality checks. |
| D16 | **Verification stack:** TLA+ (design logic) + `params.py` (real-width certificate) + oracle (bit-exact numerics) + PROOFS.md → Lean (parametric lemmas) | Each covers what the others can't. |

### Pipeline v1 (TBLIS-style; full detail in DESIGN §4)

```
K0 vec   : copy plan's scatter vectors into the workspace
K1 scale : strided pass over A rows / B cols → non-finite flags + exact int16 shifts σ, τ
K2/K3    : strided gather of A, B → RN quantize → N int8 residue panels (canonical TN layout)
K4 gemm  : CUTLASS batched int8 GEMM per K-chunk, epilogue (acc [+W]) mod p → u8 planes
K5 crt   : exact multi-word CRT → one rounding → unscale → α/β → strided scatter into C
```

Memory and traffic for each step are given in [MEMORY_MODEL.md](MEMORY_MODEL.md).

### Later
FP32 emulation, complex, accurate mode, auto-N, fused variants (V2/V3), packed micro-kernel
experiments, other GPUs (sm_90/100/120), AMX/SME CPU back ends, Python bindings.

### Non-goals
Multi-operand path optimization (opt_einsum/cotengra), sparse/block-sparse, distributed,
autodiff.

### On Rust (for fun)
Feasible as a *host/API* language, not for kernels right now. `cudarc` can drive the
driver API and launch kernels compiled from CUDA C++ (CUTLASS) via `build.rs`, and PyO3
gives excellent Python bindings. Writing the kernels in Rust (Rust-CUDA, CubeCL) would
mean giving up CUTLASS and hand-building the int8 tensor-core GEMM. A Rust wrapper over
the C ABI is a reasonable later experiment. No Rust toolchain is installed locally.

---

## 4. Where the rest lives

- Specifications, models, certificate, oracle: [`spec/README.md`](../spec/README.md).
- Milestones, testing/benchmark discipline, open questions: [PLAN.md](PLAN.md).
- Design details (numerics, pipeline, workspace, IndexMap, API): [DESIGN.md](DESIGN.md).

## References
- Ozaki, Uchino, Imamura. *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique.* arXiv:2504.08009.
- Uchino, Ozaki, Imamura. *Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit.* IJHPCA 39(3), 2025.
- Uchino et al. *High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines.* arXiv:2508.03984.
- Matthews. *High-Performance Tensor Contraction without Transposition.* SIAM J. Sci. Comput. 2018 (TBLIS).
- Springer, Bientinesi. *Design of a High-Performance GEMM-like Tensor-Tensor Multiplication.* ACM TOMS 2018 (GETT).
