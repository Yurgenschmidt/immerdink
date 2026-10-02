# immerdink — scope

**Goal:** tensor contractions in the GETT/TBLIS style, computed with Ozaki scheme II
emulation. You get FP64-accurate (and later FP32-accurate) contractions
`C[IJL] = α · Σ_K A[IKL] · B[KJL] + β · C[IJL]`, but the inner products run on INT8
matrix hardware. Operands are never explicitly transposed into matrices.

Status: specifying (see §4). No implementation yet, by design.

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
| RTX 4090 (sm_89)                | ~1.3 TF peak, 1/64 rate | ~660 TOPS dense tensor core (`mma.sync …s8.s8`) | **Big win.** Paper reports ~9.8 TF vs 0.62 TF native DGEMM at n=8192 (~16×) |
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
| D4 | **CUTLASS** for the int8 GEMM, starting with stock batched GEMM. Custom epilogue (mod-reduce → u8) is step 2. Packed/fused kernels come later. | Simple first. The memory model (MEMORY_MODEL.md) says V1 is the right first target. |
| D5 | **From scratch.** C++20 + CUDA (13.4), CMake. C ABI, then Python bindings (nanobind) later. | Everything is local. cuBLAS and cuTENSOR 2.7 are already in the `main` conda env for baselines. |
| D6 | **Dedicated conda env** (`environment.yml`) instead of `main`. | `main` is used by another project's GPU test suite; isolating the env keeps versions and benchmarks clean. |
| D7 | **Workload:** v1 serves real FP64 science (t1-dressed DF-CCSD, CCSD(T)), but the design stays general: complex (tensor networks), batched small contractions, large GEMM-like ones. | Batch indices (L) are in v1. Complex is not in v1, but nothing in the API or specs may preclude it. Small-problem behavior matters (CCSD(T) has many medium contractions). |
| D8 | **Accuracy contract:** the caller picks N. *Auto* and *bounded auto* (data-chosen N ≤ N_max) come later. A DGEMM-equivalent guarantee is low priority. | N is explicit, so memory and time are exact functions of it. Bounded auto keeps memory plannable: `workspace_size(problem, N_max)`. |
| D9 | **Workspace:** caller-provided, with `workspace_size(problem, N)` returning exactly MEMORY_MODEL §3. | Fits planning memory in the downstream code. No allocation in the hot path. |
| D10 | **CRT arithmetic:** exact signed int128 with running mod-M reduction. Caps N ≤ 15 with the greedy moduli. | Exact integers are fully specifiable in TLA+ (`CRTRealization` refines `OzakiCRT.Reconstructed`). Wider integers are a later extension if N > 15 is needed. |

### Pipeline v1 (four kernels, TBLIS-style)

```
1. scale    : strided reduction over K  → exponents e[I,L], f[J,L]
2. pack     : strided gather of A,B → scale → trunc → N residue panels (int8, canonical TN layout)
3. gemm     : CUTLASS batched int8 GEMM over (N moduli × L batch), INT32 accum,
              epilogue: mod m_l → u8                                        (V1 in MEMORY_MODEL.md)
4. crt      : N u8 residues → exact CRT → FP64 → unscale → α·, +β·C → strided scatter into C
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

## 4. Specification plan (TLA+)

TLC limits: integers are 32-bit and there is no floating point. So specs are
**parametric** and model-checked **scaled down** (e.g. 3-bit residues, 5-bit
accumulators). Floating-point error analysis (scaling truncation, final FP64 rounding) is
outside TLC. It is covered by (a) exact-rational toy-float models where TLC can enumerate,
(b) the paper's bounds, and (c) the exact Python oracle.

| Spec | Defines / checks | Status |
|---|---|---|
| `OzakiCRT` | Integer core for one output element: residues, INT-accumulator wrap, mod reduction, CRT. Invariants `ResiduesFit`, `NoAccOverflow` (order-independent), `CRTExact`. Each precondition has a negative model proving it is needed. | **drafted** |
| `OzakiScaling` | Toy floating-point format (p-bit mantissa) → per-row/column exponent choice → truncation. Shows the scaling rule *implies* `OzakiCRT.Preconditions`, and checks the error bound exactly with rationals. | next |
| `CRTRealization` | int128 running-mod reconstruction (D10) *refines* `OzakiCRT.Reconstructed`; scaled-down word widths, positive + negative (overflow) models. | next |
| `IndexMap` | Label analysis → I/J/K/L classes; pack gather is a bijection onto the canonical buffer; C scatter is injective; batch indices; stride policy (negative/zero/aliasing). | next |
| `Pipeline` | Kernel sequence and workspace lifetimes (V0/V1/V2), K-splitting, grouping. Each C element is finalized exactly once after all N contributions; workspace ≤ `W` from MEMORY_MODEL.md. | after IndexMap |
| `CtaPipeline` | Multistage cp.async/barrier protocol, only if we write our own GEMM mainloop (not needed while CUTLASS owns it). | later |

`spec/check.sh` runs all models: `MC_pos_*` must pass and `MC_neg_*` must violate their
named invariant. Run `tools/quiet_check.sh` first.

---

## 5. Milestones

| #  | Milestone | Exit criterion |
|----|-----------|----------------|
| S  | **Specs**: OzakiCRT, OzakiScaling, IndexMap, Pipeline (+ CRTRealization) | All models green; every open design question below answered or in a spec |
| M0 | Scaffolding: env, CMake, CUDA, tensor view type, exact Python oracle | Oracle matches the `OzakiCRT` semantics on random small cases |
| M1 | Ozaki-II **matrix** DGEMM on GPU: pipeline v1 with stock CUTLASS (V0) | Error vs N and dynamic range φ matches the paper's curves |
| M2 | Tensor front end: labels → I/J/K/L, gather-pack and scatter-CRT kernels | Random contractions (rank ≤ 6, random perms/strides) pass vs oracle |
| M3 | V1 epilogue (mod → u8), K-split, workspace query = MEMORY_MODEL §3 | Measured peak memory equals the formula |
| M4 | Performance: close the gap to the §5 lower bound in MEMORY_MODEL.md | Beat cuTENSOR FP64 and cuBLAS DGEMM; report % of bound |
| M5 | Auto-N, FP32 emulation, complex, fused variants | — |

---

## 6. Testing and benchmarking discipline

**Accuracy tests:**
- Inputs `(rand − 0.5)·exp(φ·randn)` with φ swept from 0 to about 4, as in the Ozaki
  literature. Report max componentwise error relative to `|A||B|`, and normwise error.
- Adversarial cases: zero rows, Inf/NaN, subnormals, huge dynamic range within one row,
  K at `K_max` and `K_max + 1`, negative and zero strides, aliasing.

**Benchmarks: the machine must be quiet first.** Every test or benchmark run starts with
`tools/quiet_check.sh`, which checks load, top CPU users, GPU utilization and other compute
processes, and records governor and turbo state. If the machine isn't quiet, we wait
rather than run. Benchmarks also:
- warm up, repeat, and report median and spread, not best-of,
- record the conditions (load, clocks, temps, driver, commit) next to every result,
- pin any CPU-side timing to P-cores.

---

## 7. Open design questions

Answered so far: D1–D10 above. Still open:

1. **Non-finite inputs.** Inf/NaN cannot go through integer residues. Detect and fall back
   to native FP64, propagate NaN to the affected rows/columns, or declare it unsupported?
2. **Determinism.** The integer path is naturally bitwise reproducible. Make that a
   guaranteed property?
3. **Stride policy.** Negative strides? Zero strides (broadcast) in A/B? Must C be
   non-self-overlapping (yes, presumably)? Aliasing between C and A/B?
4. **Small-problem fallback.** Below some size, call native cuBLAS/cuTENSOR FP64? (D7
   makes this relevant: CCSD(T) and batched workloads.)
5. **Scaling mode.** Fast (norm bound) only for v1, or accurate mode too?
6. **Streams/async.** Fully stream-ordered and async (cuBLAS-style handle + stream)?
7. **CUTLASS pinning.** Git submodule at a pinned tag, or CMake FetchContent?
8. **Batch-index scaling.** Per-batch exponents (`e[I,L]`), as in MEMORY_MODEL. Fine,
   or should batch slices share exponents?
9. **Moduli set.** Fixed greedy table (256, 255, 253, …), or allow caller-chosen moduli?

## References
- Ozaki, Uchino, Imamura. *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique.* arXiv:2504.08009.
- Uchino, Ozaki, Imamura. *Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit.* IJHPCA 39(3), 2025.
- Uchino et al. *High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines.* arXiv:2508.03984.
- Matthews. *High-Performance Tensor Contraction without Transposition.* SIAM J. Sci. Comput. 2018 (TBLIS).
- Springer, Bientinesi. *Design of a High-Performance GEMM-like Tensor-Tensor Multiplication.* ACM TOMS 2018 (GETT).
