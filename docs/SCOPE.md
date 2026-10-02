# immerdink — scope

**Goal:** tensor contractions in the GETT/TBLIS style, computed with Ozaki scheme II
emulation. You get FP64-accurate (and later FP32-accurate) contractions
`C[IJL] = α · Σ_K A[IKL] · B[KJL] + β · C[IJL]`, but the inner products run on INT8
matrix hardware. Operands are never explicitly transposed into matrices.

Status: scoping. Nothing here is built yet.

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

## 3. Proposed scope

### In scope (v1)
- Binary contraction `C = α·A·B + β·C` over real **FP64** tensors with arbitrary
  strides. Index classes: free-in-A (I), free-in-B (J), contracted (K), batch (L).
- TBLIS-like API using index labels, e.g. `contract("abcd", A, "dbe", B, "ace", C)`.
- Ozaki-II engine with a configurable modulus count N, fast mode, and auto mode
  (choose N from the requested accuracy).
- Two back ends sharing one front end (index analysis, scaling, CRT math):
  - **CPU reference / portable**: scalar + AVX-VNNI micro-kernel, BLIS loop nest.
  - **CUDA sm_89**: the performance target.
- An accuracy test harness and a benchmark harness (see §6).

### Later
- FP32 emulation (fewer moduli), complex (ZGEMM-style), accurate mode.
- AMX / SME / newer-GPU (sm_90/100/120) back ends.
- GETT-style per-contraction code generation and autotuning.
- Python bindings (NumPy / CuPy / DLPack).

### Non-goals
- Contraction-path optimization for more than two operands (leave to opt_einsum, cotengra).
- Sparse or block-sparse tensors, distributed memory, autodiff.

---

## 4. Architecture sketch

```
contract(labels, A, B, C)
  └─ index analysis      → I/J/K/L groups, fused strides, block-scatter vectors
  └─ scaling pre-pass    → per-(I,L) exponents for A, per-(J,L) for B   [one read of A and B]
  └─ backend
       ├─ pack: strided read → scale → trunc → N residue panels (int8)
       ├─ int8 kernel: N products (int32)
       └─ epilogue: mod m_l → CRT accumulate → unscale → α/β → scatter into C
```

The scaling pre-pass can't be fused into packing: each row scale needs the max over
**all** of K, but packing works one kc-slab at a time. That costs one extra streaming
read of A and B (`O(mk+kn)`), which is cheap next to the GEMM work at useful sizes.

### The central design tension: where the moduli loop lives

The CRT step needs each `C_l` summed over the **full** K before reconstruction. Three options:

1. **Unfused (GEMMul8-like):** N separate INT8 GEMMs into INT32 workspaces, then one CRT
   pass. Simple. Workspace is `4·N·m·n` bytes (~56 B per C element at N=14). Chunking
   over n bounds it.
2. **Block-fused:** keep a bounded INT32 workspace per C macro-tile (`mc×nc×N`), loop K and
   the moduli for that tile, then run the CRT epilogue. On CPU this means reordering BLIS
   loops so the `pc` (K) loop sits inside the C-tile loops.
3. **Register-fused (GPU):** a CTA owns a C tile and already loops over the full K. It
   runs the moduli sequentially, and after each one folds `C_l` into a two-word CRT
   accumulator held in registers. This saves the whole INT32 round-trip, but register
   pressure limits tile size (e.g. 64 C elements/thread × 2 doubles). It must be measured.

**Plan:** build option 1 first as the correctness baseline, then 2 on CPU and 3 on GPU.

---

## 5. Milestones

| #  | Milestone | Exit criterion |
|----|-----------|----------------|
| M0 | Scaffolding: CMake, C++20, CUDA optional, CI, tensor view type, naive einsum reference in double-double | Reference agrees with MPFR/`__float128` spot checks |
| M1 | Ozaki-II **matrix** DGEMM, CPU, unfused, scalar int8 | Error vs N and dynamic range φ matches the paper's curves |
| M2 | Tensor front end: labels → I/J/K/L, block-scatter packing fused with residue generation, scattered CRT epilogue | Random contractions (rank ≤ 6, random perms/strides) pass vs reference |
| M3 | CPU AVX-VNNI micro-kernel + block-fused loop nest | Correct; throughput reported against int8 peak (not expected to beat DGEMM) |
| M4 | CUDA sm_89 back end (CuTe/CUTLASS or hand-written mma.sync), unfused, then register-fused | Matrix case at ≥ GEMMul8-level TFLOPS; tensor case within ~10–20% of the matrix case at equal m,n,k |
| M5 | Auto-N, FP32 emulation, complex | Configurable target accuracy |

M4 compares against both cuBLAS DGEMM (native) and cuTENSOR FP64 (native tensor), so we
answer the real question: "is this faster than just calling cuTENSOR?"

---

## 6. Testing and benchmarking discipline

**Accuracy tests:**
- Inputs `(rand − 0.5)·exp(φ·randn)` with φ swept from 0 to about 4, as in the Ozaki
  literature. Report max componentwise error relative to `|A||B|`, and normwise error.
- Adversarial cases: zero rows, Inf/NaN propagation, subnormals, huge dynamic range
  within one row, K near the INT32 overflow limit, negative and zero strides, aliasing.

**Benchmarks: the machine must be quiet first.** Before any timed run, a harness script:
- checks load average and top CPU and GPU consumers; refuses to run (or asks) if it isn't quiet,
- checks GPU utilization, clocks, temperature, and other compute processes (`nvidia-smi`),
- pins CPU runs to P-cores only (`taskset`) and records governor and turbo state,
- warms up, repeats, and reports median and spread, not best-of,
- writes the conditions (load, clocks, temps, driver, commit) next to every result.

---

## 7. Open decisions

1. **Primary target:** GPU-first (the speedup lives there) or CPU-first (purer TBLIS lineage,
   easier to debug)? *Recommendation:* front end + CPU reference first (M0–M2), then go
   straight to CUDA (M4). Push M3 further out.
2. **Build on vs. from scratch:** fork TBLIS/BLIS packing infrastructure, or write a slim
   BLIS-style nest ourselves? TBLIS's kernel config assumes float/double/complex, so adding
   int8 + N-residue packing would be deep surgery. *Recommendation:* from scratch, keeping
   TBLIS's block-scatter idea.
3. **GPU kernel layer:** CUTLASS/CuTe (has GETT examples, rank-N layouts) vs. hand-written
   `mma.sync`. *Recommendation:* CuTe.
4. **Language/API surface:** C++20 header + a C ABI, with Python later?
5. **License.**

## References
- Ozaki, Uchino, Imamura. *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique.* arXiv:2504.08009.
- Uchino, Ozaki, Imamura. *Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit.* IJHPCA 39(3), 2025.
- Uchino et al. *High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines.* arXiv:2508.03984.
- Matthews. *High-Performance Tensor Contraction without Transposition.* SIAM J. Sci. Comput. 2018 (TBLIS).
- Springer, Bientinesi. *Design of a High-Performance GEMM-like Tensor-Tensor Multiplication.* ACM TOMS 2018 (GETT).
