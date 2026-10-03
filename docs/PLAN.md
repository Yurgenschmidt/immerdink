# immerdink — plan

What is built, in what order, and what "done" means for each step. The design itself is
in [DESIGN.md](DESIGN.md), memory/time formulas in [MEMORY_MODEL.md](MEMORY_MODEL.md),
and background context in [CC_NOTES.md](CC_NOTES.md).

Ground rules for every milestone:
- **Spec before code.** A design change lands in DESIGN.md and the affected TLA+ spec,
  with models passing, before any implementation change.
- **Quiet machine for benchmarks.** Timing runs go through `tools/run_bench.sh`. It
  serializes runs machine-wide, waits for `tools/quiet_check.sh`, and records the
  conditions (`QUIET_LOG`) together with the commit and env.
- Correctness tests (`tools/run_quiet.sh`) and TLC (`spec/check.sh`) are pass/fail, so they
  run ungated and in parallel.
- **Bitwise oracle.** Every GPU result is compared **bit-for-bit** with `spec/oracle.py`.
  The emulated path is exact integer arithmetic plus specified roundings, so equality is
  the right test, not a tolerance.
- Work lands on `dev`; `main` changes only by PR.
- Keep it simple. Where GEMMul8, the Ozaki papers or CUTLASS already solved something,
  reuse that approach. Novelty is not a goal; a good, general TBLIS-style Ozaki-II kernel is.

---

## S — Specification ✅ (done 2026-10-02)

| Artifact | Checks | Status |
|---|---|---|
| `spec/OzakiCRT.tla` | residues, int8 store, saturating/havoc INT32 accumulator, K-chunk carry, CRT uniqueness | 13 models ✅ |
| `spec/CRTRealization.tla` | division-free running CRT refines exact CRT (exact → word → limbs) | 21 models ✅ |
| `spec/OzakiScaling.tla` | exact-integer shift rule, RN/TRUNC, norm budget ⇒ CRT precondition, error bounds, single final rounding | 41 models ✅ |
| `spec/IndexMap.tla` | lowering, legality and status precedence, `Scatter∘GEMM∘Gather = einsum`, injectivity, fusion | 18 models ✅ |
| `spec/Pipeline.tla` | K0–K5 ordering, buffer reuse, exactly-once writes, read rules, exact workspace layout | 28 models ✅ |
| `spec/params.py` | real-width certificate for every constant and bound | 24/24 ✅ |
| `spec/oracle.py` | exact bit-level reference of DESIGN §3, the oracle for M1+ | suite passes ✅ |

Every artifact went through adversarial review with mutation testing. Every surviving
mutant was then killed by strengthening the models or tests. The review found real design
bugs, now fixed in DESIGN v1.1:
- a false error-bound inequality;
- an unsound fusion rule;
- an under-specified epilogue mod;
- a missing per-batch modulus source.

Spec follow-ups, to do before or alongside M2/M3 (see `spec/README.md`):
- the operand-role swap in IndexMap;
- workspace aliasing and MISALIGNED statuses;
- K-empty → DIRECT in Pipeline and the oracle.

## L — Lean formalization ✅ (done 2026-10-02)

- `proofs/` Lake project (Lean 4 + Mathlib) proving PROOFS.md's L1–L4: shift rule ⇒ norm
  budget, running CRT, error contract, K-chunk carry.
- Statements use exact arithmetic only; the final FP rounding is a hypothesis.

**Exit:** all four lemmas machine-checked with no `sorry`. A CI target builds `proofs/`.
**Status:** met. `tools/check_proofs.sh` passes; 48 headline theorems use only the standard axioms.

## M0 — Scaffolding

- Dedicated conda env `immerdink` (`environment.yml`).
- CMake + CUDA 13.4, `-gencode arch=compute_89,code=sm_89`, C++17 for CUTLASS TUs.
- Vendor CUTLASS v4.8.0 (`098de2a`) as a git submodule, include path only.
- `params.py --emit-header` generates the device constant tables. No hand-written constants.
- Host types: tensor descriptor, status codes, the C ABI skeleton from DESIGN §6.
- A test runner that feeds random cases to both the library and `oracle.py` and compares
  bits.

**Exit:** builds clean; `imd_workspace_size` matches `params.workspace_size` on a grid of
shapes; CI-able test target runs under `run_quiet.sh`.

## M1 — Matrix Ozaki-II on the GPU (canonical layouts only)

Kernels:
- K1 scale: shifts plus non-finite detection.
- K2/K3 pack: row-major A and column-major B only.
- K4: CUTLASS 2.x batched int8 TN GEMM with the Sm80EVT mod→u8 epilogue (store-mod and
  add-mod).
- K5: CRT, unscale, α/β, and the native path for flagged elements.

The mod reduction is verified **exhaustively** over every reachable int32 accumulator for
every modulus, on CPU against the oracle.

**Exit:**
- Bitwise equal to the oracle for random m, n, k ∈ [1, 3000], N ∈ [2, 20], RN/TRUNC,
  α/β special cases, non-finite inputs, and K-chunking at k > 131056.
- Accuracy matches the oracle's φ-sweep tables.

## M2 — Tensor front end (IndexMap)

- Label lowering per DESIGN §5.
- Int64 scatter vectors.
- Gather-pack with shared-memory transposes for interleaved permutations.
- Scatter in the CRT kernel, coalesced along C's fastest label.
- Legality checks with statuses.

**Exit:**
- Bitwise equal to the oracle on random contractions: ranks ≤ 6, random permutations,
  negative and zero strides, A/B diagonals, single-operand labels.
- Every illegal case returns the right status.
- The CC fixtures (`acQ,bdQ->acbd`, `ijdkQ,ldQ->klij`, `kjbc,kiac->abij`, the six (T)
  permutations) pass.

## M3 — Robustness

- Batch-launch splitting (gridDim.z ≤ 65535).
- DIRECT kernel for K-empty contractions.
- C-vs-A/B aliasing check.
- Zero-size and degenerate shapes.

**Exit:**
- Measured peak device memory equals `imd_workspace_size` exactly (cudaMemGetInfo
  bracketing).
- Results are bitwise identical across K-chunk lengths and residue groups.
- Every DESIGN §3.7/§5 edge case behaves as specified.

## M4 — Performance

Work order, by expected payoff:
1. Gather-pack at coalesced bandwidth on interleaved permutations. This is the
   TBLIS-style kernel's main value.
2. Epilogue mod cost (magic multiply, 128-bit stores).
3. Tile and stage selection by wave quantization.
4. CRT kernel (multi-word integer, N u8 reads per element).

**Baselines**, all run under `run_bench.sh`, reporting median and spread plus machine
conditions:
- native DGEMM (cuBLAS, `CUBLAS_COMPUTE_64F_PEDANTIC`, so env vars can't turn it into
  emulation);
- cuBLAS 13.7 fixed-point emulation (`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`);
- cuBLASLt int8 TN GEMM (the GEMM-phase roofline);
- GEMMul8 v3.5.2 (MIT; build in scratch);
- cuTENSOR 2.7 FP64 contraction.

Note that cuTENSOR's `CUTENSOR_COMPUTE_DESC_8XINT8` is Hopper+ only, so it is not
available on sm_89.

**Exit:**
- 8192³ matrix at N = 14 at or above GEMMul8's measured 12.85 TFLOPS-equivalent (4090
  Laptop). The desktop 4090 should exceed it.
- Permuted tensor contractions within 15% of the matrix-case throughput at equal m, n, k.
- Each phase reported as a % of its MEMORY_MODEL §5 lower bound.

## M5 — Real-workload check (optional)

- Run a few real contractions (CC_NOTES fixtures) and one small end-to-end calculation
  against FP64, as a sanity check of the accuracy model.

## Later (designed-for, not scheduled)

- Internal tiling within a workspace budget.
- Packed-operand reuse API.
- Complex via the 2M method (the specs are already parametric in the moduli table).
- Adaptive / bounded-auto N.
- Accurate-mode scaling.
- Python package (nanobind).
- Hopper/Blackwell back ends.
- Fused variants (MEMORY_MODEL §7).

---

## Testing matrix (applies from M1 on)

| Layer | Method |
|---|---|
| constants / bounds | `params.py --check` (exact integers) |
| design logic | TLC models (`spec/check.sh`) |
| numerics | bitwise vs `oracle.py`; oracle vs exact `Fraction` reference for error bounds |
| epilogue mod reduction | exhaustive over the reachable int32 range × every modulus |
| legality | IndexMap-derived case table: every status code exercised |
| memory | measured peak = formula, exactly |
| performance | `run_bench.sh`; median plus spread; conditions, commit and env recorded |

---

## Resolved setup decisions (2026-10-02)

- **Lean:** elan 4.2.4 / Lean 4.34.1 / Lake 5.0 are in `~/.elan/bin`, which is not on the
  non-interactive PATH; scripts use the full path.
- **Environment:** the `immerdink` conda env exists (`environment.yml`). CUTLASS v4.8.0 is
  a submodule. Apalache 0.62.3 installs via `tools/get_apalache.sh`.
- **License:** MIT.
