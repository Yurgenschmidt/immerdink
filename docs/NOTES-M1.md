# M1 notes: the first end-to-end emulated contraction

DESIGN.md is unchanged. This file records what M1 built, what is verified, and the
design gaps found on the way, each with the choice the code made and a concrete proposal.

## What M1 built

| Piece | Where | Notes |
|---|---|---|
| §5 steps 8 and 10: canonical orders, int64 scatter vectors | `src/index_map.cpp` (`build_scatter_vectors`) | Post-swap frame (see gap 1). Step 9 (fusion) is a TODO(M2); it is performance-only. |
| Plan creation | `src/api.cpp` (`imd_plan_create`) | Pinned scatter vectors (`cudaMallocHost`), §3.2 right-hand side `R·2^F` (≤ 217 bits) built on the host, RN acceptance, dispatch path. |
| K0 | `src/contract.cu` | `cudaMemcpyAsync` of the pinned vectors to `ws.vec`. |
| K1 | `src/contract.cu`, `src/kernels/numerics.cuh` | One block per row/column: exponent-field flags, γ, exact `S` in `unsigned __int128`, `u` by the floor-shift bit-length estimate plus one 256-bit compare. |
| K2/K3 | same | Gather through the scatter vectors, exact RN-even/TRUNC quantization by bit manipulation, residues for the group's moduli, zero padding. |
| K4 | `src/gemm_k4.cu` | The M0 CUTLASS 2.x GEMM + `VisitorModP`, per group and K-chunk (store-mod, then add-mod), batch count split at 65535 (`VisitorModP::batch0`), n̂ split at gridDim.y. One tile (128×256×64, 3 stages). |
| K5 | `src/contract.cu`, `src/kernels/numerics.cuh` | Running CRT in L32 u32 limbs (templated on L32), `(q·W) mod p` by the §3.5 `umulhi64` constant, centering, one correctly rounded conversion; native `__fma_rn` path for flagged elements; α/β; β-only mode. |
| DIRECT | `src/contract.cu` (`k_direct`) | Native fma dot in canonical k̄ order, acc = +0, for K-empty and mode DIRECT. |
| Host/device mod helpers | `src/device/modp.hpp` | The same code runs in the kernels and in the CPU exhaustive test. |
| Status | `include/immerdink/immerdink.h` | `IMD_STATUS_ALLOC_FAILED = 7` (appended so existing values keep their numbers). |

The oracle (`spec/oracle.py`) gained the tensor front end: `lower()` (§5 steps 1-8, 10 and
rows 1-5), `contract_tensor()` (batching, swap, SCALE / DIRECT / GEMM dispatch) and residue
planes in the trace. `spec/test_oracle.py` (8) checks it: on 400 random descriptors
`Scatter∘GEMM∘Gather` reproduces the einsum's address multiset exactly, and orders are
sorted as specified. `spec/params.py` has a new certificate `cert_crt_y_reduction`.

## Verification (all through `tools/test.sh`, about 55 s)

- `contract_*` (`tests/test_contract.py`): every case runs the library on the GPU and the
  oracle on the same bits and compares the **whole C buffer bit for bit** (including
  elements C does not address). For the emulated path it also compares the workspace
  **stage by stage**: the vec region (K0), σ/τ and flags (K1), the residues of the last group
  with their zero padding (K2/K3), and all N W planes including the padding columns (K4).
  The workspace is poisoned with 0xA5 first.
  - `stages`: 28 random matrices, every N in {2,3,8,9,14,16,20}, RN and TRUNC, zero rows and
    columns, non-finite entries.
  - `e2e`: 100 random cases (m, n ≤ 40 or, one case in four, m, n, k up to 3000 within
    the oracle's time budget), random g, layouts with negative and padded strides, data
    from uniform to the full exponent range (huge, tiny, subnormal), non-finite
    entries, α ∈ {1, 0, other}, β ∈ {0, 1, other}, NaN in C with β = 0, mode DIRECT; then
    explicit α/β special cases (α = −0 too).
  - `tensor`: `ikl,kjl->ijl` (two layouts), a permuted output that triggers the role swap,
    `acQ,bdQ->acbd`, negative/padded strides with β ≠ 0, a diagonal plus an A-only label,
    dropped extent-1 modes, K-empty outer product and batched Hadamard (DIRECT dispatch),
    mode DIRECT, k = 0, empty output, a constructed case where the canonical K order decides
    between −Inf and NaN in the native path (both orders), and b = 3300 at g = N = 20
    (66000 GEMM batches: the batch-count split).
  - `bigk`: k = 131093 (2 chunks), k = 262212 (3 chunks, N = 9, g = 4, NaN entry) and
    k = 262117 (3 chunks, N = 2 TRUNC).
  - `replay`: every matrix contraction made by the oracle's own hand-built tests (edges,
    non-finite, determinism, adversarial A1–A8 and the φ × k × N sweep: 1849 cases) replayed
    on the GPU with stage checks. This covers the shift-rule boundary rows, the subnormal
    double-rounding traps, the single β fma and the chunk split that random data never hits.
  - `api`: plan-time statuses (RN acceptance on/off, GemmCoord limit, empty output).
- `mod_exhaustive_cpu`: the kernels' `mod_p_epilogue` over **every** reachable accumulator
  value of every modulus (6.4·10¹⁰ values) and `mod_p_small` over all n < 2¹⁶, against the
  oracle's non-negative residue, on all CPU threads (≈ 2.5 s).
- `c_api`: the C ABI from C on device buffers, including the §6 status order.
- Mutation checks run during development (reverted): reading C when β = 0 and ignoring
  the batch offset in `VisitorModP` are both caught.

**Bit-exactness.** Every compared output is bit-identical to the oracle, with one stated
exception: NaN payloads (and NaN signs) are compared as "both NaN", because DESIGN does not
specify them and the GPU's fma produces its own canonical NaN. No other difference was
observed.

Not exercised by tests: the n̂ split at gridDim.y (needs n̂ > 65535·256), shapes whose
CUTLASS operands exceed 2³¹ elements per batch, and `compute-sanitizer` (not installed in
the env).

## Design gaps and proposals

1. **Canonical orders vs the role swap (§5 step 8).** The order keys use A, B, m and n, and
   the swap exchanges them, but DESIGN does not say which comes first. It matters only for
   the K order when m = n (big = A if m ≥ n): pre-swap and post-swap "A" differ. The
   order is observable only in the native path, when a finite partial sum overflows.
   **Choice:** swap first, then compute every order in the post-swap frame. "Position" is
   the first occurrence in the caller's `A.labels ++ B.labels ++ C.labels`.
   **Proposal:** state both in §5 step 8 and add the swap to `spec/IndexMap.tla`, which does
   not model it yet.

2. **NaN payloads.** DESIGN specifies when a result is NaN but not which NaN.
   **Proposal:** add to §1: "NaN results are NaN; their payload and sign are unspecified".
   Bitwise determinism then holds modulo NaN payloads. In practice the payload is
   deterministic on a given GPU anyway.

3. **Null-pointer rule (§6 contract order, step 1).** It does not say which pointers must
   be non-null. **Choice:** `plan` and `C` always. A and B only when they are read
   (α ≠ 0 and k ≠ 0). `ws` only when `workspace_size > 0`. This check runs before the
   empty-output check, so an empty output with `C = NULL` is INVALID_VALUE.
   **Proposal:** state exactly this.

4. **Row 5 GemmCoord limit.** DESIGN v1.1 applies it "in EMULATE mode with a non-empty
   output", which includes K-empty and k = 0, where no GEMM runs. The code follows the
   text literally. **Proposal:** use the RN-acceptance predicate instead (EMULATE,
   non-empty, k ≥ 2). Then `imd_workspace_size` and `imd_plan_create` accept every shape
   that never reaches CUTLASS.

5. **RN acceptance and α.** The rule is applied at plan time, before α is known, so a plan
   with κ ≥ T is rejected even if it will only be called with α = 0. That is the only
   possible plan-time reading. The matrix oracle `contract()` also rejects before its
   α = 0 branch, and `contract_tensor()` matches the library. **Proposal:** in §3.2 and §6,
   "applies when the plan's GEMM view would run the emulated GEMM (EMULATE, non-empty,
   k ≥ 2), independent of α".

6. **gridDim.y.** §4 "Launch limits" mentions only gridDim.z (batch count). With the identity
   swizzle, N tiles go to gridDim.y ≤ 65535, so with 256-wide tiles n̂ > 16.7M fails.
   **Choice:** K4 also splits n̂ (B and W pointer offsets, multiples of 16).
   **Proposal:** add "n̂ is split so that ⌈n̂_part/tile_N⌉ ≤ 65535" to §4.

7. **K1 "reads A, B once".** The exact S needs γ before any term is summed:
   `Σ ceil(t_h·2^−s) ≠ ceil(Σ ceil(t_h)·2^−s)`, so per-thread partial sums cannot be
   rescaled exactly. A single streaming pass is therefore impossible without buffering the
   row. **Choice (M1):** two passes (γ, then S); the second read mostly hits L1/L2.
   **Proposal:** §4 / MEMORY_MODEL say "reads A, B once from DRAM (the row is held in
   shared memory or re-read from L2)", and M4 caches the row in shared memory when it fits.

8. **Plan lifetime.** K0 copies from the plan's pinned buffer asynchronously, so the plan
   must outlive every `imd_contract` enqueued with it. The header now says so.
   **Proposal:** add to §6: "destroying a plan with work in flight is undefined; synchronize
   the stream first".

9. **ALLOC_FAILED mapping.** `cudaMallocHost` returning `cudaErrorMemoryAllocation` →
   ALLOC_FAILED; any other CUDA error from it → CUDA_ERROR; `std::bad_alloc` anywhere in
   the host API → ALLOC_FAILED (also `imd_create` and `imd_workspace_size`, whose lowering
   uses small `std::vector`s). **Proposal:** state the mapping in §6.

10. **Host-side thread safety.** CUTLASS's `GemmUniversalBase` caches device properties in
    static members on first use, without a lock. Concurrent first calls of `imd_contract`
    from several host threads race on that cache. **Proposal:** M4 instantiates and
    initializes the GEMM operators once in `imd_create`, which also removes the per-launch
    host overhead; until then, document "first call from one thread".

11. **Centering realization.** §3.6 describes comparing the sign bit with the borrow-out
    (signed-difference form). The code compares `x > H` limb-wise and forms `P − x` for
    the magnitude. This is the same function (`spec/CRTRealization.tla` exact layer); only
    the word-level realization differs. No change proposed. If DESIGN means to prescribe
    one realization, say "e.g.".

## What M2 / M3 / M4 should do first

- **M2.** (a) Spec first: add the swap and the post-swap order frame (gap 1) to
  `IndexMap.tla`. (b) Fusion (step 9) and a shared-memory-transposed gather-pack. The pack
  kernels now read A/B through `v_k[h]` with one thread per `h`, which is uncoalesced
  whenever K is not A's fastest axis. (c) Extend `tests/test_contract.py` with a random
  rank-≤6 descriptor generator (`spec/test_oracle.py`'s `rand_desc_triple` is a start) and
  the CC fixtures. The oracle's `contract_tensor` already handles any legal descriptor.
- **M3.** (a) The ALIASING check (§5 row 6): the TODO in `imd_contract`. (b) A test of the
  n̂ split, with a numpy-vectorized check of sampled columns, because the oracle is too slow
  at n̂ > 16.7M. (c) `cudaMemGetInfo` peak-memory bracketing. (d) `compute-sanitizer`
  initcheck/racecheck once it is installed. The batch-count split and the DIRECT kernel
  already exist and are tested.
- **M4.** In payoff order: the gather-pack (above); the K4 tile menu, plus caching the
  CUTLASS operators in the handle (gap 10); K1 single DRAM read (gap 7); K5 with
  register-resident limbs and the W reads of a warp coalesced along j (already the case),
  plus vectorized C stores when C's fastest stride is 1.
