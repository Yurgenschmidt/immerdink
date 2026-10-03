# M2 / M3 notes: tensor front end and robustness

DESIGN.md is unchanged. This file records what M2 and M3 built, how it is verified, and
the design issues found, each with the choice the code made and a proposal.

## What was built

| Piece | Where | Notes |
|---|---|---|
| §5 step 8 canonical orders, post-swap frame | `src/index_map.cpp` (`lower`) | Swap first, then orders with caller-label positions (DESIGN v1.1). Orders are now computed in `lower`, so the lowering hook and the plan see the same thing. |
| §5 step 9 fusion | same | Merge consecutive labels iff `s_X(g') = s_X(g)·e(g)` for all three tensors (a tensor that does not carry the group has stride 0 on both sides, so testing all three is the corrected rule, including stride-0 carriers). Scatter vectors are built from the fused modes (step 10). |
| Oracle fusion | `spec/oracle.py` (`lower`: `v.fused`) | Same rule; `spec/test_oracle.py` `check_fusion` proves the fused enumeration equals the label-by-label one, runs are chains and maximal, and the v1 rule's counterexample (A-only label next to a K label) does not fuse. |
| K2/K3 tiled gather-pack | `src/contract.cu` (`k23_pack`, `PackEmitter`) | 64 rows × 64 k̄ tile in shared memory. Reads are coalesced along the operand's smallest-\|stride\| mode: `pack_pivot` (`src/index_map.cpp`) picks a row/batch mode (tile rows `ρ = lo + radix·d`) when its stride is below the k̄ step, else k̄. Writes are 16-byte K-contiguous int8 vectors per modulus. `PackEmitter` is the only part that knows about residues (an Ozaki-I slice mode would replace it). |
| K5 scatter | same (`k5_crt`) | Consecutive threads take consecutive j̄, which after the swap runs along C's fastest label (when that label is in I or J). |
| Legality | `src/index_map.cpp`, `src/api.cpp` | Full §5 table rows 1-5 in precedence order (unchanged from M1 apart from the GemmCoord predicate, now `EMULATE ∧ non-empty ∧ k ≥ 2`, matching DESIGN v1.1); rows 6-7 at execute time. |
| ALIASING, MISALIGNED, INSUFFICIENT_WORKSPACE | `src/api.cpp` (`imd_contract`) | §6 order: null pointers → empty output → ALIASING → MISALIGNED → workspace size. Byte intervals in 128-bit arithmetic: C∩A, C∩B (also when α = 0 or k = 0), `[ws, ws+workspace_size)` ∩ A, B, C. A null operand (legal only when unread) has no bytes. |
| Launch-limit override (tests only) | `src/plan.hpp` (`LaunchLimits`), `src/gemm_k4.cu`, `src/test_hooks.cpp` | `imd_test_set_limits(k_chunk, max_batch, max_n_tiles)`; production values are the hardware limits and are only changed by this hook. |
| Lowering dump (tests only) | `src/test_hooks.cpp` (`imd_test_lowering`) | swap, m n k b, canonical orders, fused modes, pack pivots, for comparison with the oracle. |

## Verification

All through `tools/test.sh` (22 ctest tests). New:

- `tests/alloc_shim.c`: an `LD_PRELOAD` shim counting CUDA runtime allocations
  (`cudaMalloc*`, `cudaMallocHost`, `cudaHostAlloc`, async/pool). Every `contract_*` test
  runs with it preloaded and fails if `imd_contract` allocates. Interposition is real: the
  plan's one `cudaMallocHost` shows up in the count.
- `run_case` (`tests/test_contract.py`) now surrounds the workspace with 4 KiB poisoned
  guard zones on both sides and fails if any guard byte changes; so the library touches
  only `[ws, ws + workspace_size)`. It also has hard host/device size guards (64 MiB per
  operand buffer, 512 MiB workspace, a view-size check before the Python oracle runs) and
  the harness refuses to create a plan that would pin more than 64 MiB.
- `tests/test_frontend.py`:
  - `contract_lowering`: 3000 random descriptors (a third with "aligned" layouts that
    fuse), some mutated into illegal ones; the library's lowering (swap, m n k b, orders,
    fused modes) equals the oracle's, and the plan status equals the oracle's.
  - `contract_legality`: every §5 row 1-5 case, the precedence between rows, the
    mode-dependent GemmCoord rule, §4a uint64 overflow, malformed descriptors and options.
  - `contract_random`: 150 random legal contractions, every operand of rank ≤ 6 (a
    coverage choice, not a library limit): random permutations, negative / zero / padded
    strides in A and B, padded / negative C, diagonals, A-only / B-only / C-only labels,
    batch labels, extent 0 / 1, random N, g, rounding, α, β, mode. Requires both pack
    tilings (along rows with radix > 1, along k̄) to be exercised.
  - `contract_fixtures`: `acQ,bdQ->acbd`, `ijdkQ,ldQ->klij`, `kjbc,kiac->abij` (dense and
    with permuted / negative / padded layouts) and `abd,dc->X` for all six X; RN and TRUNC.
  - `contract_robust`: 30+ execute-time status checks in the §6 order (a failing call must
    not touch C), A == B in one buffer, `cudaMemGetInfo` bracketing, the n̂ split over
    gridDim.y (one 256-column tile per launch), the batch-count split, K-chunks of 16-128,
    bitwise invariance across g ∈ {1, 2, 3, N/2, N−1, N} and K-chunk lengths, and 19
    degenerate shapes (rank-0 tensors, k = 0 / 1, m / n / b = 0, C-only extents 0 / 1,
    stride-0 A, skinny 4000×1).
  - `contract_highrank`: operand ranks 6-12 (up to 15 distinct labels), extents mostly
    2-3 with some 1s; see below.
- `compute-sanitizer` (found in the conda package cache,
  `~/miniforge3/pkgs/cuda-sanitizer-api-13.4.92-*/bin`, not on PATH): memcheck and
  initcheck on `test_frontend.py` robust + random + highrank + fixtures and initcheck on
  `test_contract.py` tensor + stages: 0 errors. A probe with an undersized C buffer
  confirmed memcheck reports invalid writes in this setup.
- Mutation checks (reverted): fusion ignoring C's strides (caught by `contract_lowering`),
  dropping the ws∩C or C∩B aliasing test (caught by `contract_robust`), an off-by-one in
  the tile's k̄ bound (caught by the K2/K3 zero-padding stage check in `contract_random`).

### High-rank suite (`contract_highrank`)

40 drawn cases + 5 CCSD(T)-style outputs × {EMULATE, DIRECT} = 50 contractions, fixed seed
1212. Named cases: rank 8 × 8 → 8 (twice), rank 10 with an A-only label, rank 10 with a B
diagonal, rank 12 × 12 with all extents 2 (diagonals in both operands, A-only and B-only
labels), rank 12 × 11 all-2 in DIRECT mode, a K-empty rank-9 case (C rank 13, DIRECT
dispatch), extent-0 K (k = 0) and extent-0 L (no-op), and `ijae,ekbc->X` for five
permutations X of `abcijk`. Every tensor's label order and storage order are independent
random permutations; layouts are re-drawn until no group with ≥ 2 labels fuses into one
mode, and the test fails if the library's fused groups collapse in more than 10% of the
non-empty cases (observed: 0 of 39).

Hard guards, asserted per case before it runs: every tensor ≤ 2^15 elements (≤ 2^17
buffer elements with padding), b·m·n·k ≤ 2^20, workspace ≤ 64 MiB. Measured on this box:
about 1.5 s wall, peak RSS about 190 MB (`/usr/bin/time -v`, run inside a 1 GiB
`systemd-run` scope in `immerdink.slice`).

## Design issues and proposals

1. **Null check precedes the empty-output check (§6).** For an empty output with α ≠ 0 and
   k ≠ 0, A and B must still be non-null although nothing is read. The code follows the
   §6 order literally (tested). **Proposal:** keep; say in §6 that step 1 uses the plan's
   k and α only, not whether the output is empty.

2. **"C ∩ A checked even when k = 0" is vacuous when the zero is in A.** k = 0 comes from
   an extent-0 K label; if that label is in A, A has no bytes and cannot overlap. The check
   is meaningful only when the zero extent is a B-only label (A non-empty, unread). Tested
   both ways. **Proposal:** §5 row 6: "byte intervals of the non-empty tensors; checked
   even when A or B is not read (α = 0 or k = 0)".

3. **Byte-interval aliasing is conservative.** Interleaved but disjoint tensors (e.g. C on
   the even and A on the odd elements of one buffer) are rejected as ALIASING. That is
   what §5 says. **Proposal:** keep for v1; an exact lattice test is possible later if a
   user needs interleaved operands.

4. **K5 coalescing when C's fastest label is a batch (L) label.** The swap only chooses
   between I and J. With C's smallest stride on an L label, K5's j̄-fastest thread mapping
   writes C with a large stride. Correct, but slow. **Proposal (M4):** let K5 choose its
   thread-fastest axis (j̄, ī or λ) from C's fastest label, as the pack does with
   `pack_pivot`.

5. **K1 reads are not tiled.** K1 still reads each A row / B column through `v_k[h]`, so it
   is uncoalesced when k̄ is not the operand's fastest axis. **Proposal (M4):** run K1 over
   the same tiles as the pack (two passes per tile row: γ, then S), or fold it into the
   pack's read phase with a shared-memory row cache (NOTES-M1 gap 7).

6. **Pack tile waste with a late pivot.** When the pivot is a row/batch mode with a large
   radix and a small `D = b·rows/radix`, most of the 64 tile rows are empty. Correct, but
   wasteful. **Proposal (M4):** only pivot along rows when D ≥ the tile height, else pick
   the next-smallest stride, or size tiles from (D, k̂).

7. **Test hooks in the production library.** `imd_test_set_limits` and
   `imd_test_lowering` are exported (not in the public header). The limits are one
   process-global, unsynchronized; production never changes them. **Proposal:** build them
   only when `IMMERDINK_BUILD_TESTS=ON`, or move them to a test-only shared object.

8. **The oracle materializes scatter vectors.** `oracle.lower()` builds every scatter
   vector as a Python list, so lowering a legal 2^31-row descriptor needs tens of GB.
   (During M2/M3 a legality test did exactly that and reached 20-30 GB before the memory
   cap killed it.) The tests now guard it (`oracle_status` refuses views with a vector
   longer than 2^20 entries; host-buffer, view and pinned-plan guards in the harness).
   **Proposal:** give `oracle.lower` a `build_vec=False` option, or build `v.vec` lazily.

9. **Allocation check covers the runtime API.** The shim interposes CUDA runtime
   allocation entry points; the library's dynamic symbol table imports only
   `cudaMallocHost` (plan creation) and no driver-API allocator. **Proposal:** add an `nm`
   check of the library's undefined symbols to the test suite if a future change could
   allocate through the driver API.

## Not done / follow-ups

- `spec/IndexMap.tla` still does not model the swap (spec agent's file; NOTES-M1 gap 1).
- The n̂ split is tested with the test-only limit (one 256-column tile per launch), not at
  a real n̂ > 16.7M (the oracle cannot check that size).
- `compute-sanitizer` is not on PATH; it was run by hand from the conda package cache, and
  racecheck was not run.
- No performance work: tile shapes, K1 tiling and the K5 mapping are M4 (issues 4-6).
