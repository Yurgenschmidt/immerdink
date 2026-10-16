# Issue 17: the Hopper K4 with wgmma — design record and offline checks

Status: **design and offline checks done; independent review of 2026-10-15 held and its findings folded in (last section)**.
The review gates all wgmma code in issue 22. Written for a reader new to the project.

## The problem in plain words

K4 is the INT8 matrix-multiply kernel. On every call, for each of the N moduli, it multiplies two INT8 matrices
and reduces the 32-bit sums modulo that modulus's prime, writing one byte per element. It takes 77% of an
8192³ call on an H100 and runs on the *older* tensor-core instruction `mma.sync`. Hopper has a newer one,
`wgmma` (warpgroup matrix multiply, asynchronous), which is fed by a hardware tile copier (TMA) and
synchronised with shared-memory barriers (`mbarrier`). Issue 16 measures whether `wgmma` can be fast enough to
matter; if it can, issue 22 builds the kernel. `wgmma` cannot run on the RTX 4090, so every mistake found
on the cluster costs a queue wait. This issue settles everything that can be settled offline first.

## Decisions

**D1. Main loop: CUTLASS 4.8's Hopper main loop, unchanged.** (Fallback to a hand-written one only if a design
check fails.) The check was: can CUTLASS's `CollectiveBuilder` main loop (TMA + wgmma + mbarrier pipeline,
persistent, warp-specialised) be combined with *our* epilogue? **It can.** All 15 variants of
`src/k4/hopper/k4_hopper_variants.def` (tiles 128×256×128, 128×256×64, 128×128×128, 256×128×128; clusters 1×1,
1×2, 2×1; cooperative and ping-pong schedules; store-mod and add-mod; two epilogue bodies) compile for sm_90a,
contain `IGMMA` (wgmma) and `UTMALDG` (TMA load) instructions, and 13 have no register spills
(`tools/check_hopper_variants.sh`; table in "Results"). The other two (ping-pong 128×128, the generic and the
fast epilogue) spill 8 bytes: **CUTLASS's own stock ping-pong 128×128 kernel with its own epilogue spills the
same 8 bytes**, so it is a property of that tile in CUTLASS 4.8 / ptxas 13.4, not of our epilogue.

**D2. Epilogue: ours, through CUTLASS's own extension point.** `K4ModEpilogue` (`src/k4/hopper/k4_hopper.cuh`) has the shape
of CUTLASS's `DefaultEpilogue` and is wrapped in `Sm90TmaWarpSpecializedAdapter`. It receives the accumulators in
registers and the tile's coordinate (the batch coordinate z selects the modulus, `l = slot0 + z / b`, as in the sm80
K4) and writes the u8 residues of W to global memory from registers. It uses **no shared memory**, so the stages left for the main loop are the same as with no epilogue, and the producer warp keeps
loading while the consumers run the epilogue: in the cooperative kernel the stages are released before the epilogue, so the producer
prefetches up to S stages of the next tile (the same prefetch as the sm80 A2 kernel). That is **not** the same as hiding the epilogue:
both consumer warpgroups are in the epilogue then, so the tensor cores idle for its duration (a few percent at 64 k-tiles, little at
1024). In the ping-pong kernel one ordered barrier serialises the epilogues (mainloop 0, mainloop 1, epilogue 0, epilogue 1) although
ours needs no such protection, so epilogue 1 overlaps no MMA. Whether this matters is measured, not assumed: R1's `imd_ceiling` times
the stock kernels with `int8` and with `int32` output, and issue 22 times the real one.
Two bodies:
* *generic*: one byte store per accumulator at the coordinate CuTe's `partition_C` gives. Correct for any TiledMma.
  Slow. The reference.
* *fast* (`HopperEpilogue`, `src/k4/hopper/epilogue_wg.cuh`): the quad-transposed 128-bit stores of the sm80 kernel.
  It is the sm80 epilogue with one change: where the sm80 code indexes its accumulators in `mma.sync` order, this
  indexes them in `wgmma` order. A `static_assert` compares the actual TiledMma's accumulator layout with the wgmma
  order, so a tile shape for which the assumption fails does not compile.

**D3. K tile 128 bytes.** For an int8 K-major operand one 128-byte row is exactly CuTe's `SW128` swizzle atom
(8 rows × 128 B, the XOR swizzle TMA writes and the wgmma descriptor reads; conflict-free for the eight rows,
`tests/test_wgmma_layout.py`). A 128-wide K tile is 4 wgmma (k32 each) per stage and half as many barrier operations
as K = 64. The cost: stages are twice as big (a 128×256×128 stage is 48 KiB: 4 stages in the 227 KiB of an H100).
K = 64 stays in the variant matrix for 128×256 (R1's ceiling numbers decide). The K chunk of the library,
K_c' = 131008 = 1023.5 × 128, is not a multiple of 128; TMA zero-fills the out-of-range half of the last tile, which
adds zeros to the sum. For **correctness** TMA needs a 16-byte base, strides that are multiples of 16 bytes and the extent set to the
chunk's length, all of which hold (a chunk starts at a multiple of 64 bytes). It is **not** shown that there is no *speed* rule like the sm80
kernel's (DESIGN §4b: 23.5 against 32 TFLOPS from the row phase of cp.async): a 128-byte box starting at an address that is only 16-byte aligned
straddles two cache lines, and since 131008 = 2047 × 64 every odd chunk starts at phase 64 mod 128. Rung 5 times odd against even chunks. If
there is a penalty, K_c = 130944 (1023 × 128) for Hopper is the remedy; it gives the same W but changes the certificates and the launch counts.

**D4. Schedule: cooperative first.** Two consumer warpgroups share every tile (64 rows each): 128 accumulator
registers per thread at 128×256. Ping-pong (each warpgroup its own tile, alternating) stays in the matrix because the
model (below) shows what it needs: an *order barrier* between the warpgroups' mainloops. That is not only a
utilisation device: a parity wait passes whenever the barrier's phase bit differs, so a warpgroup that waits a round
early passes at once. TLC finds this, and CUTLASS has the barrier.

**D5. TMA descriptors: stock.** CUTLASS builds them from the problem shape and strides. Our packed residue buffers
satisfy TMA's rules already: base addresses 256-byte aligned (DESIGN §4a), row strides k̂ a multiple of 16 bytes, the
batch as the L mode (stride rows × k̂), a K chunk as a pointer offset (a multiple of 16 bytes) with its own extent.
TMA descriptors embed the base pointers and extents of A and B, so they **cannot live in the plan's K4 parameter blobs** (the plan is
shared by threads and calls; DESIGN §6, HostAPI spec): issue 22 encodes them per launch, on the stack, which also changes the plan host-byte
accounting and the launch counts. Rung 0 and issue 22's ladder test the rest on the card.

**D6. Accumulation.** wgmma accumulates in int32 without saturation in the CuTe atom CUTLASS picks
(`SM90_64xNx32_S32S8S8_SS_TN`; the `_SATURATE` atoms exist). The sm80 kernel runs `mma.sync .satfinite`. The two agree
wherever the sum stays in int32, and DESIGN §2/§3.5 (`cert_kc`, `ChunkFits`) prove it does for every chunk of
K_c' = 131008. **Open question for the reviewer** (Q4): is that enough, or should the SATURATE atom be used to keep the
two kernels the same function on out-of-contract inputs too?

**D7. What does not change.** The launch structure (n̂ parts, K chunks, z parts, the modulus per batch), W's layout, the
epilogue arithmetic (Barrett with the §3.5 constants, p = 256 by mask), the planner's inputs. Only the instruction stream
of the GEMM and the register order of the accumulators.

## The hand-off protocol is a TLA+ model

`spec/WgmmaPipeline.tla` (+ `MC_WgmmaPipeline__*.cfg`; `spec/check.sh WgmmaPipeline`). A persistent CTA: a producer
(TMA), NC consumer warpgroups (wgmma, up to `Depth` in flight), S stages, a FULL and an EMPTY `mbarrier` per stage with
parity waits, cooperative or ping-pong, optionally a cluster of two CTAs with TMA multicast. It follows CUTLASS's
`PipelineTmaAsync`. Checked: a copy never lands in a stage a consumer holds or a wgmma still reads (`NoOverwrite`); a
wgmma reads exactly the iteration's data, all halves landed (`ReadsCorrect`); the accumulation is exactly k = 0..KT-1 once
each (`AccExact`); no deadlock before every tile is done.

| models | count | result |
|---|---|---|
| positive (cooperative at 1–4 stages with Depth 0 and 1, tiles of 1 to 3 k-steps, ping-pong, cluster of 2 with each; three of them wrap each stage three times) | 13 | all pass |
| negative: producer ignores EMPTY; consumers ignore FULL; EMPTY expects one arrival too few; stage released before its wgmma retired; producer parity wrong; consumer parity wrong; EMPTY counts only the local CTA's consumers (cluster); ping-pong consumer does not skip the other warpgroup's iterations; ping-pong without the order barrier; last stage of a tile never released; accumulator not reset; stage released twice; Depth ≥ stages | 13 | each fails with its named invariant or `Deadlock reached` |
| witnesses (pipeline full, two wgmma in flight, a peer's half lands first, two warpgroups read one stage) | 4 | reachable |

Design rules that fell out: **Depth ≤ stages − 1** (with one stage and one wgmma in flight the loop deadlocks);
**the order barrier is required** in ping-pong (above); **a cluster's EMPTY barrier counts the consumers of both CTAs**.
Assumptions: sequentially consistent memory, no timing, one producer thread, the asynchronous-proxy fences that
CUTLASS issues are taken as given. The model does not prove CUTLASS's code; it checks the protocol we depend on and
that we do not need to change it.

## Layouts are committed tests

`tests/test_wgmma_layout.py` compiles `tests/wgmma_layout_dump.cpp` against CUTLASS's own CuTe tables and checks:
1. **wgmma accumulator** (thread t, register v) → row 16 (t/32) + (t%32)/4 + 8 ((v/2)%2), column 8 (v/4) + 2 (t%4) + v%2
   for all 8 int8 atoms CuTe defines (N = 8, 16, 32, 64, 96, 128, 192, 256), and a bijection onto 64×N for *every* N = 8..256
   step 8 (the PTX range).
2. **K4's epilogue on wgmma-ordered registers**: a Python copy of the sm80 epilogue's data movement (byte packing, 4×4 quad
   transpose, the 128-bit store) run on registers placed by the dump stores the right 16 bytes of every row, for N = 64, 128,
   192, 256. (Mutants — one butterfly stage dropped, row halves swapped — are killed.)
3. **Shared-memory swizzle** of K-major int8 tiles: SW128/SW64/SW32 equal `lin xor (((lin >> 7) mod W/16) << 4)`, are
   bijections, and the eight rows of an SW128 atom put the same 16-byte chunk in eight different bank groups.

`tests/test_hopper_epilogue.cu` (ctest `hopper_epilogue`, **runs on the RTX 4090**) is the "permuted `mma.sync` layout"
test of the issue: a random integer matrix is placed in the registers of a warpgroup in wgmma order and `HopperEpilogue` runs
on it; the same matrix is placed in the registers of K4's `mma.sync` fragment order and K4's own epilogue runs on it. Both
planes must be identical and equal to the host reference, for all 20 moduli (p = 256 included), store and add mode, full and
partial tiles, 64-, 128- and 256-column pieces: 174 cases.

## Rung 0 (optional in R1)

`src/k4/hopper/rung0.cu` (`tools/build_rung0.sh OUT run`) runs six prototype kernels (generic and fast epilogue; 128×256 and
128×128 tiles; cooperative with clusters 1×1 and 2×1; ping-pong) on a small problem against the exact CPU result, in store and add
mode, for three sets of moduli. It tells the three situations apart (summary at the top of the file). It was **compiled for
sm_90a and never run**: this machine has no Hopper. It runs only if reviewed (this review), and nothing in R1 depends on it.

## Variant matrix and test ladder for issue 22

*Variants* (`k4_hopper_variants.def`): tile {128×256×128 primary, 128×256×64, 128×128×128, 256×128×128} × schedule {cooperative,
ping-pong} × cluster {1×1, 1×2, 2×1} × epilogue {generic (debug only), fast} × {store-mod, add-mod}. R1's ceiling program (`imd_ceiling`,
issue 16) times the stock kernels of the same shapes with `int8` and `int32` output; shapes that cannot reach the ceiling are dropped
before R2.

*Ladder* (each rung is its own process; a failure skips only the rungs above it):
0. layout known answer (`rung0`, above) — also runs in R1;
1. one tile, one k-step, one modulus, vs the CPU;
2. one tile, k-steps = stages + 1 (the pipeline wraps) and 2 × stages + 1;
3. several tiles per CTA (more tiles than SMs: the persistent loop and the order barrier);
4. a batch of moduli (z → l), p = 256, store then add;
5. the K-chunk boundary (131008 and one half tile past it), zero-filled residue; odd against even chunks (phase 64 against 0 mod 128) for speed;
6. K4's whole launch structure (n̂ parts, z parts, K chunks) against the sm80 kernel: `W` byte-identical; then the library tests
   (`tools/test.sh`, the committed output hashes of issue 19) with the Hopper K4 selected;
7. speed, against `imd_ceiling` and GEMMul8 (campaign R2).

## Results (2026-10-15)

* TLC: 30 models as expected (13 positive, 13 negative, 4 witnesses; the largest 51 000 states; three positives go three rounds round the stages).
* `tests/test_wgmma_layout.py`: passes; `tests/test_hopper_epilogue.cu` on the RTX 4090: 174/174; both fail when the epilogue's
  row halves are swapped.
* `tools/check_hopper_variants.sh` (nvcc 13.4, CUTLASS 4.8.0, one translation unit per variant):

| variant | regs | spills | IGMMA | UTMALDG |
|---|---|---|---|---|
| g/f 128×256×128 1×2 coop, store | 168 | 0 | 8 | 2 |
| g/f 128×256×128 1×2 coop, add | 168 | 0 | 8 | 2 |
| g/f 128×256×128 1×1 coop, store | 168 | 0 | 8 | 2 |
| g/f 128×128×128 2×1 coop, store (f also add) | 168 | 0 | 8 | 2 |
| g/f 128×256×64 1×2 coop, store | 168 | 0 | 4 | 2 |
| g/f 256×128×128 2×1 coop, store | 168 | 0 | 16 | 2 |
| g/f 128×128×128 1×1 ping, store | 168 | 8 B (stock) | 16 | 2 |

(168 is the per-thread count of the whole kernel; consumers raise it to 232 at run time with `setmaxnreg`, as CUTLASS does.)

## Not verified, and what could go wrong

* Nothing here ran on a Hopper card. Whether wgmma's accumulators really have the CuTe layout is CUTLASS's claim; the run of rung 0 and
  the ladder of issue 22 are the check.
* The register order of *repeated* atoms (a tile taller than one atom per warpgroup, ping-pong 128×128: two repetitions down M) is
  assumed to be CuTe's `(MMA, MMA_M, MMA_N)` order, value fastest. The `static_assert` covers one atom's layout; the repetition order is
  covered only by rung 0's `f_128x128x128_1x1_ping`.
* Performance. No number in this note says the Hopper K4 is fast. Issue 16's go/no-go rule decides whether issue 22 happens at all.
* The two ping-pong 128×128 variants spill 8 bytes (stock behaviour); if that tile is chosen, issue 22 re-checks it.
* The model abstracts timing and the memory model; it does not see register pressure, clusters beyond two CTAs, or TMA descriptor errors.

## Questions for the reviewer

1. Is the "CUTLASS main loop first, own epilogue through the adapter" conclusion sound, or is there a reason to expect that
   path to break once a *real* problem (batched, chunked, persistent over many tiles) runs?
2. D3 (K tile 128): is a half-zero last tile an acceptable cost at K_c' = 131008, and is there any TMA alignment rule we miss?
3. The fast epilogue assumes one atom per thread along M or (ping-pong) a repetition order. Is the evidence (static layout
   check + rung 0) enough to proceed, or should the repetition order also be checked offline?
4. D6: non-saturating accumulation.
5. Are the ladder and the variant matrix complete for issue 22?
6. Anything in the TLA+ model that makes a positive result vacuous?

## Review outcome (independent reviewer, 2026-10-15)

Verdict: approve the path (CUTLASS main loop plus our register epilogue), with these changes, made here unless marked.

* **Defect: wrong modulus on split launches.** `K4Args` lacked the launch's first batch; the sm80 kernel uses `(batch0 + z) / b`. Fixed (`batch0`).
  rung 0 now runs with batch0 = 0, 1, 3 and two batches per modulus; before it could not have seen this.
* **Defect: TMA descriptors per launch, not in the plan.** In D5 above; a requirement on issue 22.
* **Defect: `can_implement` was `true`.** The fast path now requires a 16-byte W base, ldw and plane stride multiples of 16, n a multiple of 16.
* **Q3 (is the layout evidence enough?): no — closed offline.** `tests/test_hopper_partition.cu` (ctest `hopper_partition`, host code, built for sm_90a)
  evaluates CuTe's `partition_C` for the *real* TiledMma of every fast variant (including 256×128 cooperative and 128×128 ping-pong, where each thread
  holds two repetitions) and compares every register with the closed form the epilogue uses: 212 992 registers, all equal; a mutant (the repetition
  stride) is caught. What stays for rung 0 is hardware: that the wgmma really writes CuTe's layout.
* **Q4: non-saturating accumulation agrees.** Stronger than the note's argument: 16384 × 131008 = 2 146 435 072 < 2^31 for *every* int8 input,
  and adding W gives at most 2 146 435 327. So the two kernels are the same function on all int8 inputs. rung 0 has the extreme case (A = B = −128
  over a whole chunk, store and add).
* **Overlap**: restated in D2 (above).
* **Ladder additions for issue 22** (all added to the list below): M and N residues together with add mode; b > 1 and batch0 > 0; k̂ ≡ 16 and 48
  mod 128 and chunks that start at phase 64; kc = 16 and a half-tile K; a packed-B layout whose batch stride is not n̂·k̂ (gemm_k4.cu:119–131);
  the stage count forced to 2 and 3 (hazards hide at 4); repeated-run determinism and compute-sanitizer; ping-pong with an odd number of tiles per CTA
  (one warpgroup gets none); a cluster grid that is not a multiple of the cluster shape. Variant dimensions added: stage count and raster / swizzle
  (L2 reuse at 8192³; the sm80 kernel used a swizzle-8 order).
* **rung 0 gaps** closed: disjoint modulus sets (the 256 modulus sits at index 0, so the old sets repeated), eight k-tiles (the pipeline wraps),
  chunk boundaries at 448 and 1024 (a half-zero tile; a start at phase 64), two different fill bytes, 256×128 and 1×2-cluster variants.
* **PDL.** CUTLASS's cooperative kernel calls `launch_dependent_grids` before the epilogue and its consumers never wait on the previous grid. Safe with
  plain launches; if programmatic dependent launch is ever used between K chunks, add-mode reads of W need an explicit wait.
* **Model**: scope widened (three-round positives). Known abstractions: multicast is symmetric for both operands (really one operand), release is atomic
  per warpgroup (really one thread per destination CTA), the ping-pong order barrier is weaker than CUTLASS's (it also orders the epilogues), so the
  positive results are conservative there. `held` is redundant with `rd` outside the two mutants.
* **The compile check is weak by itself.** IGMMA and UTMALDG present says the instructions exist, not that the code is right; correctness comes from rung 0 and
  the ladder.
